import os
import io
import re
import json
import base64
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
)

# 기존 데코레이터가 없으면 no-op
try:
    from ai_scientist.utils.token_tracker import track_token_usage
except Exception:
    def track_token_usage(f):
        return f

MAX_NUM_TOKENS = 4096

# 사용 가능한 로컬 VLM 레포 매핑(필요 시 추가)
HF_VLM_REPO_MAP = {
    "qwen2-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",  # 권장
    # LLaVA OneVision (Qwen2-7B 백본) – 환경에 따라 레포가 다를 수 있으므로 실제 사용 레포로 교체
    "llava-onevision-7b": "llava-hf/llava-onevision-qwen2-7b-ov",
}

AVAILABLE_VLMS = list(HF_VLM_REPO_MAP.keys())

def _ensure_repo_id(name_or_repo: str) -> str:
    return HF_VLM_REPO_MAP.get(name_or_repo, name_or_repo)

def _str_to_dtype(dtype: Optional[str]):
    if not dtype:
        return None
    d = dtype.lower()
    if d in ("bfloat16", "bf16"):
        return torch.bfloat16
    if d in ("float16", "fp16"):
        return torch.float16
    if d in ("float32", "fp32"):
        return torch.float32
    return None

@dataclass
class LocalVLMConfig:
    repo_id: str
    dtype: Optional[str] = "bfloat16"
    device_map: str = "auto"          # accelerate 필요
    load_in_4bit: bool = False        # bitsandbytes 필요
    load_in_8bit: bool = False        # bitsandbytes 필요
    trust_remote_code: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    do_sample: bool = True

@dataclass
class LocalVLMClient:
    processor: Any
    model: Any
    gen_cfg: GenerationConfig

def encode_image_to_base64(image_path: str) -> str:
    """(남겨둠) Base64가 필요할 때 사용. 로컬 VLM 경로에서는 PIL 이미지 직접 사용."""
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")

def _open_images(image_paths: List[str], max_images: int) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for p in image_paths[:max_images]:
        img = Image.open(p)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        imgs.append(img)
    return imgs

def _has_chat_template(processor: Any) -> bool:
    # Qwen2-VL 등은 processor.apply_chat_template 제공
    return hasattr(processor, "apply_chat_template")

def prepare_vlm_prompt(
    msg: str,
    image_paths: str | List[str],
    max_images: int,
    system_message: str = "",
    msg_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    HF의 멀티모달 chat template 형식으로 messages를 구성.
    messages 예:
    [
      {"role": "system", "content": "…"},
      {"role": "user", "content": [{"type":"text","text":"…"}, {"type":"image","image":<PIL.Image>}, ...]},
      {"role": "assistant", "content": "…"},
      ...
    ]
    """
    if msg_history is None:
        msg_history = []

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # 히스토리(텍스트 위주) 그대로 반영
    messages: List[Dict[str, Any]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    for m in msg_history:
        # m은 {"role": "user"/"assistant", "content": "..."} 혹은
        # {"role":..., "content":[{"type":"text",...}, {"type":"image",...}]} 형태일 수 있음
        messages.append(m)

    # 현재 입력(텍스트 + 이미지들)
    content: List[Dict[str, Any]] = [{"type": "text", "text": msg}]
    pil_images: List[Image.Image] = _open_images(image_paths, max_images)

    # HF 멀티모달 표준: content에 {"type":"image","image": PIL.Image} 항목을 추가
    for im in pil_images:
        content.append({"type": "image", "image": im})

    messages.append({"role": "user", "content": content})
    return messages, pil_images

@track_token_usage
def create_client(model: str) -> Tuple[LocalVLMClient, str]:
    """
    OpenAI 클라이언트 대신 로컬 HF VLM 로더를 반환.
    model: "qwen2-vl-7b-instruct" 또는 HF repo_id
    """
    repo_id = _ensure_repo_id(model)
    dtype = _str_to_dtype("bfloat16")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    # 원하면 4/8bit로 전환
    # model_kwargs["load_in_4bit"] = True
    # model_kwargs["load_in_8bit"] = True
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    print(f"Loading local VLM: {repo_id}")
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    model_obj = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)

    gen_cfg = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=1024,
        do_sample=True,
        pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
    )
    return LocalVLMClient(processor=processor, model=model_obj, gen_cfg=gen_cfg), repo_id

def _build_inputs(processor, messages, images):
    """
    processor.apply_chat_template가 있으면 텍스트 프롬프트를 만들고,
    processor(...)에 messages/text & images를 함께 전달.
    """
    if _has_chat_template(processor):
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[images] if len(images) > 0 else None,
            return_tensors="pt",
        )
    else:
        # 템플릿이 없다면 간단 포맷으로 묶고 images만 전달
        sys_txt = ""
        if messages and messages[0].get("role") == "system":
            sys_txt = f"[SYSTEM]\n{messages[0]['content']}\n"
            rest = messages[1:]
        else:
            rest = messages

        buf = [sys_txt]
        for m in rest:
            role = m.get("role", "user").upper()
            cont = m.get("content")
            if isinstance(cont, list):
                # 텍스트/이미지 혼합일 수 있음
                txts = [c["text"] for c in cont if c.get("type") == "text"]
                buf.append(f"[{role}]\n" + "\n".join(txts))
            else:
                buf.append(f"[{role}]\n{cont}")
        prompt = "\n".join(buf) + "\n[ASSISTANT]\n"

        inputs = processor(
            text=[prompt],
            images=[images] if len(images) > 0 else None,
            return_tensors="pt",
        )
    return inputs, prompt

@track_token_usage
def make_vlm_call(client: LocalVLMClient, temperature: float, inputs: Dict[str, Any]):
    # 필요 시 온도 등 동적으로 오버라이드
    gen_cfg = client.gen_cfg.clone()
    gen_cfg.temperature = temperature
    with torch.no_grad():
        outputs = client.model.generate(
            **{k: v.to(client.model.device) for k, v in inputs.items()},
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            do_sample=gen_cfg.do_sample,
            eos_token_id=gen_cfg.eos_token_id,
            pad_token_id=gen_cfg.pad_token_id,
        )
    return outputs

@track_token_usage
def get_response_from_vlm(
    msg: str,
    image_paths: str | List[str],
    client: LocalVLMClient,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_images: int = 25,
) -> Tuple[str, List[Dict[str, Any]]]:
    if msg_history is None:
        msg_history = []

    # 메시지/이미지 준비
    messages, images = prepare_vlm_prompt(
        msg=msg,
        image_paths=image_paths,
        max_images=max_images,
        system_message=system_message,
        msg_history=msg_history,
    )

    # 인코딩
    inputs, prompt_text = _build_inputs(client.processor, messages, images)

    # 생성
    outputs = make_vlm_call(client, temperature, inputs)

    # 입력 길이 기반으로 생성분만 디코딩
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[:, input_len:]
    content = client.processor.tokenizer.batch_decode(
        gen_ids, skip_special_tokens=True
    )[0]

    new_msg_history = messages + [{"role": "assistant", "content": content}]

    if print_debug:
        print("\n" + "*" * 20 + " VLM START " + "*" * 20)
        for j, m in enumerate(new_msg_history):
            preview = m["content"]
            if isinstance(preview, list):
                preview = str(preview)
            print(f"{j}, {m['role']}: {preview[:400]}")
        print("*" * 21 + " VLM END " + "*" * 21 + "\n")

    return content, new_msg_history

@track_token_usage
def get_batch_responses_from_vlm(
    msg: str,
    image_paths: str | List[str],
    client: LocalVLMClient,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    n_responses: int = 1,
    max_images: int = 200,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    contents: List[str] = []
    histories: List[List[Dict[str, Any]]] = []
    for _ in range(n_responses):
        c, h = get_response_from_vlm(
            msg=msg,
            image_paths=image_paths,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=print_debug,
            msg_history=msg_history,
            temperature=temperature,
            max_images=max_images,
        )
        contents.append(c)
        histories.append(h)
    return contents, histories

def extract_json_between_markers(llm_output: str) -> Optional[dict]:
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                return json.loads(json_string_clean)
            except json.JSONDecodeError:
                continue
    return None
