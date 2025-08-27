import os
import io
import re
import json
import base64
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass

# 이미지 생성을 위한 라이브러리
# pip install diffusers
# from diffusers import StableDiffusionPipeline #  구형 모델
from diffusers import StableDiffusionXLPipeline # <<< SDXL용 파이프라인으로 변경

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
    return 

# 기존 일반 Stable diffusion
# def generate_image_from_prompt(prompt: str, output_path: str) -> str:
#     """
#     Stable Diffusion을 사용하여 프롬프트로부터 이미지를 생성하고 저장합니다.
#     """
#     try:
#         # Stable Diffusion 파이프라인 로드 (GPU 사용 권장)
#         # 1.5 버전은 VRAM 효율이 좋습니다. 'runwayml/stable-diffusion-v1-5'
#         # 또는 더 좋은 모델을 사용하려면 교체하세요. 예: 'stabilityai/stable-diffusion-xl-base-1.0'
#         # 로컬 경로에 모델을 다운로드했다면 해당 경로를 지정할 수 있습니다.
#         print("Loading Stable Diffusion pipeline...")
#         pipe = StableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             torch_dtype=torch.float16,
#         ).to("cuda")

#         print(f"Generating image for prompt: '{prompt}'")
#         image = pipe(prompt).images[0]
        
#         # 이미지 저장
#         image.save(output_path)
#         print(f"Image saved to {output_path}")
#         return output_path

#     except Exception as e:
#         print(f"Image generation failed: {e}")
#         return ""

def generate_image_from_prompt(prompt: str, output_path: str) -> str:
    """
    Animagine XL 3.0을 사용하여 포켓몬 스타일의 이미지를 생성하고 저장합니다.
    """
    try:
        # 1. 애니메이션/포켓몬 스타일에 특화된 고품질 모델로 변경
        # 기존: "runwayml/stable-diffusion-v1-5"
        model_id = "cagliostrolab/animagine-xl-3.0"
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        # ).to("cuda")

        # <<< StableDiffusionPipeline -> StableDiffusionXLPipeline으로 클래스 이름 변경
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        # GPU VRAM이 부족할 경우 아래 주석을 해제하여 메모리 사용량을 줄일 수 있습니다.
        # pipe.enable_model_cpu_offload()

        # 2. 포켓몬 공식 아트워크 스타일을 위한 프롬프트 엔지니어링
        # 사용자의 기본 프롬프트(캐릭터 외형 묘사)에 스타일 키워드를 추가합니다.
        # style_prompt = (
        #     "masterpiece, best quality, official artwork, game character, "
        #     "pokemon style, style of Ken Sugimori, vibrant colors, clean lineart, "
        #     "white background"
        # )

        ## 기존
        style_prompt = (
            "(official Pokémon artwork:1.4), (Ken Sugimori style:1.3), (Ohmura style:1.1), "
            "(single creature:1.4), (full body:1.3), (3/4 view:1.2), (centered:1.2), "
            # "(white background:1.5), (no background:1.5), "
            # "(clean thin black lineart:1.3), (flat cel shading, two-tone shadows:1.3), "
            # "(simple geometric shapes:1.2), (limited color palette 2-3 colors + accent:1.2), "
            # "(cute proportions, big expressive eyes, small mouth:1.2), "
            # "(no clothes, no armor, creature design not human:1.4)"

            "(solid white background:1.6), (plain background:1.4), "
            # "(white background:1.5), (light gray drop shadow:1.1), "
            "(clean thick-thin black lineart:1.3), (flat cel shading:1.3), (two-tone shadows:1.2), "
            "(simple geometric shapes:1.2), (limited 2-3 color palette + accent:1.2), "
            "(cute proportions, big expressive eyes, small mouth:1.1), "
            "(creature design not human:1.5), (matte finish:1.2)"
        )
        full_prompt = f"{style_prompt}, {prompt}, redesign as a creature, readable silhouette, minimal details"

        # full_prompt = f"{style_prompt}, {prompt}"

        # 3. 원치 않는 결과(실사, 3D 등)를 방지하기 위한 네거티브 프롬프트
        negative_prompt = (
            "nsfw, photorealistic, photograph, 3d, rendering, text, watermark, "
            "signature, ugly, blurry, low quality, worst quality, monochrome"
            "human, humanoid, girl, woman, man, realistic anatomy, fingers, hands,"
            "armor, clothing, dress, stockings, heels, cleavage,"
            "angel, angelic, goddess, halo,"
            "photorealistic, photograph, 3d, cgi, rendering,"
            "cinematic lighting, volumetric, bloom, lens flare, bokeh,"
            "magic circle, particles, energy aura, effects, feathers storm,"
            "background, sky, clouds, landscape,"
            "text, logo, watermark, signature,"
            "ugly, lowres, blurry, noisy, worst quality, monochrome"
            # <<< 추가된 부분: 그림자와 회색 배경을 명시적으로 금지합니다.
            "shadows, drop shadow, gray background, gradient background"
        )
        # ## 기존

        # # 2. 포켓몬 공식 아트워크 스타일을 위한 프롬프트 엔지니어링
        # style_prompt = (
        #     "(official Pokémon artwork:1.4), (Ken Sugimori style:1.3), "
        #     "(single creature:1.6), (full body:1.3), (front view:1.3), (centered:1.3), "
        #     "(clear face:1.4), (expressive eyes:1.2), "
        #     "(pure white background:2.2), (solid white background:2.2), (no background:2.0), "
        #     "(clean bold black outline:1.4), (flat cel shading:1.3), (two-tone shadows:1.2), "
        #     "(simple geometric shapes:1.2), (limited 2-3 color palette + 1 accent:1.2), "
        #     "(creature design not human:1.5), (matte finish:1.2)"
        # )

        # full_prompt = f"{style_prompt}, {prompt}, redesign as a creature, readable silhouette, minimal details"


        # # 3. 원치 않는 결과(실사, 3D 등)를 방지하기 위한 네거티브 프롬프트
        # negative_prompt = (
        #     "nsfw, photorealistic, photograph, 3d, rendering, cgi, "
        #     "text, logo, watermark, signature, caption, copyright, words, numbers, "
        #     "title, label, trademark, letters, symbols, "
        #     "background, scenery, landscape, sky, gradient, colored background, gray background, "
        #     "drop shadow, vignette, spotlight, aura, particles, effects, magic circle, "
        #     "human, humanoid, girl, boy, man, woman, realistic anatomy, fingers, hands, "
        #     "clothing, armor, accessories, props, "
        #     "ugly, lowres, blurry, noisy, worst quality"
            
        # )

        print(f"Generating Pokemon-style image for prompt: '{prompt}'")
        
        # 4. 강화된 프롬프트로 이미지 생성
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=28,  # 추론 스텝 수 (품질과 속도 조절)
            guidance_scale=7.5,     # 프롬프트 충실도
        ).images[0]
        
        # 이미지 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"Image generation failed: {e}")
        # traceback.print_exc() # 더 자세한 에러를 보고 싶을 때 주석 해제
        return ""

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
