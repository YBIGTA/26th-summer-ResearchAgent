import os
import re
import json
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# (선택) 기존 토큰 카운터가 있다면 유지, 없다면 no-op로 대체
try:
    from ai_scientist.utils.token_tracker import track_token_usage
except Exception:
    def track_token_usage(f):
        return f

MAX_NUM_TOKENS = 4096  # 생성 토큰 상한 (모델/VRAM에 맞게 조절)

HF_REPO_MAP: Dict[str, str] = {
    "qwen-7b": "Qwen/Qwen2.5-3B-Instruct",
    "gpt-oss-20b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",  
    "LGAI-EXAONE"  : "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "Qwen3-Embedding" : "Qwen/Qwen3-Embedding-0.6B",
}


AVAILABLE_LLMS = list(HF_REPO_MAP.keys())


# 로컬 로딩 옵
@dataclass
class LocalModelConfig:
    repo_id: str
    dtype: Optional[str] = "bfloat16"    # "float16", "bfloat16", None
    device_map: str = "auto"             # "auto" 권장 (Accelerate 필요)
    load_in_8bit: bool = False           # bitsandbytes 필요
    load_in_4bit: bool = False           # bitsandbytes 필요
    trust_remote_code: bool = True       # 일부 모델은 필요 (Qwen 등)
    rope_scaling: Optional[dict] = None  # 긴 컨텍스트 필요 시
    # generation 기본값
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    do_sample: bool = True

@dataclass
class LocalClient:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    gen_cfg: GenerationConfig

def _str_to_dtype(dtype: Optional[str]):
    if dtype is None:
        return None
    dt = dtype.lower()
    if dt == "float16" or dt == "fp16":
        return torch.float16
    if dt == "bfloat16" or dt == "bf16":
        return torch.bfloat16
    if dt == "float32" or dt == "fp32":
        return torch.float32
    return None

def _ensure_repo_id(model_name_or_repo: str) -> str:
    # "qwen-7b" 같은 키가 들어오면 매핑, 아니면 그대로 repo_id로 간주
    return HF_REPO_MAP.get(model_name_or_repo, model_name_or_repo)

def _build_chat_prompt(
    tokenizer: AutoTokenizer,
    system_message: str,
    msg_history: Optional[List[Dict[str, Any]]],
    user_msg: str,
) -> Dict[str, Any]:
    """
    Hugging Face chat template 우선 사용. 없으면 간단한 수동 포맷.
    msg_history 형식: [{"role": "user"/"assistant", "content": "..."}]
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    if msg_history:
        messages.extend(msg_history)

    messages.append({"role": "user", "content": user_msg})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"text": prompt, "messages": messages}
    else:
        # 단순 접착식 프롬프트 (템플릿 없음)
        joined = []
        if system_message:
            joined.append(f"[SYSTEM]\n{system_message}\n")
        if msg_history:
            for m in msg_history:
                role = m.get("role", "user")
                content = m.get("content", "")
                joined.append(f"[{role.upper()}]\n{content}\n")
        joined.append(f"[USER]\n{user_msg}\n[ASSISTANT]\n")
        return {"text": "\n".join(joined), "messages": messages}

def create_client(model_name_or_repo: str,
                  config: Optional[LocalModelConfig] = None) -> Tuple[LocalClient, str]:
    """
    OpenAI/Anthropic client 대신 로컬 HF 모델 로더를 반환.
    model_name_or_repo: "qwen-7b" 같은 별칭 또는 HF repo_id
    """
    repo_id = _ensure_repo_id(model_name_or_repo)
    cfg = config or LocalModelConfig(repo_id=repo_id)

    dtype = _str_to_dtype(cfg.dtype)
    model_kwargs = {
        "trust_remote_code": cfg.trust_remote_code,
        "device_map": cfg.device_map,
    }
    if cfg.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    elif cfg.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if cfg.rope_scaling is not None:
        model_kwargs["rope_scaling"] = cfg.rope_scaling

    print(f"Loading local model: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=cfg.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)

    gen_cfg = GenerationConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return LocalClient(tokenizer, model, gen_cfg), repo_id

def get_response_from_llm(
    prompt: str,
    client: LocalClient,
    model: str,  # 호환성 유지용(미사용)
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
) -> Tuple[str, List[Dict[str, Any]]]:
    
    try:
        gen_cfg = copy.deepcopy(client.gen_cfg)
    except Exception:
        # 일부 환경에서 deepcopy가 문제면 dict 기반 복제
        gen_cfg = GenerationConfig.from_dict(client.gen_cfg.to_dict())

    # 런타임 파라미터 오버라이드
    gen_cfg.temperature = temperature

    built = _build_chat_prompt(client.tokenizer, system_message, msg_history, prompt)
    input_text = built["text"]
    new_history = built["messages"].copy()

    inputs = client.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(client.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = client.model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            do_sample=gen_cfg.do_sample,
            eos_token_id=gen_cfg.eos_token_id,
            pad_token_id=gen_cfg.pad_token_id,
        )

    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    content = client.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    new_history.append({"role": "assistant", "content": content})
    return content, new_history

def get_batch_responses_from_llm(
    prompt: str,
    client: LocalClient,
    model: str,  # 호환성 유지용(미사용)
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    n_responses: int = 1,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    contents: List[str] = []
    histories: List[List[Dict[str, Any]]] = []

    for _ in range(n_responses):
        c, hist = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=print_debug,
            msg_history=msg_history,
            temperature=temperature,
        )
        contents.append(c)
        histories.append(hist)

    return contents, histories

# utils/json_extract.py (새 파일)

def extract_json_object(text: str):
    if not text:
        raise ValueError("Empty LLM text")

    # 1) ```json ... ``` 우선
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    # 2) 마지막에 등장하는 "가장 바깥 { ... }" 블록 추출
    start_idxs = [i for i,c in enumerate(text) if c == "{"]
    end_idxs   = [i for i,c in enumerate(text) if c == "}"]
    if start_idxs and end_idxs:
        # 뒤에서부터 시도 (노이즈가 적음)
        for s in reversed(start_idxs):
            stack = 0
            for i in range(s, len(text)):
                if text[i] == "{": stack += 1
                elif text[i] == "}":
                    stack -= 1
                    if stack == 0:
                        cand = text[s:i+1]
                        try:
                            return json.loads(cand)
                        except Exception:
                            break  # 다음 s로 재시도

    # 3) top-level 리스트도 허용: [{...}, ...]
    m = re.search(r"```json\s*(\[\s*\{.*?\}\s*\])\s*```", text, flags=re.S)
    if m:
        arr = json.loads(m.group(1))
        if isinstance(arr, list) and arr:
            return arr[0]

    # 4) "choices": [{"...":...}] 형태 보정
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            cand = data["choices"][0]
            if isinstance(cand, dict):
                return cand
    except Exception:
        pass

    raise ValueError("No JSON object found in LLM text")


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
