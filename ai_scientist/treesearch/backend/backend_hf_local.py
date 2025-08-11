# 기존 query 시그니처를 아래처럼 교체/확장

from typing import Tuple, Any, List, Dict, Optional
import time
from ai_scientist.llm import create_client as create_local_client, get_batch_responses_from_llm

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

def _as_messages(
    user_message: Optional[str],
    msg_history: Optional[List[Dict[str, str]]],
    messages: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """backend_* 호출 케이스를 통합해서 messages 형태로 정규화."""
    if messages:  # 이미 messages가 준비되어 있는 경우
        return messages
    hist = msg_history or []
    if user_message is None:
        # hist만 있는 경우(드묾): 그대로 반환
        return hist
    # history + 마지막 user 메시지로 구성
    return [*hist, {"role": "user", "content": user_message}]

def query(
    # 케이스 A: anthropic/openai 백엔드가 쓰던 형태
    user_message: Optional[str] = None,
    system_message: str = "",
    model: str = "qwen-7b",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n: int = 1,
    seed: int = 0,
    msg_history: Optional[List[Dict[str, Any]]] = None,
    # 케이스 B: 이미 messages로 들어오는 형태
    messages: Optional[List[Dict[str, Any]]] = None,
    # 기타 키워드(툴콜, json_mode 등) 무시해도 안정 동작
    **kwargs,
) -> Tuple[List[str], float, int, int, Dict[str, Any]]:
    """
    HF 로컬 백엔드. anthropic/openai 스타일 모두 수용:
    - A) user_message + msg_history(+system_message)
    - B) messages + system_message
    반환: (outputs, req_time, in_tok, out_tok, info)
    """
    t0 = time.time()

    # messages 정규화
    msgs = _as_messages(user_message=user_message, msg_history=msg_history, messages=messages)

    # prompt/히스토리 분리
    if len(msgs) == 0:
        prompt = ""
        hist = []
    else:
        # 마지막 user/assistant 여부 상관없이 마지막 content를 prompt로 쓰되
        # 일반적으로는 마지막이 user여야 자연스러움
        prompt = msgs[-1].get("content", "")
        hist = msgs[:-1]

    client, repo_id = create_local_client(model)

    contents, histories = get_batch_responses_from_llm(
        prompt=prompt,
        client=client,
        model=repo_id,
        system_message=system_message or "",
        msg_history=hist,
        temperature=temperature,
        n_responses=n,
    )

    # 토큰 카운트는 아직 미집계(필요 시 tokenizer로 계산 가능)
    in_tok = 0
    out_tok = 0
    info = {"backend": "hf_local", "model": repo_id}

    return contents, (time.time() - t0), in_tok, out_tok, info
