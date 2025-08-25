# -*- coding: utf-8 -*-
import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from ai_scientist.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
    create_client, ## 처음 업스테이지 client 호출 시 필요
)

# 1) 루브릭 & 가중치 정의

# 1–4 점수(저/중/높음/매우높음). Overall은 1–10
RUBRIC_WEIGHTS = {
    "Originality": 0.15,         # 독창성과 새로운 아이디어
    "ThematicConsistency": 0.12, # 포켓몬 세계관과의 일관성
    "VisualDesign": 0.12,        # 시각적 디자인과 매력도
    "CompetitiveViability": 0.18,# 대전 환경에서의 실용성
    "TypeBalance": 0.15,         # 타입 조합과 밸런스
    "MovesetDesign": 0.10,       # 기술 구성의 합리성
    "EvolutionLogic": 0.08,      # 진화 체계의 논리성
    "Accessibility": 0.05,       # 신규 플레이어 친화성
    "Marketability": 0.03,       # 상품성과 인기 요소
    "LoreIntegration": 0.02,     # 배경 설정과의 통합
}

RUBRIC_KEYS_ORDERED = [
    "Summary",
    "Strengths", 
    "Weaknesses",
    "Originality",
    "ThematicConsistency",
    "VisualDesign", 
    "CompetitiveViability",
    "TypeBalance",
    "MovesetDesign",
    "EvolutionLogic",
    "Accessibility",
    "Marketability",
    "LoreIntegration",
    "WeightedBreakdown",
    "Questions",
    "Limitations",
    "EthicalConcerns",
    "Soundness",
    "Presentation", 
    "Contribution",
    "Overall",
    "Confidence",
    "Decision",
]

# 2) 시스템 프롬프트(기본/페르소나)

REFERENCE_SITES_TEXT = (
    "When grounding judgments, you MAY reference general knowledge patterns from:\n"
    "- Smogon University (competitive analysis, tier lists, usage stats)\n"
    "- Bulbapedia/Serebii (official Pokédex data, move mechanics, type effectiveness)\n"
    "- GamePress/Game8 (PvE viability, raid counters, meta analysis)\n"
    "- Reddit r/TheSilphRoad, r/CompetitivePokemon (community insights)\n"
    "- Official Pokémon websites (design philosophy, canonical lore)\n"
    "Do NOT fabricate specific URLs or quote fake pages; use them only as style/standard references."
)

# 영어 시스템 프롬프트
# reviewer_system_prompt_base = (
#     "You are an expert Pokémon game designer and competitive analyst reviewing a newly proposed Pokémon.\n"
#     "Evaluate this Pokémon's design, competitive viability, and overall fit within the Pokémon universe.\n"
#     "Be thorough, analytical, and consider both casual and competitive perspectives.\n"
#     + REFERENCE_SITES_TEXT
# )

# 한글 시스템 프롬프트
reviewer_system_prompt_base = (
    "당신은 새로운 포켓몬을 검토하는 전문 게임 디자이너이자 대전 분석가입니다.\n"
    "이 포켓몬의 디자인, 대전 적합성, 포켓몬 세계관 내의 일관성을 평가하세요.\n"
    "철저하고 분석적으로 답하며, 캐주얼/경쟁적 관점을 모두 고려하세요.\n"
    "평가할 때는 다음의 일반적인 지식 출처에서의 패턴을 참고해도 됩니다:\n"
    "- Smogon University (대전 분석, 티어 리스트, 사용 통계)\n"
    "- Bulbapedia/Serebii (공식 포켓덱스, 기술 메커니즘, 타입 상성)\n"
    "- GamePress/Game8 (PvE 활용도, 레이드 카운터, 메타 분석)\n"
    "- Reddit r/TheSilphRoad, r/CompetitivePokemon (커뮤니티 인사이트)\n"
    "- 공식 포켓몬 웹사이트 (디자인 철학, 정식 설정)\n"
    "가짜 URL을 만들거나 허위 페이지를 인용하지 마세요. 오직 스타일/기준 참조용으로만 활용하세요."
)

# 부정/긍정 바이어스(선택적으로 사용)
# reviewer_system_prompt_neg = (
#     reviewer_system_prompt_base + "\nIf the Pokémon design is flawed or questionable, assign low scores and Reject."
# )
# reviewer_system_prompt_pos = (
#     reviewer_system_prompt_base + "\nIf the Pokémon design is solid or shows potential, assign high scores and Accept."
# )

reviewer_system_prompt_neg = reviewer_system_prompt_base + "\n만약 포켓몬 디자인에 결함이 있거나 의문점이 크다면, 낮은 점수를 주고 Reject으로 판정하세요."
reviewer_system_prompt_pos = reviewer_system_prompt_base + "\n만약 포켓몬 디자인이 탄탄하거나 가능성이 보인다면, 높은 점수를 주고 Accept로 판정하세요."


# ――― 페르소나 별 시점(여러 평가자 앙상블용)
PERSONA_SYSTEM_PROMPTS = {
    "HardcoreCompetitive": (
        reviewer_system_prompt_base
        + "\n페르소나: 하드코어 경쟁 포켓몬 플레이어. 대회 적합성과 메타 영향력을 최적화합니다."
        + "\n편향: 경쟁 밸런스, 티어 배치 가능성, 전략적 깊이를 최우선시합니다. 과도하게 강하거나 약한 디자인은 감점합니다."
        + "\n전문성: VGC, Smogon 티어, 사용 통계, 팀 빌딩, 대미지 계산."
    ),

    "CreativeDesigner": (
        reviewer_system_prompt_base
        + "\n페르소나: 혁신적인 포켓몬 디자이너. 신선한 콘셉트와 창의적인 메커니즘을 중시합니다."
        + "\n편향: 독특한 타입 조합, 새로운 특성, 창의적인 기술 구성을 높게 평가합니다. 평범하거나 모방적인 디자인은 감점합니다."
        + "\n전문성: 게임 디자인 원리, 타입 상호작용, 특성 메커니즘, 시각적 스토리텔링."
    ),

    "CasualKidPlayer": (
        reviewer_system_prompt_base
        + "\n페르소나: 멋있는 몬스터와 단순하고 재미있는 게임플레이를 좋아하는 초등학생 포켓몬 팬."
        + "\n편향: 시각적 매력, 기억에 남는 특징, 직관적이면서도 신나는 기술 구성을 높게 평가합니다. 지나치게 복잡한 메커니즘은 감점합니다."
        + "\n전문성: 포켓몬을 '멋지게' 만드는 요소, 애니메이션/만화적 매력, 장난감/머천다이즈 흥행성, 접근성."
    ),

    "SeriesVeteran": (
        reviewer_system_prompt_base
        + "\n페르소나: 포켓몬 시리즈의 오랜 팬으로, 세계관의 일관성과 프랜차이즈 전통을 매우 중시합니다."
        + "\n편향: 기존 패턴 준수, 논리적인 진화 라인, 설정과 세계관 구축에 충실한 디자인을 높게 평가합니다. 설정을 깨뜨리는 요소는 감점합니다."
        + "\n전문성: 포켓몬 역사, 지역 테마, 진화 패턴, 공식 설정과의 연속성."
    ),
}

# 3) 템플릿/폼/리플렉션
template_instructions = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>

<THOUGHT>에는 이 포켓몬의 주요 강점과 약점, 대전 잠재력, 디자인 철학, 포켓몬 세계 전체에 대한 기여도를 분석하세요.  
이 디자인이 성공하거나 실패하는 구체적인 이유를 서술하세요.  

<JSON>에는 반드시 아래 순서로 필드를 포함하세요:
"Summary": 포켓몬과 그 컨셉의 간단한 개요  
"Strengths": 긍정적인 측면 리스트  
"Weaknesses": 부정적인 측면이나 우려사항 리스트  
"Originality": 1–4 (얼마나 독창적이고 창의적인가?)  
"ThematicConsistency": 1–4 (포켓몬 세계관과 얼마나 잘 맞는가?)  
"VisualDesign": 1–4 (시각적 매력과 디자인의 일관성)  
"CompetitiveViability": 1–4 (대전에서의 잠재적 영향력)  
"TypeBalance": 1–4 (타입 조합의 밸런스와 효과성)  
"MovesetDesign": 1–4 (기술 구성의 합리성과 품질)  
"EvolutionLogic": 1–4 (진화 라인이 자연스럽고 논리적인가?)  
"Accessibility": 1–4 (신규 플레이어도 쉽게 이해할 수 있는가?)  
"Marketability": 1–4 (상품성·대중적 매력)  
"LoreIntegration": 1–4 (기존 포켓몬 설정과 잘 맞는가?)  
"WeightedBreakdown": 각 평가 항목별 가중 점수 합산(object)  
"Questions": 명확화를 위한 질문 리스트(array)  
"Limitations": 디자인의 한계나 우려사항 리스트(array)  
"EthicalConcerns": boolean (윤리적으로 문제가 있는 요소가 있는가?)  
"Soundness": 1–4 (디자인의 내부적 일관성)  
"Presentation": 1–4 (설계 문서의 명확성)  
"Contribution": 1–4 (포켓몬 라인업에 의미 있는 기여인가?)  
"Overall": 1–10 (전체적인 품질과 채택 권고)  
"Confidence": 1–5 (이 평가에 대한 확신 정도)  
"Decision": "Accept" 또는 "Reject"  

반드시 유효한 JSON을 출력하고, 각 점수는 지정된 숫자 범위를 지켜야 합니다. 점수와 서술이 일관되도록 작성하세요.
"""
pokemon_review_form = """
포켓몬 디자인 리뷰 폼
당신은 새롭게 제안된 포켓몬 디자인을 평가합니다. 평가는 게임 밸런스, 플레이어 흥미, 세계관 일관성을 고려해야 합니다.
평가 기준 (점수 1–4):
독창성과 창의성: 새로운 아이디어를 얼마나 잘 가져왔는가?
세계관 일관성: 포켓몬 세계관, 지역 특성, 전체 톤과 얼마나 잘 맞는가?
시각적 디자인: 디자인이 매력적이고 기억할 만하며, 형태가 뚜렷한가?
경쟁적 활용성: 대전에서 얼마나 쓸모 있고, 밸런스가 맞는가?
타입 밸런스: 타입 조합이 논리적이고 전략적 가치를 제공하는가?
기술 구성: 기술들이 합리적이고 재미있는 패턴을 만드는가?
진화 논리: 진화 라인이 있다면, 그 진행이 자연스럽고 설정에 맞는가?
접근성: 신규 플레이어도 쉽게 이해하고 즐길 수 있는가?
상품성: 머천다이즈, 애니 등에서 인기를 끌 수 있는가?
세계관 통합성: 기존 포켓몬 생태/역사와 잘 맞는가?
평가 시에는 대전 영향, 디자인 선례, 플레이어 심리, 장기적 메타 건강을 고려하세요.
""" + template_instructions
reviewer_reflection_prompt = """라운드 {current_round}/{num_reflections}.
당신의 평가가 점수와 코멘트 사이에서 일관성이 있는지 다시 검토하세요. 고려할 사항은 다음과 같습니다:

- 경쟁 분석과 티어 예측이 현실적인가?  
- 이 디자인이 실제로 포켓몬의 규범과 품질 기준에 부합하는가?  
- 간과된 밸런스 문제나 디자인 결함은 없는가?  
- 개별 점수들이 전체 추천과 일치하는가?  

만약 개선할 점이 없다면, 이전 JSON을 그대로 반복 출력하고 JSON 앞에 "I am done"을 포함하세요.  

응답 형식은 동일하게 유지하세요:
THOUGHT:
<THOUGHT>
REVIEW JSON:
<JSON>
```"""

meta_reviewer_system_prompt = (
    "당신은 수석 포켓몬 게임 디렉터입니다. {reviewer_count}개의 서로 다른 시각(경쟁, 창의, 캐주얼, 시리즈 팬)에서 작성된 디자인 리뷰를 종합하세요. "
    "각 리뷰의 통찰을 균형 잡힌 최종 결론으로 통합하세요."
)

# 4) 유틸: 스코어 정합성/가중합

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_weighted_breakdown_and_overall(review: Dict[str, Any]) -> Dict[str, Any]:
    """1–4 점수를 가중합(0–10 스케일)으로 변환하고 Overall을 산출."""
    wb = {}
    total = 0.0
    for k, w in RUBRIC_WEIGHTS.items():
        raw = review.get(k, None)
        if isinstance(raw, (int, float)):
            raw = _clip(float(raw), 1.0, 4.0)
        else:
            raw = 1.0  # 누락 시 최저점 처리(보수적)
        # 1–4 → 0–1 정규화 후 가중치 적용, 최종 10점 스케일
        norm = (raw - 1.0) / 3.0  # 0~1
        wb[k] = norm * (w * 10.0)
        total += wb[k]
    review["WeightedBreakdown"] = {k: round(v, 2) for k, v in wb.items()}
    # Overall: 1~10, 가중합(0~10)에 바닥 1을 보장하며 클리핑
    overall = _clip(total + 1.0, 1.0, 10.0)  # +1 for base score
    review["Overall"] = int(round(overall))
    return review

def coerce_schema(review: Dict[str, Any]) -> Dict[str, Any]:
    """필수 키 채움 + 범위 강제."""
    # 점수 키 범위 보정 (1-4 범위)
    for key in [
        "Originality", "ThematicConsistency", "VisualDesign", "CompetitiveViability",
        "TypeBalance", "MovesetDesign", "EvolutionLogic", "Accessibility", 
        "Marketability", "LoreIntegration", "Soundness", "Presentation", "Contribution"
    ]:
        if key not in review or not isinstance(review[key], (int, float)):
            review[key] = 1
        review[key] = int(_clip(float(review[key]), 1, 4))

    # Confidence: 1-5 범위
    if "Confidence" not in review or not isinstance(review["Confidence"], (int, float)):
        review["Confidence"] = 3
    review["Confidence"] = int(_clip(float(review["Confidence"]), 1, 5))

    # 불리언/리스트 기본값
    review.setdefault("EthicalConcerns", False)
    review.setdefault("Strengths", [])
    review.setdefault("Weaknesses", [])
    review.setdefault("Questions", [])
    review.setdefault("Limitations", [])
    review.setdefault("Summary", "")

    # Decision 정규화
    decision = str(review.get("Decision", "Reject")).strip().lower()
    review["Decision"] = "Accept" if decision == "accept" else "Reject"

    # Weighted/Overall 재계산
    review = compute_weighted_breakdown_and_overall(review)
    return review

# 5) 리뷰 수행(단일/앙상블/리플렉션)

def _make_review_prompt(pokemon_json_text: str, form_text: str) -> str:
    return form_text + f"""
    아래는 리뷰할 포켓몬 디자인 JSON입니다:
    {pokemon_json_text}
    대전 적합성, 디자인의 일관성, 세계관과의 부합성, 플레이어 매력을 고려하여 철저하게 평가하세요.
    """

def _parse_review_safely(llm_text: str) -> Optional[Dict[str, Any]]:
    try:
        js = extract_json_between_markers(llm_text)
        return js
    except Exception:
        return None

def perform_review(
    text: str,
    model: str,
    client: Any,
    num_reflections: int = 1,
    use_persona_ensemble: bool = True,
    temperature: float = 0.7,
    msg_history: Optional[List[Dict[str, str]]] = None,
    return_msg_history: bool = False,
    reviewer_system_prompt: str = reviewer_system_prompt_neg,
    review_instruction_form: str = pokemon_review_form,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, str]]]]:
    """
    text: Pokémon JSON 문자열
    use_persona_ensemble: True면 4 페르소나로 별도 리뷰 후 메타 집계
    """

    base_prompt = _make_review_prompt(text, review_instruction_form)

    if use_persona_ensemble:
        # 각 페르소나별로 독립 호출
        persona_results = []
        persona_histories = []

        for persona_name, persona_sys in PERSONA_SYSTEM_PROMPTS.items():
            llm_review, hist = get_response_from_llm(
                base_prompt,
                model=model,
                client=client,
                system_message=persona_sys,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            js = _parse_review_safely(llm_review)
            if js:
                js = coerce_schema(js)
                persona_results.append((persona_name, js))
                persona_histories.append(hist)

        if not persona_results:
            # 백업: 단일 프롬프트로라도 생성
            llm_review, hist = get_response_from_llm(
                base_prompt,
                model=model,
                client=client,
                system_message=reviewer_system_prompt,
                print_debug=False,
                msg_history=msg_history,
                temperature=temperature,
            )
            js = _parse_review_safely(llm_review) or {}
            review = coerce_schema(js)
            return (review, hist) if return_msg_history else (review, None)

        # 메타 리뷰: LLM으로 집계 + 점수 평균 반영
        reviews_only = [r for _, r in persona_results]
        meta = get_meta_review(model, client, temperature, reviews_only)
        if not meta:
            meta = {}
            
        # 점수 필드 평균 (가중 평균 고려)
        persona_weights = {
            "HardcoreCompetitive": 0.3,  # 경쟁적 관점 중시
            "CreativeDesigner": 0.25,    # 창의성 중시
            "CasualKidPlayer": 0.25,     # 대중적 어필 중시
            "SeriesVeteran": 0.2,        # 전통과 일관성 중시
        }
        
        for score, limits in [
            ("Originality", (1, 4)),
            ("ThematicConsistency", (1, 4)),
            ("VisualDesign", (1, 4)),
            ("CompetitiveViability", (1, 4)),
            ("TypeBalance", (1, 4)),
            ("MovesetDesign", (1, 4)),
            ("EvolutionLogic", (1, 4)),
            ("Accessibility", (1, 4)),
            ("Marketability", (1, 4)),
            ("LoreIntegration", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            weighted_sum = 0
            total_weight = 0
            for (persona_name, r) in persona_results:
                val = r.get(score)
                if isinstance(val, (int, float)) and limits[0] <= val <= limits[1]:
                    weight = persona_weights.get(persona_name, 0.25)
                    weighted_sum += val * weight
                    total_weight += weight
            
            if total_weight > 0:
                meta[score] = int(round(weighted_sum / total_weight))

        meta.setdefault("Summary", "Consensus review aggregated from multiple expert perspectives.")
        meta.setdefault("Strengths", [])
        meta.setdefault("Weaknesses", [])
        meta.setdefault("Questions", [])
        meta.setdefault("Limitations", [])
        meta.setdefault("EthicalConcerns", any(r.get("EthicalConcerns", False) for r in reviews_only))
        meta.setdefault("Decision", "Accept" if meta.get("Overall", 5) >= 6 else "Reject")

        meta = coerce_schema(meta)

        # 이력 관리
        aggregate_history = []
        if persona_histories:
            aggregate_history = persona_histories[0][:-1] if persona_histories[0] else []
            aggregate_history.append({
                "role": "assistant",
                "content": f"THOUGHT:\nAggregated {len(persona_results)} expert reviews from competitive, creative, casual, and veteran perspectives.\n\nREVIEW JSON:\n```json\n{json.dumps(meta)}\n```"
            })

        # 리플렉션 수행
        if num_reflections > 1:
            for j in range(num_reflections - 1):
                reflection_text, aggregate_history = get_response_from_llm(
                    reviewer_reflection_prompt.format(current_round=j+2, num_reflections=num_reflections),
                    client=client,
                    model=model,
                    system_message=reviewer_system_prompt_base,
                    msg_history=aggregate_history,
                    temperature=temperature,
                )
                js2 = _parse_review_safely(reflection_text)
                if js2:
                    meta = coerce_schema(js2)
                if "I am done" in reflection_text:
                    break

        return (meta, aggregate_history) if return_msg_history else (meta, None)

    else:
        # 단일 시점 리뷰
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = _parse_review_safely(llm_review) or {}
        review = coerce_schema(review)

        if num_reflections > 1:
            for j in range(num_reflections - 1):
                text2, msg_history = get_response_from_llm(
                    reviewer_reflection_prompt.format(current_round=j+2, num_reflections=num_reflections),
                    client=client,
                    model=model,
                    system_message=reviewer_system_prompt,
                    msg_history=msg_history,
                    temperature=temperature,
                )
                js2 = _parse_review_safely(text2)
                if js2:
                    review = coerce_schema(js2)
                if "I am done" in text2:
                    break

        return (review, msg_history) if return_msg_history else (review, None)

# 6) 메타 리뷰(LLM 집계)
def get_meta_review(model: str, client: Any, temperature: float, reviews: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
        리뷰 {i + 1}/{len(reviews)}:
        {json.dumps(r, ensure_ascii=False, indent=2)}
        """
    base_prompt = pokemon_review_form + review_text
    llm_review, _ = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    try:
        meta_review = extract_json_between_markers(llm_review)
        return meta_review
    except Exception:
        return None

# 7) 저장하고 json 처리하는 코드
def load_pokemon_json(json_path: str) -> Dict[str, Any]:
    """포켓몬 JSON 파일 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_review_result(review: Dict[str, Any], output_path: str):
    """리뷰 결과 저장"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

def get_pokemon_tier_prediction(review: Dict[str, Any]) -> str:
    """리뷰 점수를 기반으로 예상 티어 예측"""
    overall = review.get("Overall", 1)
    competitive = review.get("CompetitiveViability", 1)

    if overall >= 9 and competitive >= 4:
        return "OU (OverUsed) - Top Tier"
    elif overall >= 7 and competitive >= 3:
        return "UU (UnderUsed) - High Tier"
    elif overall >= 5 and competitive >= 2:
        return "RU (RarelyUsed) - Mid Tier"
    elif overall >= 3:
        return "NU (NeverUsed) - Low Tier"
    else:
        return "PU (PU) - Bottom Tier"

# 8) 최종 실행 예시
def main():
    # 예시 포켓몬 아이디어 JSON (크롤링 데이터 보고 만들어달라고함 -> 나중에 실제 genereate된 json과 일치시킬것)
    example_pokemon_idea = {
        "idea": {
            "Name": "Voltorrent",
            "Title": "Electric Torrent Pokémon",
            "Short Hypothesis": "Electric/Water type with ability to create electromagnetic whirlpools",
            "Related Work": "Similar to Lanturn and Rotom-Wash but with unique storm generation mechanics",
            "Abstract": (
                "A spherical Pokémon that resembles a fusion of Voltorb and a water vortex. "
                "Can generate powerful electromagnetic storms in aquatic environments. "
                "Ability 'Storm Surge' boosts Electric moves in rain and Water moves in Electric Terrain."
            ),
            "Experiments": (
                "Base stats: 60/75/70/120/80/95 (500 BST). Signature move 'Hydro Surge' - "
                "Electric/Water hybrid move. Access to Thunder, Surf, Rain Dance, Electric Terrain."
            ),
            "Risk Factors and Limitations": (
                "Electric/Water typing creates 4x weakness to Grass. "
                "Storm Surge ability might be too complex for casual players. "
                "Visual design might be too similar to existing round Electric types."
            ),
        }
    }

    # 입력단 문자열화
    text = json.dumps(example_pokemon_idea, ensure_ascii=False)

    # Upstage 클라이언트 생성 (환경변수 UPSTAGE_API_KEY 필요)
    client, client_model = create_client("upstage:solar-1-mini-chat")

    # 리뷰 실행 
    review, _ = perform_review(
        text=text,
        model=client_model,
        client=client,
        num_reflections=1,
        use_persona_ensemble=True, # 페르소나 앙상블 on
        temperature=0.7,
    )

    # 5) 결과 출력 및 저장
    print(json.dumps(review, ensure_ascii=False, indent=2))
    os.makedirs('ai_scientist/ideas/reviews/', exist_ok=True)
    out_path = "ai_scientist/ideas/reviews/kor_tmp_review_result.json"
    save_review_result(review, out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    # env 파일 내부에 UPSTAGE_API_KEY 집어넣을 것. 
    main()
