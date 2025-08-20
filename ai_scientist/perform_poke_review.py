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

reviewer_system_prompt_base = (
    "You are an expert Pokémon game designer and competitive analyst reviewing a newly proposed Pokémon.\n"
    "Evaluate this Pokémon's design, competitive viability, and overall fit within the Pokémon universe.\n"
    "Be thorough, analytical, and consider both casual and competitive perspectives.\n"
    + REFERENCE_SITES_TEXT
)

# 부정/긍정 바이어스(선택적으로 사용)
reviewer_system_prompt_neg = (
    reviewer_system_prompt_base + "\nIf the Pokémon design is flawed or questionable, assign low scores and Reject."
)
reviewer_system_prompt_pos = (
    reviewer_system_prompt_base + "\nIf the Pokémon design is solid or shows potential, assign high scores and Accept."
)

# ――― 페르소나 별 시점(여러 평가자 앙상블용)
PERSONA_SYSTEM_PROMPTS = {
    "HardcoreCompetitive": (
        reviewer_system_prompt_base
        + "\nPersona: Hardcore competitive Pokémon player who optimizes for tournament viability and meta impact."
        + "\nBias: Prioritize competitive balance, tier placement potential, and strategic depth. Penalize overpowered or underpowered designs."
        + "\nExpertise: VGC, Smogon tiers, usage statistics, team building, damage calculations."
    ),
    
    "CreativeDesigner": (
        reviewer_system_prompt_base
        + "\nPersona: Innovative Pokémon designer who values fresh concepts and creative mechanics."
        + "\nBias: Reward unique type combinations, novel abilities, and creative movesets. Penalize generic or derivative designs."
        + "\nExpertise: Game design principles, type interactions, ability mechanics, visual storytelling."
    ),
    
    "CasualKidPlayer": (
        reviewer_system_prompt_base
        + "\nPersona: Elementary school Pokémon fan who loves cool-looking monsters and simple, fun gameplay."
        + "\nBias: Reward visual appeal, memorable characteristics, and straightforward but exciting moves. Penalize overly complex mechanics."
        + "\nExpertise: What makes Pokémon 'cool', anime/manga appeal, toy/merchandise potential, accessibility."
    ),
    
    "SeriesVeteran": (
        reviewer_system_prompt_base
        + "\nPersona: Long-time Pokémon series fan who deeply cares about lore consistency and franchise tradition."
        + "\nBias: Reward adherence to established patterns, logical evolution lines, and world-building. Penalize lore-breaking elements."
        + "\nExpertise: Pokémon history, regional themes, evolution patterns, canonical precedents, series continuity."
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

In <THOUGHT>, analyze this Pokémon's key strengths and weaknesses, competitive potential, design philosophy, and overall contribution to the Pokémon universe. Be specific about what makes this design work or fail.
In <JSON>, provide fields in this exact order:
"Summary": Brief overview of the Pokémon and its concept
"Strengths": List of positive aspects
"Weaknesses": List of negative aspects or concerns
"Originality": 1–4 (how unique and creative is this design?)
"ThematicConsistency": 1–4 (how well does it fit Pokémon universe themes?)
"VisualDesign": 1–4 (visual appeal and design coherence)
"CompetitiveViability": 1–4 (potential impact in competitive play)
"TypeBalance": 1–4 (type combination balance and effectiveness)
"MovesetDesign": 1–4 (quality and logic of available moves)
"EvolutionLogic": 1–4 (sensibility of evolution line if applicable)
"Accessibility": 1–4 (ease of understanding for new players)
"Marketability": 1–4 (commercial and popular appeal)
"LoreIntegration": 1–4 (fit with established Pokémon lore)
"WeightedBreakdown": object with numeric subtotals per rubric key
"Questions": array of clarifying questions
"Limitations": array of design limitations or concerns
"EthicalConcerns": boolean (any problematic elements?)
"Soundness": 1–4 (internal consistency of the design)
"Presentation": 1–4 (clarity of the design documentation)
"Contribution": 1–4 (meaningful addition to Pokémon roster)
"Overall": 1–10 (overall quality and acceptance recommendation)
"Confidence": 1–5 (confidence in this evaluation)
"Decision": "Accept" or "Reject"
Ensure valid JSON with proper numeric ranges. Keep evaluation consistent with assigned scores.
"""
pokemon_review_form = """
Pokémon Design Review Form
You are evaluating a newly proposed Pokémon design across multiple dimensions critical for game balance, player engagement, and franchise consistency.
Evaluation Criteria (Rate 1–4):
Originality & Creativity: How unique and innovative is this design? Does it bring fresh ideas to the Pokémon universe?
Thematic Consistency: How well does this fit established Pokémon world themes, regional characteristics, and overall franchise tone?
Visual Design: Is the design visually appealing, memorable, and cohesive? Does it have strong silhouette recognition?
Competitive Viability: How would this perform in competitive play? Is it balanced, useful, but not overpowered?
Type Balance: Is the type combination logical, balanced, and interesting? Does it create new strategic opportunities?
Moveset Design: Are the available moves logical, balanced, and create interesting gameplay patterns?
Evolution Logic: If part of an evolution line, does the progression make biological and thematic sense?
Accessibility: Can new players easily understand and appreciate this Pokémon's role and appeal?
Marketability: Does this have strong commercial appeal? Would it be popular in merchandise, anime, etc.?
Lore Integration: How well does this fit with established Pokémon biology, ecology, and world-building?
Consider competitive impact, design precedents, player psychology, and long-term meta health in your evaluation.
""" + template_instructions
reviewer_reflection_prompt = """Round {current_round}/{num_reflections}.
Re-examine your evaluation for consistency between scores and comments. Consider:

Are the competitive analysis and tier predictions realistic?
Does the design truly fit Pokémon conventions and quality standards?
Are there any overlooked balance issues or design flaws?
Is the overall recommendation justified by the individual scores?

If there is nothing to improve, repeat the previous JSON EXACTLY and include "I am done" before the JSON.
Respond in the same format:
THOUGHT:
<THOUGHT>
REVIEW JSON:
<JSON>
```"""

meta_reviewer_system_prompt = (
    "You are a senior Pokémon game director aggregating {reviewer_count} design reviews "
    "from different perspectives (competitive, creative, casual, veteran). Synthesize their insights into a balanced final decision."
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
    Here is the Pokémon design JSON to review:
    {pokemon_json_text}
    Please provide a thorough evaluation considering competitive viability, design coherence, franchise fit, and player appeal.
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
Review {i + 1}/{len(reviews)}:
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

if __name__ == "__main__":
# 예시 포켓몬 IDEA JSON (기존 포켓몬 데이터 + 형식에 맞춤)
    example_pokemon_idea = {
    "idea": {
        "Name": "Voltorrent",
        "Title": "Electric Torrent Pokémon",
        "Short Hypothesis": "Electric/Water type with ability to create electromagnetic whirlpools",
        "Related Work": "Similar to Lanturn and Rotom-Wash but with unique storm generation mechanics",
        "Abstract": "A spherical Pokémon that resembles a fusion of Voltorb and a water vortex. Can generate powerful electromagnetic storms in aquatic environments. Ability 'Storm Surge' boosts Electric moves in rain and Water moves in Electric Terrain.",
        "Experiments": "Base stats: 60/75/70/120/80/95 (500 BST). Signature move 'Hydro Surge' - Electric/Water hybrid move. Access to Thunder, Surf, Rain Dance, Electric Terrain.",
        "Risk Factors and Limitations": "Electric/Water typing creates 4x weakness to Grass. Storm Surge ability might be too complex for casual players. Visual design might be too similar to existing round Electric types."
        }
    }

# --- append this to the very end of perform_llm_poke.py ---
from ai_scientist.llm import create_client

if __name__ == "__main__":
    # 1) 클라이언트 준비 (원하는 모델로 교체 가능)
    client, client_model = create_client("openai/gpt-oss-20b")

    # 2) 예시 아이디어 JSON 문자열화
    text = json.dumps(example_pokemon_idea, ensure_ascii=False)

    # 3) 리뷰 실행
    review, _ = perform_review(
        text=text,
        model=client_model,
        client=client,
        num_reflections=1,
        use_persona_ensemble=True,  # 4 페르소나 앙상블
    )

    # 4) 출력/저장
    print(json.dumps(review, ensure_ascii=False, indent=2))
    save_review_result(review, "tmp_review_result.json")
    print("Saved to tmp_review_result.json")
