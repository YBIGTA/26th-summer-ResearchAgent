import argparse
import json
import os
import re
import traceback
import random
from typing import Any, Dict, List, Optional

import sys

from ai_scientist.perform.perform_ideation_temp_free import parse_tool_call
from ai_scientist.perform.perform_poke_review import perform_review
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Import from your actual modules ---
from ai_scientist.llm import create_client, extract_json_object, get_response_from_llm
from ai_scientist.vlm import generate_image_from_prompt, encode_image_to_base64

# --- Configuration & Prompts ---
POKEMON_DATA_DIR = os.path.join("crawler", "json")

# Prompt for the first "Creative Agent"
CREATIVE_AGENT_SYSTEM_PROMPT = """당신은 ‘포켓몬 캐릭터 생성자’ 페르소나를 가진 크리에이티브 디자이너입니다. 
당신의 목표는 참신하고 매력적인 신규 포켓몬 캐릭터 아이디어를 자유롭게 브레인스토밍하는 것입니다. 
ACTION/ARGUMENTS 형식을 사용하여 아이디어를 제안하고, 'ReviewbyLLM'을 통해 아이디어를 다듬은 후 'FinalizeCharacter'로 최종 아이디어 설명을 제출하세요."""

# --- NEW: Prompt for the second "Structuring Agent" ---
STRUCTURING_AGENT_SYSTEM_PROMPT = """당신은 데이터를 정확하게 변환하는 전문가입니다. 
사용자가 제공한 텍스트에서 정보를 추출하여 지정된 JSON 형식으로 완벽하게 변환하는 것이 당신의 유일한 임무입니다. 
정보를 추가하거나 변경하지 말고, 오직 주어진 텍스트의 내용만을 사용하여 JSON 구조를 채우세요. 출력은 JSON 객체만 포함해야 합니다."""

# --- Helper functions ---
def parse_tool_call(text: str) -> (Optional[str], Optional[str]):
    try:
        action_match = re.search(r"ACTION:\s*(.*?)\s*ARGUMENTS:", text, re.DOTALL | re.IGNORECASE)
        arguments_match = re.search(r"ARGUMENTS:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        action = action_match.group(1).strip().replace('"', '') if action_match else "FinalizeCharacter"
        arguments_text = arguments_match.group(1).strip() if arguments_match else text
        return action, arguments_text
    except Exception:
        return "FinalizeCharacter", text

def structure_creative_output(client: Any, client_model: str, markdown_text: str) -> Optional[Dict]:
    """Takes messy markdown from the creative agent and structures it into the final, correct JSON schema."""
    print("    -> Structuring Agent: Converting markdown to the required JSON schema...")
    print(markdown_text)
    
    prompt = f"""
당신은 JSON 구조화 전문가입니다. 당신의 유일한 임무는 구조화되지 않은 텍스트를 아래 예시와 정확히 동일한 "RAW" JSON 형식으로 변환하는 것입니다.

### 중요 규칙 ###
-   아래 제공된 **완벽한 RAW JSON 예시**의 구조를 **정확히** 따르십시오.
-   예시 구조에 없는 키('Skills', 'Rarity', 'Retreat Cost' 등)를 추가, 제거, 또는 변경하지 마십시오.
-   모든 키 이름(예: 'Name', 'Typing', 'Stats', 'HP')은 반드시 **영문**으로 유지해야 합니다.
-   정보가 없는 경우 `null` 대신 빈 리스트 `[]`나 빈 객체 `{{}}`를 사용하십시오.

---

### 완벽한 RAW JSON 예시 ###

#### 예시 입력 텍스트 ####
캐릭터 이름은 플로라리스(Floralis)야. "꽃잎으로 꿈을 수놓는 수호자"라는 별명을 가지고 있어. 타입은 풀이랑 페어리. 특성은 'Natural Cure' 와 'Aroma Veil'이야. HP는 95, 공격 60, 방어 85, 특공 105, 특방 115, 스피드 70이야. 고유 기술은 "Petal Dream"인데, 페어리 타입이고 아군을 치유하는 기술이라 위력은 없어. 명중률은 100%고 PP는 10. 다른 기술로는 "Moonblast"랑 "Giga Drain"을 배워.

#### 예시 출력 RAW JSON ####
{{
  "character": {{
    "Name": "Floralis",
    "Korean Name": "플로라리스",
    "Title": "꽃잎으로 꿈을 수놓는 수호자",
    "Typing": ["Grass", "Fairy"],
    "Abilities": ["Natural Cure", "Aroma Veil"],
    "Stats": {{
      "HP": 95,
      "Attack": 60,
      "Defense": 85,
      "Sp.Atk": 105,
      "Sp.Def": 115,
      "Speed": 70
    }},
    "Signature Move": {{
      "Name": "Petal Dream",
      "Type": "Fairy",
      "Category": "Status",
      "Power": null,
      "Accuracy": "100%",
      "PP": 10,
      "Effect": "자신과 동료 포켓몬의 HP를 소량 회복하고 상태 이상을 치유한다."
    }},
    "Movepool Highlights": ["Moonblast", "Giga Drain", "Leech Seed"],
    "Image": null
  }}
}}

---

### 새로운 작업 ###

#### 변환할 사용자 입력 텍스트 ####

{markdown_text}

#### 출력 RAW JSON ####
    """
    try:
        response_text, _ = get_response_from_llm(
            prompt=prompt, client=client, model=client_model,
            system_message=STRUCTURING_AGENT_SYSTEM_PROMPT, msg_history=[]
        )
        structured_json = extract_json_object(response_text)
        return structured_json.get("character")
    except Exception as e:
        print(f"    -> Structuring Agent failed: {e}")
        return None

# --- Main Agent Workflow ---
def run_generation_workflow(client: Any, client_model: str, inspiration_prompt: str) -> Optional[Dict]:
    """Orchestrates the two-stage (Creative -> Structuring) generation process."""
    msg_history = []
    last_tool_results = ""
    last_arguments_text = ""
    num_reflections = 3

    # --- STAGE 1: Creative Agent Brainstorming ---
    for reflection_round in range(num_reflections):
        print(f"  - Creative Agent: Reflection Round {reflection_round + 1}/{num_reflections}...")
        
        prompt_text = (f"다음 영감을 바탕으로 캐릭터 초안을 생성하세요...\n'''\n{inspiration_prompt}\n'''\nACTION/ARGUMENTS 형식을 준수하세요." 
                       if reflection_round == 0 else 
                       f"라운드 {reflection_round + 1}/{num_reflections}\n\n캐릭터를 점검하고 개선하세요. 최종 아이디어라고 생각되면 'FinalizeCharacter'로 제출하세요.\n\n최근 결과:\n{last_tool_results or '결과 없음.'}")
        
        response_text, msg_history = get_response_from_llm(
            prompt=prompt_text, client=client, model=client_model,
            system_message=CREATIVE_AGENT_SYSTEM_PROMPT, msg_history=msg_history
        )
        
        action, arguments_text = parse_tool_call(response_text)
        
        if arguments_text: last_arguments_text = arguments_text
        if not action: last_tool_results = "Error: Could not parse ACTION."; continue

        print(f"    -> Creative Agent Action: {action}")
        if "FinalizeCharacter" in action:
            print("    -> Creative Agent finalized its idea.")
            break # Exit the loop to move to the structuring stage
        
        try:
            arguments_json = extract_json_object(arguments_text)
            if "ReviewbyLLM" in action:
                review_result, _ = perform_review(text=arguments_json, model=client_model, client=client)
                last_tool_results = json.dumps(review_result, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError) as e:
            last_tool_results = f"Error: Agent provided invalid JSON. Error: {e}"

    # --- STAGE 2: Structuring Agent Formatting ---
    if not last_arguments_text:
        print("  -> Creative Agent did not produce any output.")
        return None
        
    final_structured_character = structure_creative_output(client, client_model, last_arguments_text)
    return final_structured_character


def extract_core_data(data: Dict) -> Dict:
    typing = data.get("infobox", {}).get("타입", ["Normal"])
    abilities = [data.get("infobox", {}).get("특성")]
    if hidden := data.get("infobox", {}).get("숨겨진 특성"): abilities.append(hidden)
    stats_raw = data.get("abilities", {})
    stats = {k: stats_raw[k].get("종족값") for k in ["HP", "공격", "방어", "특수공격", "특수방어", "스피드"] if k in stats_raw}
    stats["Total"] = sum(stats.values())
    all_explanations = [txt for gen in data.get("explanations", {}).values() for txt in gen.values()]
    pokedex_entries = " | ".join(all_explanations)
    return {"typing": typing, "abilities": abilities, "stats": stats, "pokedex_entries": pokedex_entries}

def main():
    parser = argparse.ArgumentParser(description="Generate new Pokémon data based on existing files.")
    parser.add_argument("-c", "--count", type=int, required=True, help="Number of Pokémon to generate.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path for the generated JSON.")
    parser.add_argument("--model", type=str, default="upstage:solar-pro", help="Model to use.")
    args = parser.parse_args()

    client, client_model = create_client(args.model)
    
    generated_characters = []
    try:
        files = [f for f in os.listdir(POKEMON_DATA_DIR) if f.endswith('.json')]
        if not files: raise FileNotFoundError(f"'{POKEMON_DATA_DIR}' has no JSON files.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr) # Print errors to stderr
        return

    for i in range(args.count):
        print(f"--- Generating Pokémon {i+1} of {args.count} ---", file=sys.stderr) # Print progress to stderr
        random_file = os.path.join(POKEMON_DATA_DIR, random.choice(files))
        with open(random_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        core_data = extract_core_data(source_data)
        inspiration_prompt = (f"타입: {core_data['typing']}\n특성: {core_data['abilities']}\n종족값: {core_data['stats']}\n"
                              f"기존 도감 설명: {core_data['pokedex_entries']}")
        
        final_character = run_generation_workflow(client, client_model, inspiration_prompt)
        
        if final_character:
            final_character["Typing"] = core_data["typing"]
            final_character["Stats"] = core_data["stats"]
            final_character["Abilities"] = core_data["abilities"]
            
            if prompt := final_character.get("Sample Image Prompt"):
                name_sanitized = "".join(c for c in final_character.get("Name", "unknown") if c.isalnum())
                img_path = f"generated_pokemon/img/{name_sanitized}.png"
                if generate_image_from_prompt(prompt, img_path):
                    final_character["Image"] = encode_image_to_base64(img_path)

            final_character_data = {"character": final_character}
            generated_characters.append(final_character_data)
            
            print(json.dumps(final_character_data, ensure_ascii=False))
            sys.stdout.flush() # Ensure the output is sent immediately

        else:
            print(f"  -> Failed to generate Pokémon {i+1}", file=sys.stderr)

    # Still save the complete file at the end
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(generated_characters, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Success! Generated {len(generated_characters)} Pokémon and saved to '{args.output}'.", file=sys.stderr)

if __name__ == "__main__":
    main()