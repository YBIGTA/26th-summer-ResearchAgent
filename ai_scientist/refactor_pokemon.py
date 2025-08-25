import os
import json
import random
import argparse
from typing import Dict
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
POKEMON_DATA_DIR = os.path.join("..", "crawler", "json")

def extract_core_data(data: Dict) -> Dict:
    # (This function is the same as before)
    typing = data.get("infobox", {}).get("타입", ["Normal"])
    abilities = [data.get("infobox", {}).get("특성")]
    if hidden := data.get("infobox", {}).get("숨겨진 특성"): abilities.append(hidden)
    stats_raw = data.get("abilities", {})
    stats = {k: stats_raw[k].get("종족값") for k in ["HP", "공격", "방어", "특수공격", "특수방어", "스피드"] if k in stats_raw}
    stats["Total"] = sum(stats.values())
    all_explanations = [txt for gen in data.get("explanations", {}).values() for txt in gen.values()]
    pokedex_entries = " | ".join(all_explanations)
    moveset = [move.get("기술") for gen in ["9세대", "8세대"] if gen in data.get("moveset", {}) for move in data["moveset"][gen].get("레벨업으로 배우는 기술", [])[:6]]
    
    return {"typing": typing, "abilities": abilities, "stats": stats, "pokedex_entries": pokedex_entries, "move_highlights": moveset}

def generate_creative_data(core_data: Dict) -> Dict:
    # (This function is the same as before)
    llm = ChatUpstage(model="solar-pro2", temperature=0.8)
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""
        당신은 창의적인 포켓몬 디자이너입니다. 기존 포켓몬의 다음 데이터를 기반으로, *콘셉트는 비슷하지만 독창적인* 새로운 포켓몬 캐릭터를 만들어 주세요. 새로운 캐릭터의 텍스트 정보는 **반드시 한국어**로 작성해야 합니다.
        **영감을 위한 핵심 데이터:**
        - 타입: {typing}
        - 종족값: {stats}
        - 특성: {abilities}
        - 도감 설명 요약: {pokedex_entries}
        - 기술 예시: {move_highlights}
        **출력은 *반드시* 다음 키를 포함하는 유효한 JSON 형식이어야 합니다:**
        - "Name": 새롭고 멋진 영어 이름.
        - "Korean Name": 포켓몬의 한국어 이름.
        - "Title": 짧고 서사적인 느낌의 칭호 (한국어).
        - "Region/Habitat": 서식지에 대한 간략한 설명 (한국어).
        - "Appearance": 외형에 대한 묘사 (한국어).
        - "Personality": 성격에 대한 요약 (한국어).
        - "Pokedex Entry": 새롭고 간결한 도감 설명 (한국어).
        - "Signature Move": 다음 키를 포함하는 새로운 전용 기술 JSON 객체: "Name"(영어), "Type"(영어), "Category", "Power", "Accuracy", "PP", "Effect"(한국어).
        - "Playstyle": 배틀 스타일에 대한 간략한 요약 (한국어).
        - "Sample Image Prompt": AI 이미지 생성을 위한 짧은 프롬프트 (영어).
        - "Design Rationale": 디자인 콘셉트에 대한 설명 (한국어).
        JSON 출력:
        """,
        input_variables=["typing", "stats", "abilities", "pokedex_entries", "move_highlights"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    return chain.invoke(core_data)

def main():
    """Main function to run the refactoring process from the command line."""
    parser = argparse.ArgumentParser(description="Generate new Pokémon data based on existing files.")
    parser.add_argument("-c", "--count", type=int, required=True, help="Number of Pokémon to generate.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path for the generated JSON.")
    args = parser.parse_args()

    generated_characters = []
    
    try:
        files = [f for f in os.listdir(POKEMON_DATA_DIR) if f.endswith('.json')]
        if not files:
            raise FileNotFoundError(f"No Pokémon data files found in '{POKEMON_DATA_DIR}'.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    for i in range(args.count):
        print(f"Generating Pokémon {i+1} of {args.count}...")
        random_file = os.path.join(POKEMON_DATA_DIR, random.choice(files))
        with open(random_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        core_data = extract_core_data(source_data)
        creative_data = generate_creative_data(core_data)
        creative_data.pop("Typing", None) # prevent LLM from overwriting our correctly formatted list
        
        final_pokemon = {
            "character": {
                **creative_data,
                "Typing": core_data["typing"],
                "Stats": core_data["stats"],
                "Abilities": core_data["abilities"],
                "Movepool Highlights": core_data["move_highlights"],
            }
        }
        generated_characters.append(final_pokemon)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(generated_characters, f, ensure_ascii=False, indent=2)

    print(f"\nSuccess! Generated {args.count} Pokémon and saved to '{args.output}'.")

if __name__ == "__main__":
    main()