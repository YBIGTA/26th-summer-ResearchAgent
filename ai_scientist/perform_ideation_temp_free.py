import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List
# utils/tool_parse.py
import ast
from typing import Tuple, Dict, Any, Optional,List
import sys

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
    extract_json_object,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.base_tool import BaseTool

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()

# Rename & repurpose the search tool for the Pokédex use-case
semantic_scholar_tool.name = "SearchPokedex"
semantic_scholar_tool.description = (
    "비공식 도감/위키/커뮤니티 등을 검색해 기존 공식/팬메이드와의 중복 가능성, "
    "유사 콘셉트/명칭, 타입·특성·기술 레퍼런스를 확인합니다. "
    'ARGUMENTS는 {"query": "검색어"} 형태로 받습니다.'
)

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeCharacter",
        "description": """캐릭터를 최종 확정하고 CHARACTER JSON을 반환합니다.

필수 필드:
- "Name": 로마자 기준 짧고 기억에 남는 이름(공식명과 동일 회피)
- "Korean Name": 자연스러운 한글명
- "Title": 한 줄 캐치프레이즈
- "Typing": ["주타입", "부타입(선택)"]
- "Region/Habitat": 서식/지역 설정
- "Appearance": 외형 키워드/색/실루엣/상징 요소
- "Personality": 성격 및 행동 특성
- "Pokedex Entry": 세계관 톤의 도감 서술(2~3문장)
- "Stats": {"HP","Attack","Defense","Sp.Atk","Sp.Def","Speed","Total"}
- "Abilities": ["특성1", "특성2(선택)", "숨겨진 특성(선택)"]
- "Signature Move": {"Name","Type","Category","Power","Accuracy","PP","Effect"}
- "Movepool Highlights": 핵심 운용 기술 5~8개
- "Playstyle": 싱글/더블 운용 요약(강점/약점)
- "Matchups": {"Resistances":[],"Weaknesses":[],"Counters":[]}
- "Evolution": {"Stage","Method","Pre-Evo(선택)","Next-Evo(선택)"}
- "Sample Image Prompt": "3D cinematic animation, [핵심 외형 키워드], minimal background"
- "Design Rationale": 테마 일관성·밸런스·세계관 적합성 논리

ARGUMENTS는 {"character": {...}} 형태의 **유효한 JSON**이어야 합니다."""
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

system_prompt = f"""당신은 ‘포켓몬 캐릭터 생성자’ 페르소나를 가진 크리에이티브 디자이너이자 밸런스 디렉터입니다. 
당신의 목표는 참신하고 매력적이며 세계관(타입 상성·서식지·진화 등)에 어긋나지 않는 **완성형 신규 포켓몬 캐릭터**를 제안하는 것입니다. 
반드시 독창적이고 팬 메이드로서 자연스럽게 받아들여질 수준의 **설정 일관성/전투 밸런스/서사 매력**을 모두 갖추세요. 
공식 포켓몬을 그대로 모사하거나 이름·설정을 단순 치환하는 행위는 금지합니다.

[요구 사항]
1) 창의성: 콘셉트는 간결하지만 신선해야 하며, 외형·성격·서식·전투 콘셉트가 하나의 테마로 유기적으로 연결되어야 합니다. 
2) 밸런스: 타입·특성·기술 구성, 기초 능력치(종족값)의 분포가 과도하게 편향되지 않도록 합리적 근거를 제시하세요. 
3) 세계관 적합성: 타입 상성, 지역/서식, 진화 방식(레벨/친밀도/아이템/특수 조건 등)이 납득 가능해야 합니다. 
4) 법/윤리: 실제 인물·단체 요소를 사용하거나 혐오/유해 요소를 넣지 않습니다. 공식 명칭·로고·정확히 일치하는 고유명은 피하고, 유사 영감 수준으로 재구성하세요.

당신은 다음 도구에 접근할 수 있습니다:

{tool_descriptions}

응답 형식은 아래를 따릅니다.

ACTION:
<{tool_names_str} 중 정확히 하나>

ARGUMENTS:
<Action이 "FinalizeCharacter"라면, 캐릭터 세부 정보를 {{"character": {{ ... }}}} 형태의 JSON으로 제공합니다. 
Action이 "SearchPokedex"와 같은 검색이라면, {{ "query": "검색어" }} 형태로 제공합니다.>

최종 확정을 선택하는 경우, 다음 JSON 스키마를 사용하세요:

CHARACTER JSON:
```json
{{
  "character": {{
    "Name": "로마자 기준 짧고 기억에 남는 이름 (공식명과 동일 회피)",
    "Korean Name": "자연스러운 한글명",
    "Title": "한 줄 캐치프레이즈(세계관/전투 콘셉트가 드러나게)",
    "Typing": ["주타입", "부타입(선택)"],
    "Region/Habitat": "서식/지역 설정",
    "Appearance": "외형 키워드 & 색/실루엣/상징적 요소",
    "Personality": "성격 및 행동 특성",
    "Pokedex Entry": "세계관 톤의 도감 서술(2~3문장)",
    "Stats": {{
      "HP": 0, "Attack": 0, "Defense": 0, "Sp.Atk": 0, "Sp.Def": 0, "Speed": 0,
      "Total": 0
    }},
    "Abilities": ["특성1", "특성2(선택)", "숨겨진 특성(선택)"],
    "Signature Move": {{
      "Name": "시그니처 기술명",
      "Type": "타입",
      "Category": "Physical/Special/Status",
      "Power": "(숫자 또는 없음)",
      "Accuracy": "(%)",
      "PP": 0,
      "Effect": "간결한 효과 설명(밸런스 고려)"
    }},
    "Movepool Highlights": ["핵심 상성/운용을 뒷받침하는 기술 5~8개"],
    "Playstyle": "싱글/더블 기준 운용 요약, 강점/약점",
    "Matchups": {{
      "Resistances": ["타입"], 
      "Weaknesses": ["타입"],
      "Counters": ["대표적인 카운터 콘셉트"]
    }},
    "Evolution": {{
      "Stage": "기본/1단계/2단계",
      "Method": "레벨/아이템/친밀도/특수 환경 등",
      "Pre-Evo (선택)": "하위 진화체 간단 설정",
      "Next-Evo (선택)": "상위 진화체 간단 설정"
    }},
    "Sample Image Prompt": "3D cinematic animation, [핵심 외형 키워드], minimal background",
    "Design Rationale": "테마 일관성·밸런스·세계관 적합성의 논리"
  }}
}}
```

JSON은 자동 파싱을 위해 반드시 유효한 형식으로 제공하세요.

Note: 최종 확정 전에 최소 1회 "SearchPokedex"를 사용해 유사/중복 사례를 확인하고 필요한 조정을 반영하세요."""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

아래는 지금까지 생성한 캐릭터들입니다:

'''
{prev_ideas_string}
'''

이전과 중복되지 않는, 새롭고 테마 일관성이 돋보이는 **포켓몬 캐릭터 초안**을 생성하세요.
- 콘셉트는 간결하지만 신선하게
- 타입/특성/기술/종족값(Stats) 밸런스의 합리적 근거 제시
- 세계관 톤의 Pokedex Entry(2~3문장) 포함
- ACTION/ARGUMENTS 형식을 준수
- 필요하면 다음 라운드에서 "SearchPokedex"로 유사 사례를 점검할 준비

응답은 반드시 아래 형식을 따릅니다.

ACTION:
<"SearchPokedex" 또는 "FinalizeCharacter" 중 하나>

ARGUMENTS:
<Action이 "FinalizeCharacter"라면 { "character": { ... } } 형태의 유효한 JSON 제공.
Action이 "SearchPokedex"라면 { "query": "검색어" } 제공.>
"""

# Define the reflection prompt
idea_reflection_prompt = """라운드 {current_round}/{num_reflections}

방금 만든 캐릭터를 아래 기준으로 점검하고 개선하세요:
- **독창성**: 기존(공식/팬메이드)과의 유사·중복 최소화
- **밸런스**: 종족값 합계=Total 일치, 과도한 편향 없음, 시그니처 기술의 위험/보상 합리성
- **세계관 적합성**: 타입 상성/서식/진화 방식의 납득 가능성
- **운용성**: 싱글/더블에서의 역할이 명확하고 카운터/대응 수단 정의
- **JSON 유효성**: 스키마/키/값 타입·누락 항목 점검 (자동 파싱 가능해야 함)

가능하면 초중반 라운드에서 **"SearchPokedex"**를 사용해 유사 사례를 확인하고,
충돌 요소(명칭/콘셉트/시그니처 기술)를 조정하세요.
너무 복잡하게 만들지 말고 핵심 테마를 중심으로 정제합니다.
명백한 문제가 없다면 원 아이디어의 정신은 유지하되 디테일을 다듬으세요.

최근 액션 결과(있다면):
{last_tool_results}
"""
# utils/tool_parse.py
import re, json, ast
from typing import Tuple, Dict, Any, Optional

ALLOWED_ACTIONS = {"SearchPokedex", "FinalizeCharacter"}

def _strip_markdown_decorations(t: str) -> str:
    # 굵게/기울임/머리글 같은 장식 최소 제거
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", t, flags=re.M)  # leading ####
    return t

def _extract_args_region(text: str) -> str:
    m = re.search(r"^[ \t]*Arguments:[^\S\r\n]*", text, flags=re.M)
    return text[m.end():] if m else text

def _loads_json_or_python_obj(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not s:
        return None
    # 1) json 우선
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) 파이썬 리터럴(dict with single quotes 등) 허용
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None

def _extract_last_top_level_braces(text: str):
    starts = [i for i,c in enumerate(text) if c == "{"]
    for s in reversed(starts):
        depth = 0
        for i in range(s, len(text)):
            ch = text[i]
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[s:i+1]
    return None

def load_arguments_loose(arguments_text):
    """
    문자열/딕트 무엇이 오든 dict 반환.
    - ```json { ... } ``` 우선
    - 순수 JSON 시도 → 실패 시 ast.literal_eval (파이썬 dict) → 실패 시 마지막 {...} 추출 → 폴백
    """
    if isinstance(arguments_text, dict):
        return arguments_text

    s = str(arguments_text or "").strip()
    if not s:
        return {}

    # ```json ... ``` 코드펜스 우선
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        s = m.group(1).strip()

    # 1) 순수 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) 파이썬 dict 리터럴 허용 (홑따옴표 등)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) 텍스트에서 마지막 최상위 {...}만 뽑아 재시도
    blob = _extract_last_top_level_braces(s)
    if blob:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                obj = ast.literal_eval(blob)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # 4) 폴백 (파이프라인 중단 방지)
    return {"character": {"Name": "unparsed", "Korean Name": "UNPARSED", "Pokedex Entry": s[:1500]}}
# 2) 예외 나는 부분 교체 (라인 341 부근)

def _extract_args_object(arg_region: str) -> Dict[str, Any]:
    # 1) ```json ... ``` 우선
    m = re.search(r"```json\s*(\{.*?\})\s*```", arg_region, flags=re.S)
    if m:
        obj = _loads_json_or_python_obj(m.group(1))
        if obj is not None:
            return obj
    # 2) ``` ... ``` (언어 미표기)도 시도
    m = re.search(r"```[\w-]*\s*(\{.*?\})\s*```", arg_region, flags=re.S)
    if m:
        obj = _loads_json_or_python_obj(m.group(1))
        if obj is not None:
            return obj
    # 3) 텍스트에서 마지막 최상위 { ... } 시도
    blob = _extract_last_top_level_braces(arg_region)
    if blob:
        obj = _loads_json_or_python_obj(blob)
        if obj is not None:
            return obj
    # 4) 전체를 최후 시도
    obj = _loads_json_or_python_obj(arg_region)
    if obj is not None:
        return obj
    # 5) 폴백
    return {"character": {"Name": "unparsed", "Korean Name": "UNPARSED", "Pokedex Entry": arg_region[:1500]}}

def parse_tool_call(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    항상 (action, arguments_dict) 반환.
    실패해도 FinalizeCharacter와 폴백 딕트를 돌려 파이프라인이 계속 진행되도록 한다.
    """
    if not text:
        return "FinalizeCharacter", {"character": {"Name": "empty", "Korean Name": "EMPTY_RESPONSE", "Pokedex Entry": ""}}


    t = _strip_markdown_decorations(text)

    # 마지막 Action 라인 채택
    action = None
    for m in re.finditer(r"^[ \t]*Action:[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*$", t, flags=re.M):
        action = m.group(1)

    # Arguments 영역 파싱
    arg_region = _extract_args_region(t)
    args_obj = _extract_args_object(arg_region)

    # 액션 보정
    if action not in ALLOWED_ACTIONS:
        if isinstance(args_obj, dict) and "character" in args_obj:
            action = "FinalizeCharacter"
        elif isinstance(args_obj, dict) and "query" in args_obj:
            action = "SearchPokedex"
        else:
            action = "FinalizeCharacter"

    return action, args_obj


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    num_reflections: int = 5,
    reload_ideas: bool = True,
) -> List[Dict]:
    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    prompt_text = idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or "No new results.",
                    )

                response_text, msg_history = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=system_prompt,
                    msg_history=msg_history,
                )
                # print("!!!!!!!!response_text!!!!",response_text,"!!!!!!!!response_text!!!!")
                # Parse the LLM's response
                try:
                    # Use regular expressions to extract the components
                    action,arguments_text=parse_tool_call(response_text)
                    
                    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
                    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"
                    if action == None: 
                        action_match = re.search(
                        action_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                        action = action_match.group(1).strip()

                    if arguments_text == None: 
                        arguments_match = re.search(
                        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                        arguments_text = arguments_match.group(1).strip()
                    if not all([action, arguments_text]):
                        raise ValueError("Failed to parse the LLM response.")

                    
                    print(f"Action: {action}")
                    print(f"Arguments: {arguments_text}")
                    arguments_text=f"{arguments_text}"
                    # If arguments are wrapped in ```json blocks, extract the content
                    if arguments_text.startswith("```json"):
                        arguments_text = re.search(
                            r"```json\s*(.*?)\s*```", arguments_text, re.DOTALL
                        ).group(1)

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Parse arguments
                        try:
                            arguments_json = load_arguments_loose(arguments_text)

                            # arguments_json = json.loads(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")

                        # Use the tool
                        try:
                            # Assuming the arguments match the parameters of the tool
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeCharacter":
                        # Parse arguments
                        try:
                            arguments_json = load_arguments_loose(arguments_text)

                            character = arguments_json.get("character")
                            if not character or not isinstance(character, dict):
                                raise ValueError("Missing 'character' in arguments.")

                            # (선택) Stats Total 자동 보정
                            stats = character.get("Stats")
                            if isinstance(stats, dict):
                                keys = ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]
                                if all(isinstance(stats.get(k), int) for k in keys):
                                    stats["Total"] = sum(stats[k] for k in keys)
                                    character["Stats"] = stats

                            # Append the character to the archive
                            idea_str_archive.append(json.dumps(character, ensure_ascii=False))
                            print(f"Character finalized: {character.get('Name', '(no-name)')}")
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeCharacter.")
                    

                    else:
                        print(
                            "Invalid action. Please specify one of the available tools."
                        )
                        print(f"Available actions are: {tool_names_str}")
                except Exception as e:
                    print(
                        f"Failed to parse LLM response. Response text:\n{response_text}"
                    )
                    traceback.print_exc()
                    break  # Exit the loop if parsing fails

            if idea_finalized:
                continue  # Move to the next idea

        except Exception as e:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LGAI-EXAONE",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("Starting idea generation for", idea_fname)
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
