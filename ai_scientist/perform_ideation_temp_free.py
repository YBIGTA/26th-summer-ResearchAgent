import argparse
import json
import os.path as osp
import re
import regex
import traceback
from typing import Any, Dict, List
# utils/tool_parse.py
import ast
from typing import Tuple, Dict, Any, Optional,List
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
    extract_json_object,
)

from ai_scientist.tools.feedback import ReviewbyLLM_tool
from ai_scientist.tools.base_tool import BaseTool

# Create tool instances
ReviewbyLLM_tool = ReviewbyLLM_tool()

# Define tools at the top of the file
tools = [
    ReviewbyLLM_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
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

system_prompt = f"""You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {{"query": "your search query"}}. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }}
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research."""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

# Define the reflection prompt
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""
# utils/tool_parse.py
import re, json, ast
from typing import Tuple, Dict, Any, Optional

ALLOWED_ACTIONS = {"ReviewbyLLM", "FinalizeIdea"}

def similarity_check(embedmodel,arguments_text,log_callback,idea_fname,idea_fname2) :
    with open(idea_fname, "r",encoding="utf-8") as f:
        content=f.read()
        matches = regex.findall(r'\{(?:[^{}]|(?R))*\}', content, flags=regex.DOTALL)
        ideas= [match.strip() for match in matches]
    with open(idea_fname2, "r",encoding="utf-8") as f:
        content2=f.read()
        matches2 = regex.findall(r'\{(?:[^{}]|(?R))*\}', content2, flags=regex.DOTALL)
        for match in matches2:
            ideas.append(match.strip())
            
    def extract_fields(text):
        fields=["Title","Short Hypothesis","Related Work","Abstract","Experiments","Risk Factors and Limitations"]
        extracked=[]
        for field in fields:
            match = re.search(rf'"{field}"\s*:\s*"([^"]+)"', text)
            if match:
                extracked.append(match.group(1).strip())
        return "".join(extracked)
    def get_embedding(text):
        pretext = extract_fields(text)
        embedding= embedmodel.encode(pretext)
        return np.array(embedding).reshape(1, -1)
    
    user_embedding = get_embedding(arguments_text)
    best_score = -1
    worst_score = 1
    best_idea = None
    k=0
    for idea in ideas:
        k+=1
        idea_text=f"idea: {idea}"
        idea_embedding = get_embedding(idea_text)
        score = cosine_similarity(user_embedding, idea_embedding)[0][0]
        if score > best_score:
            best_score = score
            best_idea = idea_text
        if score < worst_score:
            worst_score = score
            
    log_callback(f"Similarity check: {k} ideas checked, best score: {best_score}, worst score: {worst_score}")
    return best_score
    
    

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
    return {"idea": {"Name": "unparsed", "Title": "UNPARSED_IDEA", "Abstract": s[:1500]}}
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
    return {"idea": {"Name": "unparsed", "Title": "UNPARSED", "Abstract": arg_region[:1500]}}

def parse_tool_call(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    항상 (action, arguments_dict) 반환.
    실패해도 FinalizeIdea와 폴백 딕트를 돌려 파이프라인이 계속 진행되도록 한다.
    """
    if not text:
        return "FinalizeIdea", {"idea": {"Name": "empty", "Title": "EMPTY_RESPONSE", "Abstract": ""}}

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
        if isinstance(args_obj, dict) and "idea" in args_obj:
            action = "FinalizeIdea"
        elif isinstance(args_obj, dict) and "query" in args_obj:
            action = "ReviewbyLLM"
        else:
            action = "FinalizeIdea"

    return action, args_obj


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    client_embed: Any,
    idea_fname2: str ,
    client2: Any,
    model2: str,
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
            sim= 0
            review=0
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

                    print(f"Similarity check start")
                    similarity=similarity_check(client_embed,arguments_text,print,idea_fname,idea_fname2)
                    if similarity > 0.8:
                        last_tool_results = arguments_text
                        last_tool_results += f"유사도 값이 {similarity}입니다. 아이디어를 수정해야 합니다."
                        sim +=1
                        if sim > 3:
                            idea_finalized = True
                            break
                        continue
                    # Process the action and arguments
                    if action == "ReviewbyLLM":
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
                            result = tool.use_tool(client2,model2,client_embed,arguments_text,logcallback=print)
                            last_tool_results = str(result)
                        except Exception as e:
                            review+=1
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # Parse arguments
                        try:
                            arguments_json = load_arguments_loose(arguments_text)

                            # arguments_json = json.loads(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")

                            # Append the idea to the archive
                            idea_str_archive.append(json.dumps(idea))
                            print(f"Proposal finalized: {idea}")
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeIdea.")
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
        "--model2",
        type=str,
        default="LGAI-EXAONE",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--emb_model",
        type=str,
        default="Qwen3-Embedding",
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
    client2, client_model2 = create_client(args.model2)
    client_emd, _ = create_client(args.emb_model)

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
        client_embed=client_emd,
        idea_fname2=idea_fname ,
        client2= client2,
        model2= client_model2,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
    
