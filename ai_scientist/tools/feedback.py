import os
import requests
import time
import warnings
from typing import Dict, List, Optional, Union
import backoff

import json
import numpy as np
from ai_scientist.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)

from ai_scientist.tools.base_tool import BaseTool
AVAILABLE_reflections = ["conservative","futuristic"]

def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class ReviewbyLLM_tool(BaseTool):
    def __init__(
        self,
        name: str = "ReviewbyLLM",
        description: str = """
            "생성한 아이디어에 대해 LLM을 사용하여 피드백을 제공합니다. "
            "이 도구는 아래 생성한 아이디어의 강점과 약점을 분석하고, 개선할 수 있는 방법을 제안합니다."

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
            
            ARGUMENTS는 {"character": {...}} 형태의 **유효한 JSON**이어야 합니다.""",
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "ReviewbyLLM",
                "type": "str",
                "description": "생성한 아이디어에 대해 LLM을 사용하여 피드백을 제공합니다. 이 도구는 아이디어의 강점과 약점을 분석하고, 개선할 수 있는 방법을 제안합니다.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.reviewer_system_prompt_base = (
            "당신은 새롭게 생성된 포켓몬 아이디어에 대한 피드백을 제공하는 전문가입니다. "
            "아이디어의 강점과 약점을 분석하고, 개선할 수 있는 방법을 제안합니다. "
            "아이디어는 다음과 같습니다: {idea}. "
            "피드백은 간결하고 명확하게 작성되어야 합니다."
        )
          

    def use_tool(self, client, model, client_embed, IDEA_JSON,log_callback) -> Optional[str]:
        review_text= self.perform_review(IDEA_JSON, model,client, client_embed, log_callback)
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        idea_dir=f"experiments/{date}_attempt_{0}"
        os.makedirs(idea_dir, exist_ok=True)
        with open(os.path.join(idea_dir, "review.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(review_text, ensure_ascii=False, indent=4))
        return review_text
    
    def perform_review(self, text, model: str, client, client_embed, log_callback,llm_reviewer_option=None, temperature=0.75,msg_history=None) -> str:
        
        mynewprompt =f"""
        아이디어는 다음과 같습니다. 
        '''
        {text}
        '''
        당신은 다음 아이디어를 평가하고 개선할 책임이 있습니다.
        1. 아이디어의 강점과 약점을 분석합니다.
        2. 아이디어를 개선할 수 있는 방법을 제안합니다.
        3. 피드백은 간결하고 명확하게 작성되어야 합니다.
        '''json
        {{
            "feedback": "아이디어에 대한 피드백을 여기에 작성하세요."
        }}
        '''
        주의: 1번과 2번에 대한 답변외에는 생성하지 마세요.
        """
        reviewer_system_prompt = self.reviewer_system_prompt_base.format(idea=text)
        if llm_reviewer_option is 'conservative':
            reviewer_system_prompt += "\n\n주의: 보수적인 관점에서 피드백을 작성하세요."
        elif llm_reviewer_option is 'futuristic':
            reviewer_system_prompt += "\n\n주의: 미래지향적인 관점에서 피드백을 작성하세요."

        response, msg_history = get_response_from_llm(
            prompt=mynewprompt,
            client=client,
            model=model,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature
        )
        result= f"""
        리뷰결과: 
        """
        result += response
        log_callback(result)
        
        
        return result

    
