
# ResearchAgent: AI-Driven Pokémon Card Generator

**ResearchAgent**는 완전히 새로운 **포켓몬 카드**를 상상하고 제작하는 **AI 기반 시스템**입니다.  
전통적인 고정 템플릿 기반 자동화 파이프라인과 달리, **대규모 언어 모델(LLM)**을 활용해 창의적인 포켓몬 아이디어를 생성하고 이를 완성도 높은 **트레이딩 카드 이미지**로 시각화합니다.

---

## 🚀 How It Works

이 프로젝트의 워크플로우는 **세 가지 단계**로 구성됩니다.

### 1. Ideation (`perform_ideation_temp_free.py`)
- LLM을 활용하여 창의적인 포켓몬 아이디어를 생성합니다.  
- 타입, 능력치, 스킬, 특수 효과 조합을 탐색하고, 구조화된 **JSON 파일**로 저장합니다.  
- 각 포켓몬에는 HP, 공격력, 방어력, 타입, 기술 설명 등이 포함됩니다.

---

### 2. Card Formatting (`make_pokemoncard.py`)
- 아이디어 단계에서 생성된 원본 JSON 데이터를 **카드 게임에 적합한 형식**으로 변환합니다.  
- 카드 희귀도 설정, 후퇴 비용 계산, 타입별 색상 매핑, 스킬 표시 형식 준비 등을 수행합니다.  
- 변환된 JSON은 바로 **시각화 단계**로 전달할 수 있습니다.

---

### 3. Visualization (`make_visualize.py`)
- Pillow 라이브러리를 이용해 카드 템플릿을 렌더링합니다.  
- 포켓몬 타입에 따른 색상 적용, 타입 배지 표시, 능력치/기술 텍스트 표시, 이미지 삽입을 지원합니다.  
- 최종 출력물은 포켓몬 카드별 `.png` 이미지로 저장됩니다.

---

## 🛠️ Usage

### 1. Install Dependencies
Python 3 환경에서 의존성 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

---

### 2. Run Streamlit
스트림릿 실행합니다:

```bash
streamlit run main.py
```

---

## 📂 Project Layout

```
├── ai_scientist/               # 아이디어 생성 프로젝트
│ └── perform/                  # 아이디어 생성 실행 코드
├── crawler/
├── main.py                     # Streamlit 애플리케이션 실행 파일
├── card_template.html          # 카드 렌더링을 위한 HTML 템플릿
├── requirements.txt            # 의존성 패키지 목록
├── README.md                   # 프로젝트 설명 파일
└── generated_pokemon/          # 생성된 데이터 및 결과물 저장 디렉토리
    ├── ideas/                  # 생성된 포켓몬 아이디어 (JSON)
    └── img/                    # 생성된 포켓몬 이미지 (PNG)
```

---

## 📝 Notes

- **Artwork**  
  JSON의 `Image` 필드에 `data:image/...` 형식의 URI를 입력하면 카드에 해당 이미지를 포함할 수 있습니다.  
  입력하지 않으면 기본 카드 레이아웃으로 생성됩니다.

- **Customisation**  
  타입 색상과 희귀도 매핑은 스크립트 내에서 직접 커스터마이징할 수 있습니다.

- **Safety**  
  LLM이 생성한 코드 실행에는 위험이 있을 수 있으므로 **샌드박스 환경**에서 테스트하는 것을 권장합니다.

---

## 📜 License
이 리포지토리는 **AI Scientist** 원본 라이선스를 따릅니다.  
사용 및 수정 시 반드시 라이선스 조건을 준수해야 합니다.
