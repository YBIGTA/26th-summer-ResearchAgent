# Pokémon Wiki Web Crawler 🕸️

A Python-based web crawler designed to fetch comprehensive Pokémon data from the [Pokémon Fandom Wiki (Korean)](https://pokemon.fandom.com/ko/wiki/전국도감).

- It uses **Selenium** to navigate and render dynamic JavaScript content and **BeautifulSoup4** to parse the HTML and extract structured information.
- The crawler systematically goes through the National Pokédex, scrapes data for each Pokémon, and saves it into an individual, well-structured JSON file.

## Some Features ✨

  * **Data Extraction:** Scrapes information (types, species, height, weight, etc.), descriptions, evolution chains, abilities, and full movesets.
  * **Output:** Saves data for each Pokémon in a separate file within a `json/` directory (e.g., `이상해씨.json`).
  * **Miscellaneous:** Allows resume from the last Pokémon using the `--continue_from` command-line argument, and saves `crawler.log` file for debug logs.

-----

## Project Structure

```
crawler/
├── crawl_nat_pokedex.py    # Main driver script to run the full scrape
├── web_crawler.py          # Core helper functions (driver setup, page fetching)
├── extract/                # Modules for scraping specific page sections
│   ├── abilities_extract.py
│   ├── evolutions_extract.py
│   ├── explanations_extract.py
│   ├── infobox_extract.py
│   └── moveset_extract.py
│
├── json/                   # Output directory for scraped data
├── save/                   # Saved webpages (see web_crawler.py)
├── chromedriver.exe*       # (Required for manual setup) Local browser driver
└── requirements.txt        # Project dependencies
```

-----

## Step-by-Step Guide 🚀

### 1\. Prerequisites

  * Python 3.8 or newer.
  * Google Chrome browser installed.

### 2\. Setup & Installation

**A. Clone the Repository**

```bash
# assumed you already git cloned the full repository
cd crawler
```

**B. Install Dependencies**

It's highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

**C. ChromeDriver Configuration**

Selenium requires a [browser driver](https://googlechromelabs.github.io/chrome-for-testing/) to interact with Google Chrome.

### 3\. Run the Crawler

Make sure you are in the project's root directory and have your virtual environment activated.

To start the process from the beginning and scrape all Pokémon:

```bash
python crawl_nat_pokedex.py
```

#### Resume an Interrupted Scrape

If the script stopped for any reason, you can resume it from a specific Pokémon. For example, to resume from "리자몽" (Charizard):

```bash
python crawl_nat_pokedex.py --continue_from "리자몽"
```

The script will find "리자몽" in its list and continue scraping from that point forward.

#### Use the crawler for individual pages

```bash
# python web_crawler.py [url] [infobox|explanations|evolutions|abilities|moveset|all]
# optional flags: --save along with file_name to save webpage as html file which can be used via -local (url becomes file path)
# use double quotes if korean text is being converted to hexadecimal
python web_crawler.py "<full-link>"  all 
```

-----

## Output 🗒️

The scraped data will be saved in the `json/` directory.

- Each file corresponds to one Pokémon and contains a JSON object.
- There are five JSON dictionaries (infobox, explanations, evolutions, abilities, moveset) denoting a specific group of information.
- The data contains images and audio in base64 encoded raw-data format.
- None/ Null types are denoted `-1` for levels and power.
- See the provided example .json below:

**Example: `이상해씨.json` (Bulbasaur)**

```json
{
    "infobox": {
        "이미지": "data:image/png;base64, ...
        "타입": [
            "풀",
            "독"
        ],
        "분류": "씨앗포켓몬",
        "특성": "심록",
        "숨겨진 특성": "엽록소",
        "LV.100 경험치량": 1059860,
        "도감 번호": [
            "관동#0001",
            "성도#231",
            "호연#203",
            "칼로스#080",
            "갑옷섬#068",
            "블루베리#164"
        ],
        "형태": "data:image/png;base64, ...
        "발자국": (생략)
        "포켓몬 도감 색": "초록",
        "기초 친밀도": 70,
        "키": "0.7m",
        "몸무게": "6.9kg",
        "포획률": 45,
        "성비": {
            "수컷": 87.5,
            "암컷": 12.5
        },
        "포켓몬 교배": {
            "알그룹": [
                "알그룹",
                "괴수",
                "식물"
            ],
            "부화 걸음수": "5,120걸음"
        },
        "울음소리": "data:application/ogg;base64, ...
    ,
    "explanations": {
        "1세대": {
            "적": "태어났을 때부터 등에 식물의 씨앗이 있으며 조금씩 크게 자란다.",
            "청": "태어났을 때부터 등에 이상한 씨앗이 심어져 있으며 몸과 함께 자란다고 한다.",
            "피카츄": "며칠 동안 아무것도 먹지 않아도 건강하다! 등에 있는 씨앗에는 많은 영양분이 있어서 문제없다!"
        },
        ...
        "9세대": {
            "스칼렛": "태어나서 얼마 동안 등의 씨앗에 담긴 영양을 섭취하며 자란다.",
            "바이올렛": "태어날 때부터 등에 씨앗을 짊어지고 있다. 몸이 크게 성장함에 따라 씨앗도 커진다."
        }
    },
    "evolutions": [
        {
            "이름": "이상해씨",
            "타입": [
                "풀",
                "독"
            ],
            "이미지": (생략)
            "진화 방법": "Lv. 16"
        },
        ...
    ],
    "abilities": {
        "HP": {
            "종족값": 45,
            "능력치 범위": {
                "Lv. 50일 때": "105 - 152",
                "Lv. 100일 때": "200 - 294"
            },
            "노력치": 0
        },
        "공격": {
            "종족값": 49,
            "능력치 범위": {
                "Lv. 50일 때": "48 - 111",
                "Lv. 100일 때": "92 - 216"
            },
            "노력치": 0
        },
        "방어": {
            "종족값": 49,
            "능력치 범위": {
                "Lv. 50일 때": "48 - 111",
                "Lv. 100일 때": "92 - 216"
            },
            "노력치": 0
        },
        "특수공격": {
            "종족값": 65,
            "능력치 범위": {
                "Lv. 50일 때": "63 - 128",
                "Lv. 100일 때": "121 - 251"
            },
            "노력치": 1
        },
        "특수방어": {
            "종족값": 65,
            "능력치 범위": {
                "Lv. 50일 때": "63 - 128",
                "Lv. 100일 때": "121 - 251"
            },
            "노력치": 0
        },
        "스피드": {
            "종족값": 45,
            "능력치 범위": {
                "Lv. 50일 때": "45 - 106",
                "Lv. 100일 때": "85 - 207"
            },
            "노력치": 0
        },
        "총합": {
            "종족값": 318
        }
    },
    "moveset": {
        "9세대": {
            "레벨업으로 배우는 기술": [
                {
                    "레벨": -1,
                    "기술": "몸통박치기",
                    "타입": "노말",
                    "분류": "물리",
                    "위력": 40,
                    "명중률": "100%",
                    "PP": 35
                },
                ...
            ],
            "교배로 배우는 기술": [
                {
                    "부모": "",
                    "기술": "꽃잎댄스",
                    "타입": "풀",
                    "분류": "특수",
                    "위력": 120,
                    "명중률": "100%",
                    "PP": 10
                },
                ...
            ]
        },
        ...
        "1세대": {
              ...
            ]
        }
    }
}
```
