# make_pokemoncard.py
import json
import base64
import io
import unicodedata, re
from typing import Dict, List, Any
import pandas as pd  # optional: 요약 출력
from PIL import Image

# --------- 경로 ---------
DETAILS_JSON_PATH = "ideas/i_cant_believe_its_not_better.json"          # 아이디어 디테일(원본)
IMAGE_JSON_PATH   = "ideas/i_cant_believe_its_not_better_image.json"    # 이미지 전용(Name, Image)
OUTPUT_JSON_PATH  = "ideas/pokemon_cards_output.json"                    # 카드 생성용 출력

# --------- 유틸 ---------
def normalize_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).strip().lower()
    return re.sub(r"\s+", "", s)

def _json_load_safely(entry) -> Any:
    if isinstance(entry, str):
        try:
            return json.loads(entry)
        except Exception:
            return {}
    return entry

def coerce_character(entry: Any) -> Dict[str, Any]:
    """문자열/중첩 등 무엇이 와도 일관된 character dict로 변환 + 기본값 보정."""
    entry = _json_load_safely(entry)
    ch = entry.get("character", entry) if isinstance(entry, dict) else {}
    if not isinstance(ch, dict):
        ch = {}

    # 필수 키 기본값
    ch.setdefault("Name", "unknown")
    ch.setdefault("Korean Name", "")
    # Typing: 리스트 보정
    typing = ch.get("Typing", [])
    if isinstance(typing, str):
        typing = [typing]
    if not isinstance(typing, list):
        typing = []
    ch["Typing"] = typing

    # Stats 보정 (누락/비정상 대비)
    keys = ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]
    stats = ch.get("Stats")
    if not isinstance(stats, dict):
        stats = {}
    fixed = {}
    for k in keys:
        v = stats.get(k, 0)
        try:
            fixed[k] = int(v)
        except Exception:
            fixed[k] = 0
    fixed["Total"] = sum(fixed[k] for k in keys)
    ch["Stats"] = fixed

    # 나머지 필드 기본값
    ch.setdefault("Abilities", [])
    ch.setdefault("Signature Move", {})
    ch.setdefault("Movepool Highlights", [])
    ch.setdefault("Image", None)
    return ch

def load_image_index(path: str) -> Dict[str, str]:
    """image.json → {정규화이름: base64(or data-url)}"""
    idx: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return idx
    except Exception as e:
        print(f"[warn] image.json read failed: {e}")
        return idx

    if isinstance(data, dict):
        data = [data]
    for rec in data or []:
        key = normalize_key(rec.get("Name") or rec.get("Korean Name"))
        if not key:
            continue
        b64 = rec.get("Image") or rec.get("image_base64")
        if b64:
            idx[key] = b64
    return idx

# --------- 도메인 클래스 ---------
class Skill:
    def __init__(self, name, type_, power, accuracy, pp, effect=None):
        self.name = name
        self.type = type_
        self.power = power
        self.accuracy = accuracy
        self.pp = pp
        self.effect = effect
        if power is None or power == -1:
            self.energy_cost = 0
        elif power <= 60:
            self.energy_cost = 1
        elif power <= 90:
            self.energy_cost = 2
        else:
            self.energy_cost = 3

    def as_dict(self):
        return {
            "Name": self.name,
            "Type": self.type,
            "Power": self.power,
            "Accuracy": self.accuracy,
            "EnergyCost": self.energy_cost,
            "Effect": self.effect,
        }

class PokemonCard:
    def __init__(self, entry):
        ch = entry.get("character", entry) if isinstance(entry, dict) else {}
        ch = coerce_character(ch)  # 안전 보정

        self.name = ch["Name"]
        self.types = ch["Typing"] if ch["Typing"] else ["노말"]
        self.image = ch.get("Image", None)

        # Stats
        self.stats = ch["Stats"]
        # Abilities
        self.abilities = ch.get("Abilities", [])

        # Skills
        self.skills: List[Skill] = []
        sig = ch.get("Signature Move") or {}
        if sig:
            power = sig.get("Power")
            if isinstance(power, str):
                try:
                    power = int(power)
                except Exception:
                    power = None
            acc = sig.get("Accuracy")
            acc_val = None
            if isinstance(acc, str) and acc.endswith("%"):
                try:
                    acc_val = float(acc.replace("%", ""))
                except Exception:
                    acc_val = None
            self.skills.append(
                Skill(sig.get("Name"), sig.get("Type") or (self.types[0] if self.types else "노말"),
                      power, acc_val, sig.get("PP"), sig.get("Effect"))
            )

        # Movepool Highlights → 기본 스펙으로 카드 공격 생성
        attack = self.stats.get("Attack", 0)
        sp_atk = self.stats.get("Sp.Atk", 0)
        base_power = max(40, min(max(attack, sp_atk) // 2, 120))
        base_accuracy = 100
        base_pp = 20
        base_type = self.types[0]
        for mv in ch.get("Movepool Highlights", []):
            self.skills.append(Skill(mv, base_type, base_power, base_accuracy, base_pp, None))

        # 합/파워/희귀도 계산용
        self.total_stats = self.stats.get("Total") or sum(
            self.stats.get(k, 0) for k in ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]
        )
        self.power_score = (
            1.2 * self.stats.get("Attack", 0) +
            1.2 * self.stats.get("Sp.Atk", 0) +
            1.0 * self.stats.get("Defense", 0) +
            1.0 * self.stats.get("Sp.Def", 0) +
            0.8 * self.stats.get("HP", 0) +
            1.0 * self.stats.get("Speed", 0)
        )
        hp = self.stats.get("HP", 50)
        self.retreat = 1 if hp < 60 else 2 if hp < 100 else 3
        self.rarity = None

    def as_dict(self):
        return {
            "Name": self.name,
            "Types": self.types,
            "Stats": self.stats,
            "Total Stats": self.total_stats,
            "Power Score": round(self.power_score, 2),
            "Retreat Cost": self.retreat,
            "Rarity": self.rarity,
            "Abilities": self.abilities,
            "Skills": [sk.as_dict() for sk in self.skills[:2]],
            "Image": self.image,  # (data-url or base64) – draw 단계가 소비
        }

# --------- 파이프라인 ---------
# 1) details 로드 + 정규화
with open(DETAILS_JSON_PATH, "r", encoding="utf-8") as f:
    details_raw = json.load(f)

characters: List[Dict[str, Any]] = []
for e in (details_raw if isinstance(details_raw, list) else [details_raw]):
    ch = coerce_character(e)
    characters.append({"character": ch})

# 2) image.json을 이름으로 조인하여 Image 붙이기
img_index = load_image_index(IMAGE_JSON_PATH)
for rec in characters:
    ch = rec["character"]
    key = normalize_key(ch.get("Name") or ch.get("Korean Name"))
    if key in img_index and not ch.get("Image"):
        ch["Image"] = img_index[key]

# 3) 카드 객체 생성
cards = [PokemonCard(rec) for rec in characters]

# 4) 파워 스코어로 희귀도 부여
sorted_cards = sorted(cards, key=lambda c: c.power_score, reverse=True)
n = len(sorted_cards)
for i, c in enumerate(sorted_cards):
    if i < max(1, int(n * 0.10)):
        c.rarity = "Legendary"
    elif i < max(1, int(n * 0.40)):
        c.rarity = "Rare"
    else:
        c.rarity = "Common"

# 5) 요약 출력(옵션)
summary = [
    {
        "Name": c.name,
        "Types": ", ".join(c.types),
        "Power Score": round(c.power_score, 1),
        "Rarity": c.rarity,
        "Retreat Cost": c.retreat,
    }
    for c in sorted_cards
]
try:
    print(pd.DataFrame(summary))
except Exception:
    print(summary)

# 6) 출력 저장
output_data = [c.as_dict() for c in cards]
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(output_data)} cards -> {OUTPUT_JSON_PATH}")
