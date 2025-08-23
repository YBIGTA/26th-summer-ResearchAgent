import json
import pandas as pd

class Skill:
    def __init__(self, name, type_, power, accuracy, pp, effect=None):
        self.name = name
        self.type = type_
        self.power = power
        self.accuracy = accuracy
        self.pp = pp
        self.effect = effect
        # 에너지 소모량 결정: 파워가 -1이거나 None이면 0, 그 외 위력 구간에 따라 1~3
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
            "Effect": self.effect
        }

class PokemonCard:
    def __init__(self, entry):
        # JSON 구조가 두 가지가 있으므로, 'character' 키가 있으면 내부를 사용
        ch = entry['character'] if 'character' in entry else entry
        self.name = ch['Name']
        # 타입은 리스트 또는 문자열
        self.types = ch['Typing'] if isinstance(ch['Typing'], list) else [ch['Typing']]
        self.image = ch.get('Image', None)
        # 스탯 숫자 변환
        self.stats = {}
        for k, v in ch['Stats'].items():
            if isinstance(v, (int, float)):
                self.stats[k] = v
            elif isinstance(v, str):
                try:
                    self.stats[k] = int(v)
                except:
                    self.stats[k] = 0
            else:
                self.stats[k] = 0
        # 특성
        self.abilities = ch.get('Abilities', [])
        # 기술: 시그니처 기술
        self.skills = []
        sig = ch.get('Signature Move')
        if sig:
            # Power와 Accuracy를 숫자로 변환
            power = sig.get('Power')
            if isinstance(power, str):
                try:
                    power = int(power)
                except:
                    power = None
            acc = sig.get('Accuracy')
            acc_val = None
            if isinstance(acc, str) and acc.endswith('%'):
                try:
                    acc_val = float(acc.replace('%', ''))
                except:
                    acc_val = None
            self.skills.append(
                Skill(
                    sig.get('Name'),
                    sig.get('Type'),
                    power,
                    acc_val,
                    sig.get('PP'),
                    sig.get('Effect'),
                )
            )
        # Movepool Highlights → 카드 게임 기술로 변환(기본 파워 계산)
        attack = self.stats.get('Attack', 0)
        sp_atk = self.stats.get('Sp.Atk', 0)
        base_power = max(attack, sp_atk) // 2
        base_power = max(40, min(base_power, 120))  # 최소 40, 최대 120
        base_accuracy = 100
        base_pp = 20
        base_type = self.types[0]
        for mv in ch.get('Movepool Highlights', []):
            self.skills.append(
                Skill(
                    mv,
                    base_type,
                    base_power,
                    base_accuracy,
                    base_pp,
                    None,
                )
            )
        # 총합 스탯/파워 스코어 계산
        self.total_stats = ch['Stats'].get('Total') or sum(
            [self.stats.get(stat, 0) for stat in ['HP','Attack','Defense','Sp.Atk','Sp.Def','Speed']]
        )
        self.power_score = (
            1.2*self.stats.get('Attack',0) +
            1.2*self.stats.get('Sp.Atk',0) +
            1.0*self.stats.get('Defense',0) +
            1.0*self.stats.get('Sp.Def',0) +
            0.8*self.stats.get('HP',0) +
            1.0*self.stats.get('Speed',0)
        )
        # 후퇴 비용 (HP 기준 간단한 휴리스틱)
        hp = self.stats.get('HP', 50)
        if hp < 60:
            self.retreat = 1
        elif hp < 100:
            self.retreat = 2
        else:
            self.retreat = 3
        self.rarity = None  # 나중에 부여

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
            "Image": self.image  # 이미지도 함께 저장
        }

# JSON 로딩 및 카드 객체 생성
with open('ideas/i_cant_believe_its_not_better_image.json') as f:
    raw = json.load(f)
cards = [PokemonCard(entry) for entry in raw]

# 파워 스코어에 따라 희귀도 부여: 상위 10% Legendary, 10~40% Rare, 나머지 Common
sorted_cards = sorted(cards, key=lambda c: c.power_score, reverse=True)
for i, c in enumerate(sorted_cards):
    if i < len(cards)*0.1:
        c.rarity = 'Legendary'
    elif i < len(cards)*0.4:
        c.rarity = 'Rare'
    else:
        c.rarity = 'Common'

# 요약 출력
summary = [
    {
        "Name": c.name,
        "Types": ', '.join(c.types),
        "Power Score": round(c.power_score, 1),
        "Rarity": c.rarity,
        "Retreat Cost": c.retreat,
    }
    for c in cards
]
df = pd.DataFrame(summary)
print(df)

# 개별 카드 데이터 확인
for c in cards:
    print('\\n--- Card Data ---')
    print(json.dumps(c.as_dict(), indent=2, ensure_ascii=False))
    
# cards 데이터를 JSON으로 저장
output_data = [c.as_dict() for c in cards]

with open('ideas/pokemon_cards_output.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
