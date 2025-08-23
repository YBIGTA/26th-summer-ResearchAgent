import json, base64, io, os
from PIL import Image, ImageDraw, ImageFont

# 타입 매핑과 색상 정의 (영문/한글 모두 대응)
KO_TO_EN = {'전기':'Electric','강철':'Steel','불꽃':'Fire','물':'Water','풀':'Grass',
            '악':'Dark','페어리':'Fairy','얼음':'Ice','바위':'Rock','고스트':'Ghost',
            '땅':'Ground'}
# 영어 타입은 그대로 사용
for t in ['Electric','Steel','Fire','Water','Grass','Dark','Fairy','Ice','Rock','Ghost','Ground']:
    KO_TO_EN[t] = t

TYPE_COLORS = {
    'Electric': (255,215,0), 'Steel': (180,180,200), 'Fire': (255,69,0),
    'Water': (64,164,223), 'Grass': (120,200,80), 'Dark': (87,71,75),
    'Fairy': (238,153,172), 'Ice': (173,216,230), 'Rock': (184,160,56),
    'Ghost': (123,104,171), 'Ground': (222,184,135),
}

# 폰트 설정
font_title  = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
font_header = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
font_text   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
font_small  = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)

# 카드 데이터 구조 및 희귀도 계산
cards_data = []
with open('ideas/pokemon_cards_output.json') as f:
    raw = json.load(f)
for entry in raw:
    ch = entry['character'] if 'character' in entry else entry
    # 이미지 디코딩
    header, encoded = ch['Image'].split(',')
    art = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGBA')
    # 타입, 스탯
    types = ch['Typing'] if isinstance(ch['Typing'], list) else [ch['Typing']]
    types_en = [KO_TO_EN.get(t, t) for t in types]
    primary = types_en[0]
    stats = ch['Stats']
    # 파워 스코어 계산
    atk, spa = stats['Attack'], stats['Sp.Atk']
    pwr_score = 1.2*atk + 1.2*spa + stats['Defense'] + stats['Sp.Def'] + 0.8*stats['HP'] + stats['Speed']
    # 후퇴 비용: HP가 낮으면 1, 중간이면 2, 높으면 3
    retreat = 1 if stats['HP'] < 60 else 2 if stats['HP'] < 100 else 3
    # 기술: 시그니처 + 하이라이트를 카드 규격에 맞춰 변환
    skills = []
    sig = ch.get('Signature Move')
    if sig:
        pwr = int(sig['Power']) if isinstance(sig['Power'], str) else sig['Power']
        acc = None
        if isinstance(sig['Accuracy'], str) and sig['Accuracy'].endswith('%'):
            acc = float(sig['Accuracy'].replace('%',''))
        skills.append({
            'name': sig['Name'],
            'type': KO_TO_EN.get(sig['Type'], sig['Type']),
            'power': pwr,
            'accuracy': acc,
        })
    # 하이라이트는 공격/특수공격 절반을 위력으로 간주
    base_power = max(atk, spa)//2
    base_power = max(40, min(base_power, 120))
    for mv in ch.get('Movepool Highlights', []):
        skills.append({'name': mv, 'type': primary, 'power': base_power, 'accuracy': 100})
    # 에너지 비용 계산 (위력구간에 따라 1~3)
    for s in skills:
        if s['power'] is None or s['power'] == -1:
            cost = 0
        elif s['power'] <= 60:
            cost = 1
        elif s['power'] <= 90:
            cost = 2
        else:
            cost = 3
        s['energy_cost'] = cost
    cards_data.append({
        'name': ch['Name'], 'types_en': types_en, 'primary_type': primary,
        'color': TYPE_COLORS.get(primary, (200,200,200)),
        'stats': stats, 'hp': stats['HP'], 'skills': skills,
        'power_score': pwr_score, 'retreat': retreat, 'image': art
    })

# 희귀도 부여 (상위10% Legendary, 10~40% Rare, 나머지 Common)
sorted_cards = sorted(cards_data, key=lambda x: x['power_score'], reverse=True)
for i, c in enumerate(sorted_cards):
    if i < len(cards_data)*0.1:
        c['rarity'] = 'Legendary'
    elif i < len(cards_data)*0.4:
        c['rarity'] = 'Rare'
    else:
        c['rarity'] = 'Common'

# 카드 생성 함수
def draw_energy_icons(draw, x, y, cost, color, r=8, gap=5):
    for i in range(cost):
        cx = x + i*(2*r + gap)
        draw.ellipse([cx-r, y-r, cx+r, y+r], fill=color+(255,), outline=(0,0,0))
    return x + cost*(2*r + gap)

def draw_rarity_symbol(draw, rarity, x, y, size=12):
    if rarity == 'Common':
        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=(0,0,0))
    elif rarity == 'Rare':
        half = size//2
        points = [(x, y-half), (x+half, y), (x, y+half), (x-half, y)]
        draw.polygon(points, fill=(0,0,0))
    else:  # Legendary -> 5각별
        import math
        pts=[]
        for j in range(10):
            ang = math.pi/2 + j*math.pi/5
            r = size/2 if j%2==0 else (size/2)*0.5
            pts.append((x + r*math.cos(ang), y - r*math.sin(ang)))
        draw.polygon(pts, fill=(0,0,0))

def create_card(card, W=480, H=680):
    border=8; hdr_h=80; art_h=250; atk_h=200
    img = Image.new('RGBA', (W,H), (255,255,255,255))
    draw = ImageDraw.Draw(img)
    # 테두리와 헤더
    col = card['color']; header_col = tuple(int(c*0.85) for c in col)
    draw.rectangle([0,0,W-1,H-1], outline=col+(255,), width=border)
    draw.rectangle([border+6, border+6, W-border-6, border+hdr_h], fill=header_col+(255,))
    # 상단: Stage, 이름, HP, 타입 아이콘
    draw.text((border+16, border+12), 'Basic Pokémon', font=font_small, fill=(0,0,0))
    draw.text((border+16, border+36), card['name'], font=font_title, fill=(0,0,0))
    hp_txt = f"HP {card['hp']}"
    hp_x = W - border - 60 - draw.textsize(hp_txt, font=font_header)[0]
    draw.text((hp_x, border+36), hp_txt, font=font_header, fill=(205,0,0))
    # 타입 아이콘
    ic_center = (W-border-30, border+46)
    r=12
    draw.ellipse([ic_center[0]-r, ic_center[1]-r, ic_center[0]+r, ic_center[1]+r], fill=col+(255,), outline=(0,0,0))
    draw.text((ic_center[0]-6, ic_center[1]-8), card['primary_type'][0], font=font_small, fill=(255,255,255))
    # 삽화
    ar = [border+10, border+hdr_h+10, W-border-10, border+hdr_h+10+art_h]
    aw, ah = card['image'].size
    scale = min((ar[2]-ar[0])/aw, (ar[3]-ar[1])/ah)
    new_w, new_h = int(aw*scale), int(ah*scale)
    art_resized = card['image'].resize((new_w, new_h), Image.LANCZOS)
    dx = ar[0] + ((ar[2]-ar[0]) - new_w)//2
    dy = ar[1] + ((ar[3]-ar[1]) - new_h)//2
    img.paste(art_resized, (dx, dy), art_resized)
    # 공격 섹션
    atk_r = [border+10, ar[3]+10, W-border-10, ar[3]+10+atk_h]
    draw.rectangle(atk_r, fill=(245,245,245,255))
    y = atk_r[1] + 10
    for skill in card['skills'][:3]:
        # 에너지 아이콘
        endx = draw_energy_icons(draw, atk_r[0]+10, y+12, skill['energy_cost'],
                                 TYPE_COLORS.get(skill['type'], (160,160,160)))
        # 이름
        draw.text((endx+5, y), skill['name'], font=font_text, fill=(0,0,0))
        dmg = '' if skill['power'] is None or skill['power']==-1 else str(skill['power'])
        dmg_w = draw.textsize(dmg, font=font_text)[0]
        draw.text((atk_r[2]-20-dmg_w, y), dmg, font=font_text, fill=(0,0,0))
        y += 30
    # 하단: 타입/후퇴/희귀도
    bottom_y = atk_r[3] + 10
    type_str = '/'.join(card['types_en'])
    draw.text((border+10, bottom_y+5), type_str, font=font_small, fill=(80,80,80))
    # 후퇴 비용(무색 에너지 아이콘)
    ret_x = border+10 + draw.textsize(type_str, font=font_small)[0] + 40
    for i in range(card['retreat']):
        cx = ret_x + i*22; cy = bottom_y+12
        draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=(200,200,200,255), outline=(0,0,0))
    draw_rarity_symbol(draw, card['rarity'], W-border-40, bottom_y+12, size=14)
    return img

# 카드 출력 및 저장
out_dir = 'pokemon_cards_game'
os.makedirs(out_dir, exist_ok=True)
for card in cards_data:
    img = create_card(card)
    img.save(os.path.join(out_dir, f"card_{card['name'].replace(' ','_')}.png"))
