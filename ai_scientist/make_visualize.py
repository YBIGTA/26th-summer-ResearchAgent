import os
import json
import base64
import io
import random
import re
import unicodedata
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------
# 카드 디테일(JSON). 여기엔 Name/Types/Stats/Skills 등 정보만 있다고 가정
INPUT_JSON_PATH = "ideas/pokemon_cards_output.json"     # better.json에 해당

# image.json (이름-이미지 Base64만 저장되어 있는 파일)
IMAGE_JSON_PATH = "ideas/i_cant_believe_its_not_better_image.json"

# 로컬 원본 이미지 디렉토리(있으면 파일이 최우선)
IMAGE_SRC_DIR = "ideas/img"

# 출력 디렉토리
OUTPUT_DIR = "ideas/improved_card_images"

# ------------------------------------------------------------
# 타입 컬러(에너지 색)
# ------------------------------------------------------------
TYPE_COLORS = {
    '전기': '#F7D02C', 'Electric': '#F7D02C',
    '물': '#6390F0', 'Water': '#6390F0',
    '불꽃': '#EE8130', 'Fire': '#EE8130',
    '풀': '#7AC74C', 'Grass': '#7AC74C',
    '얼음': '#96D9D6', 'Ice': '#96D9D6',
    '격투': '#C22E28', 'Fighting': '#C22E28',
    '독': '#A33EA1', 'Poison': '#A33EA1',
    '땅': '#E2BF65', 'Ground': '#E2BF65',
    '비행': '#A98FF3', 'Flying': '#A98FF3',
    '에스퍼': '#F95587', 'Psychic': '#F95587',
    '벌레': '#A6B91A', 'Bug': '#A6B91A',
    '바위': '#B6A136', 'Rock': '#B6A136',
    '유령': '#735797', 'Ghost': '#735797',
    '드래곤': '#6F35FC', 'Dragon': '#6F35FC',
    '악': '#705746', 'Dark': '#705746',
    '강철': '#B7B7CE', 'Steel': '#B7B7CE',
    '페어리': '#D685AD', 'Fairy': '#D685AD',
    '노말': '#A8A77A', 'Normal': '#A8A77A',
    '공통': '#A8A77A',  # colorless energy
}

# ------------------------------------------------------------
# 유틸: 이름 매칭 키 정규화
# ------------------------------------------------------------
def normalize_key(s: str) -> str:
    """이름 매칭용 키 표준화: 전각/소문자/공백 제거."""
    s = unicodedata.normalize("NFKC", str(s or "")).strip().lower()
    s = re.sub(r"\s+", "", s)
    return s

# ------------------------------------------------------------
# 유틸: image.json 로드 → {정규화된이름: base64(or data-url)} 딕셔너리
# ------------------------------------------------------------
def load_image_index(path: str) -> Dict[str, str]:
    idx = {}
    if not os.path.exists(path):
        return idx
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    if isinstance(arr, dict):
        arr = [arr]
    for rec in arr or []:
        key = normalize_key(rec.get("Name") or rec.get("Korean Name"))
        if not key:
            continue
        b64 = rec.get("Image") or rec.get("image_base64")  # 호환
        if not b64:
            continue
        idx[key] = b64
    return idx

# ------------------------------------------------------------
# 유틸: base64 혹은 data-url 모두 허용해서 PIL Image로 변환
# ------------------------------------------------------------
def load_image_from_b64_any(s: str) -> Image.Image:
    if s.startswith("data:"):
        _, b64 = s.split(",", 1)
    else:
        b64 = s
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA")

# (호환) 기존 인터페이스 유지
def load_image_from_data_url(data_url: str) -> Image.Image:
    return load_image_from_b64_any(data_url)

# ------------------------------------------------------------
# 색상 라이트닝
# ------------------------------------------------------------
def lighten_color(hex_color: str, factor: float) -> tuple:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return (r, g, b)

# ------------------------------------------------------------
# 카드 렌더링
# ------------------------------------------------------------
def draw_card(card: dict, width: int = 400, height: int = 560) -> Image.Image:
    card_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card_img)

    def text_wh(text: str, font: ImageFont.ImageFont):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)

    primary_type = (card.get('Types') or ['노말'])[0]
    base_color   = TYPE_COLORS.get(primary_type, '#A8A77A')
    rarity       = str(card.get('Rarity', 'Common'))
    rarity_levels = {'Common': 1, 'Uncommon': 2, 'Rare': 3, 'Epic': 4, 'Legendary': 5}
    level        = rarity_levels.get(rarity, 1)

    border_w     = 8 + (level - 1) * 2
    inner_margin = border_w + 4
    glitter_density = (level - 1) * 20
    border_color = lighten_color(base_color, min(0.1 * (level - 1), 0.5))

    # 배경/보더
    draw.rectangle([0, 0, width, height], fill=border_color)
    draw.rectangle([inner_margin, inner_margin, width - inner_margin, height - inner_margin], fill='#FDFDFD')

    # 글리터 효과
    for _ in range(glitter_density):
        x = random.randint(inner_margin, width - inner_margin)
        y = random.randint(inner_margin, height - inner_margin)
        size = random.randint(1, 3)
        if random.random() < 0.5:
            color = (255, 255, 255, random.randint(100, 200))
        else:
            color = (255, 255, random.randint(200, 255), random.randint(100, 200))
        draw.ellipse([x, y, x + size, y + size], fill=color)

    # 상단 바
    bar_h = 60
    bar_rect = [inner_margin + 2, inner_margin + 2, width - inner_margin - 2, inner_margin + bar_h]
    draw.rectangle(bar_rect, fill=base_color)

    # 폰트
    try:
        font_title    = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 24)
        font_subtitle = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 16)
        font_text     = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 14)
    except Exception:
        try:
            font_title    = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
            font_subtitle = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
            font_text     = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        except Exception:
            font_title = font_subtitle = font_text = ImageFont.load_default()

    # 이름/HP
    draw.text((inner_margin + 10, inner_margin + 15), card.get('Name', 'Unknown'), font=font_title, fill='black')
    hp_value = (card.get('Stats') or {}).get('HP', 0)
    hp_text  = f"HP {hp_value}"
    hp_w, _  = text_wh(hp_text, font_subtitle)
    draw.text((width - inner_margin - 10 - hp_w, inner_margin + 20), hp_text, font=font_subtitle, fill='black')

    # 아트 영역
    art_top = inner_margin + bar_h + 6
    art_h   = 240
    art_area = [inner_margin + 6, art_top, width - inner_margin - 6, art_top + art_h]

    art_img = None
    if card.get('artwork_image'):
        art_img = card['artwork_image']
    elif card.get('Image'):
        try:
            art_img = load_image_from_b64_any(card['Image'])
        except Exception:
            print(f"Warning: Could not decode embedded image for {card.get('Name')}")

    if art_img:
        try:
            art_img = art_img.convert("RGBA")
            art_ratio = art_img.width / art_img.height
            target_w  = art_area[2] - art_area[0]
            target_h  = art_area[3] - art_area[1]
            target_ratio = target_w / target_h
            if art_ratio > target_ratio:
                new_w, new_h = target_w, int(target_w / art_ratio)
            else:
                new_h, new_w = target_h, int(target_h * art_ratio)
            art_resized = art_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x_off = art_area[0] + (target_w - new_w) // 2
            y_off = art_area[1] + (target_h - new_h) // 2
            card_img.paste(art_resized, (x_off, y_off), art_resized)
        except Exception as e:
            print(f"Error pasting artwork for {card.get('Name')}: {e}")

    # 디테일 영역
    details_top = art_area[3] + 8
    tinted_color = lighten_color(base_color, 0.7)
    draw.rectangle([inner_margin, details_top, width - inner_margin, height - inner_margin], fill=tinted_color)

    # 타입 배지
    type_x = inner_margin + 10
    type_y = details_top + 4
    for t in (card.get('Types') or []):
        c = TYPE_COLORS.get(t, '#A8A77A')
        badge_w, badge_h = 70, 22
        badge_rect = [type_x, type_y, type_x + badge_w, type_y + badge_h]
        draw.rounded_rectangle(badge_rect, radius=8, fill=c, outline='black')
        tw, th = text_wh(t, font_text)
        draw.text((type_x + (badge_w - tw) / 2, type_y + (badge_h - th) / 2), t, font=font_text, fill='white')
        type_x += badge_w + 6

    # 스킬(최대 2개)
    skills = card.get('Skills') or []
    if skills:
        y_pos = type_y + 22 + 10
        draw.text((inner_margin + 10, y_pos), 'Skills:', font=font_subtitle, fill='black')
        y_pos += 22
        for skill in skills[:2]:
            name  = skill.get('Name', '?')
            stype = skill.get('Type', '?')
            draw.text((inner_margin + 20, y_pos), f"• {name} ({stype})", font=font_text, fill='black')
            y_pos += 16
            power = skill.get('Power')
            dmg_text = f"Damage: {power}" if power not in (None, -1) else "Damage: -"
            draw.text((inner_margin + 40, y_pos), dmg_text, font=font_text, fill='dimgray')
            y_pos += 18

            energy_cost = skill.get('EnergyCost') or skill.get('Energy') or skill.get('Energy Cost') or skill.get('Cost')
            try:
                energy_count = int(energy_cost)
            except Exception:
                energy_count = 0
            if energy_count > 0:
                icon_size = 12
                icon_x = inner_margin + 40
                icon_y = y_pos
                color_key = stype if stype in TYPE_COLORS else '공통'
                for idx in range(energy_count):
                    et_color = TYPE_COLORS.get(color_key, '#A8A77A')
                    draw.ellipse([icon_x + idx * (icon_size + 4), icon_y,
                                  icon_x + idx * (icon_size + 4) + icon_size,
                                  icon_y + icon_size], fill=et_color, outline='black')
                y_pos += icon_size + 6
            else:
                y_pos += 6
            y_pos += 8

    # 리트리트
    bottom_y = height - inner_margin - 40
    retreat = card.get('Retreat Cost', 0) or 0
    label   = 'Retreat:'
    lbl_w, _ = text_wh(label, font_text)
    rc_start = width - inner_margin - 10 - (retreat * 18)
    draw.text((rc_start - lbl_w - 4, bottom_y), label, font=font_text, fill='black')
    for i in range(retreat):
        cx = rc_start + i * 18
        cy = bottom_y + 5
        draw.ellipse([cx, cy, cx + 12, cy + 12], fill='gray', outline='black')

    return card_img

# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    # 입력 카드 데이터
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: Input file not found at {INPUT_JSON_PATH}")
        return
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        cards: List[Dict] = json.load(f)

    # image.json 로드 → 이름→이미지 맵
    image_index = load_image_index(IMAGE_JSON_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated = []
    for card in cards:
        # 매칭 키
        card_name = card.get('Name') or card.get('Korean Name') or 'Unknown'
        key = normalize_key(card_name)

        # 1) 로컬 파일 우선
        sanitized_name = "".join(c for c in (card.get('Name') or '') if c.isalnum())
        local_img_path = os.path.join(IMAGE_SRC_DIR, f"{sanitized_name}_temp.png")

        art_img = None
        if os.path.exists(local_img_path):
            try:
                art_img = Image.open(local_img_path).convert("RGBA")
                card['artwork_image'] = art_img
                print(f"[art] local file => {card_name} ({local_img_path})")
            except Exception as e:
                print(f"[art] fail local {local_img_path}: {e}")

        # 2) image.json (Base64) 사용
        if art_img is None and key in image_index:
            try:
                art_img = load_image_from_b64_any(image_index[key])
                card['artwork_image'] = art_img
                print(f"[art] image.json => {card_name}")
            except Exception as e:
                print(f"[art] fail image.json for {card_name}: {e}")

        # 3) better.json 내부의 Image 필드 (마지막 폴백)
        if art_img is None and card.get('Image'):
            try:
                card['artwork_image'] = load_image_from_b64_any(card['Image'])
                print(f"[art] embedded base64 => {card_name}")
            except Exception as e:
                print(f"[art] fail embedded for {card_name}: {e}")

        # 카드 출력
        img = draw_card(card)
        safe_name = (card.get('Name') or 'Unknown').replace(' ', '_')
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_improved.png")
        img.save(out_path)
        generated.append(out_path)

    print("\nGenerated card files:", generated)

if __name__ == '__main__':
    main()
