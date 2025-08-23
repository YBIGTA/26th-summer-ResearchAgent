import os
import json
import base64
import io
import random
from PIL import Image, ImageDraw, ImageFont

# Path to the input JSON containing card definitions.
INPUT_JSON_PATH = "ideas/pokemon_cards_output.json"

# Output directory for improved card images.
OUTPUT_DIR = "ideas/improved_card_images"

# catch-all for colorless energy (`공통`).
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


def load_image_from_data_url(data_url: str) -> Image.Image:
    """Decode a base64 data URL into a PIL Image."""
    header, b64data = data_url.split(',', 1)
    raw = base64.b64decode(b64data)
    return Image.open(io.BytesIO(raw)).convert('RGBA')


def lighten_color(hex_color: str, factor: float) -> tuple:
    """Lighten a hex RGB color by blending it with white.

    Args:
        hex_color: Color in '#RRGGBB' form.
        factor: Amount to lighten (0–1). 0 returns the original color,
                and 1 returns white.

    Returns:
        A 3-tuple with the lightened RGB values.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return (r, g, b)


def draw_card(card: dict, width: int = 400, height: int = 560) -> Image.Image:
    """Draw a single Pokémon card according to the given specification.

    The card design adapts to the primary type and rarity, applies a sparkly
    border for higher rarities, separates the artwork and details areas, and
    shows energy costs as colored circles next to each skill.

    Args:
        card: A dictionary containing card information (Name, Types, Rarity,
              Skills, Image, etc.).
        width: Width of the resulting card image.
        height: Height of the resulting card image.

    Returns:
        A PIL Image representing the rendered card.
    """
    # Create a transparent canvas
    card_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card_img)

    # Helper for text size
    def text_wh(text: str, font: ImageFont.ImageFont):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)

    # Determine primary type and base color
    primary_type = card.get('Types')[0] if card.get('Types') else '노말'
    base_color   = TYPE_COLORS.get(primary_type, '#A8A77A')
    # Determine rarity level
    rarity       = str(card.get('Rarity', 'Common'))
    rarity_levels = {'Common': 1, 'Uncommon': 2, 'Rare': 3, 'Epic': 4, 'Legendary': 5}
    level        = rarity_levels.get(rarity, 1)

    # Border and glitter settings
    border_w     = 8 + (level - 1) * 2
    inner_margin = border_w + 4
    glitter_density = (level - 1) * 20

    # Lighten the border color based on rarity
    border_color = lighten_color(base_color, min(0.1 * (level - 1), 0.5))

    # Draw border and base
    draw.rectangle([0, 0, width, height], fill=border_color)
    draw.rectangle([inner_margin, inner_margin, width - inner_margin, height - inner_margin], fill='#FDFDFD')

    # Draw glitter behind the artwork and details areas
    for _ in range(glitter_density):
        x = random.randint(inner_margin, width - inner_margin)
        y = random.randint(inner_margin, height - inner_margin)
        size = random.randint(1, 3)
        if random.random() < 0.5:
            color = (255, 255, 255, random.randint(100, 200))
        else:
            color = (255, 255, random.randint(200, 255), random.randint(100, 200))
        draw.ellipse([x, y, x + size, y + size], fill=color)

    # Draw the top bar
    bar_h = 60
    bar_rect = [inner_margin + 2, inner_margin + 2, width - inner_margin - 2, inner_margin + bar_h]
    draw.rectangle(bar_rect, fill=base_color)

    # Load fonts (prefer NotoSansCJK for multi-language support)
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

    # Draw the name and HP
    draw.text((inner_margin + 10, inner_margin + 15), card.get('Name', 'Unknown'), font=font_title, fill='black')
    hp_value = (card.get('Stats') or {}).get('HP', 0)
    hp_text  = f"HP {hp_value}"
    hp_w, _  = text_wh(hp_text, font_subtitle)
    draw.text((width - inner_margin - 10 - hp_w, inner_margin + 20), hp_text, font=font_subtitle, fill='black')

    # Determine artwork area and paste the artwork image if available
    art_top = inner_margin + bar_h + 6
    art_h   = 240
    art_area = [inner_margin + 6, art_top, width - inner_margin - 6, art_top + art_h]
    if card.get('Image'):
        try:
            art_img = load_image_from_data_url(card['Image'])
            art_ratio = art_img.width / art_img.height
            target_w = art_area[2] - art_area[0]
            target_h = art_area[3] - art_area[1]
            target_ratio = target_w / target_h
            if art_ratio > target_ratio:
                # Fit to width
                new_w, new_h = target_w, int(target_w / art_ratio)
            else:
                # Fit to height
                new_h, new_w = target_h, int(target_h * art_ratio)
            art_resized = art_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x_off = art_area[0] + (target_w - new_w) // 2
            y_off = art_area[1] + (target_h - new_h) // 2
            card_img.paste(art_resized, (x_off, y_off), art_resized)
        except Exception:
            pass

    # Fill details area with a pale tint of the base color
    details_top = art_area[3] + 8  # start below the artwork
    tinted_color = lighten_color(base_color, 0.7)
    draw.rectangle([inner_margin, details_top, width - inner_margin, height - inner_margin], fill=tinted_color)

    # Draw type badges
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

    # Draw the Skills section (limit to two skills)
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
            # Extract energy cost (integer) and draw colored circles
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
            # Space before the next skill
            y_pos += 8

    # Draw retreat cost at the bottom of the card
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


def main() -> None:
    """Main entry point for generating improved cards."""
    # Load card data from JSON
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate and save each card
    generated = []
    for card in cards:
        img = draw_card(card)
        safe_name = card.get('Name', 'Unknown').replace(' ', '_')
        filename = f"{safe_name}_improved.png"
        path = os.path.join(OUTPUT_DIR, filename)
        img.save(path)
        generated.append(path)
    print("Generated card files:", generated)


if __name__ == '__main__':
    main()