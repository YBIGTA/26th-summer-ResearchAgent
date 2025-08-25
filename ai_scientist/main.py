import base64
import json
import os
import random
import io
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import streamlit as st
import subprocess

# -----------------------------------------------------------------------------
# Colour definitions and helpers
#
# The card generator uses a simple mapping of Pokémon types to colours.  This
# dictionary maps both English and Korean type names to a hex colour.  When
# drawing cards the primary type is looked up to determine border and badge
# colours.  If an unknown type is encountered a neutral grey is used instead.
# -----------------------------------------------------------------------------

TYPE_COLORS: Dict[str, str] = {
    '전기': '#F7D02C', 'Electric': '#F7D02C',
    '물': '#6390F0',  'Water':    '#6390F0',
    '불꽃': '#EE8130', 'Fire':     '#EE8130',
    '풀': '#7AC74C',  'Grass':    '#7AC74C',
    '얼음': '#96D9D6', 'Ice':      '#96D9D6',
    '격투': '#C22E28', 'Fighting': '#C22E28',
    '독': '#A33EA1',  'Poison':   '#A33EA1',
    '땅': '#E2BF65',  'Ground':   '#E2BF65',
    '비행': '#A98FF3', 'Flying':   '#A98FF3',
    '에스퍼': '#F95587', 'Psychic': '#F95587',
    '벌레': '#A6B91A', 'Bug':      '#A6B91A',
    '바위': '#B6A136', 'Rock':     '#B6A136',
    '유령': '#735797', 'Ghost':    '#735797',
    '드래곤': '#6F35FC', 'Dragon':  '#6F35FC',
    '악': '#705746',  'Dark':     '#705746',
    '강철': '#B7B7CE', 'Steel':    '#B7B7CE',
    '페어리': '#D685AD','Fairy':    '#D685AD',
    '노말': '#A8A77A', 'Normal':   '#A8A77A',
    '공통': '#A8A77A',
}


def lighten_color(hex_color: str, factor: float) -> tuple:
    """Lighten a hex colour by blending it with white.

    Args:
        hex_color: Colour in "#RRGGBB" format.
        factor: Fractional amount to lighten (0–1).  A value of 0 returns the
            original colour, while 1 returns white.

    Returns:
        A 3‑tuple containing the lightened RGB values.
    """
    colour = hex_color.lstrip('#')
    r = int(colour[0:2], 16)
    g = int(colour[2:4], 16)
    b = int(colour[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return (r, g, b)


def load_image_from_data_url(data_url: str) -> Image.Image:
    """Load an image from a data URI.

    If the string cannot be decoded for any reason a 1×1 transparent image is
    returned instead.  This helps avoid crashes when optional artwork is
    malformed or missing.
    """
    try:
        header, b64data = data_url.split(',', 1)
        raw = base64.b64decode(b64data)
        return Image.open(io.BytesIO(raw)).convert('RGBA')
    except Exception:
        return Image.new('RGBA', (1, 1), (0, 0, 0, 0))


# -----------------------------------------------------------------------------
# Data models representing skills and cards
# -----------------------------------------------------------------------------

@dataclass
class Skill:
    """Representation of a move that can appear on a Pokémon card.

    The energy_cost is derived automatically from the power value unless it is
    explicitly provided.  A power of None or -1 denotes a non‑damaging move.
    """

    name: str
    type: str
    power: Optional[int]
    accuracy: Optional[float]
    pp: Optional[int]
    effect: Optional[str] = None
    energy_cost: int = 0

    def __post_init__(self) -> None:
        p = self.power
        if p is None or p == -1:
            self.energy_cost = 0
        elif p <= 60:
            self.energy_cost = 1
        elif p <= 90:
            self.energy_cost = 2
        else:
            self.energy_cost = 3

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Type": self.type,
            "Power": self.power,
            "Accuracy": self.accuracy,
            "EnergyCost": self.energy_cost,
            "Effect": self.effect,
        }


class PokemonCard:
    """Internal class representing a Pokémon card's data.

    It converts raw JSON entries into strongly typed attributes, computes
    derived values such as retreat cost and power score, and assigns a rarity
    externally after sorting by power.
    """

    def __init__(self, entry: Dict[str, Any]):
        ch = entry.get('character', entry)
        self.name: str = ch.get('Name', 'Unknown')
        typing = ch.get('Typing', [])
        if isinstance(typing, list):
            self.types: List[str] = typing
        elif typing:
            self.types = [typing]
        else:
            self.types = []
        self.image: Optional[str] = ch.get('Image')
        raw_stats = ch.get('Stats', {}) or {}
        self.stats: Dict[str, int] = {}
        for key, value in raw_stats.items():
            if isinstance(value, (int, float)):
                self.stats[key] = int(value)
            elif isinstance(value, str):
                try:
                    self.stats[key] = int(value)
                except ValueError:
                    self.stats[key] = 0
            else:
                self.stats[key] = 0
        self.abilities: List[str] = ch.get('Abilities', []) or []
        self.skills: List[Skill] = []
        sig = ch.get('Signature Move')
        if sig:
            power = sig.get('Power')
            if isinstance(power, str):
                try:
                    power = int(power)
                except Exception:
                    power = None
            acc = sig.get('Accuracy')
            acc_val: Optional[float] = None
            if isinstance(acc, str) and acc.endswith('%'):
                try:
                    acc_val = float(acc.replace('%', ''))
                except Exception:
                    acc_val = None
            self.skills.append(
                Skill(
                    name=sig.get('Name', 'Unknown Move'),
                    type=sig.get('Type', self.types[0] if self.types else 'Normal'),
                    power=power,
                    accuracy=acc_val,
                    pp=sig.get('PP'),
                    effect=sig.get('Effect'),
                )
            )
        attack = self.stats.get('Attack', 0)
        sp_atk = self.stats.get('Sp.Atk', 0)
        base_power = max(attack, sp_atk) // 2
        base_power = max(40, min(base_power, 120))
        base_accuracy = 100
        base_pp = 20
        base_type = self.types[0] if self.types else 'Normal'
        for mv in ch.get('Movepool Highlights', []) or []:
            self.skills.append(
                Skill(
                    name=mv,
                    type=base_type,
                    power=base_power,
                    accuracy=base_accuracy,
                    pp=base_pp,
                    effect=None,
                )
            )
        self.total_stats: int = raw_stats.get('Total') or sum(
            self.stats.get(stat, 0) for stat in ['HP','Attack','Defense','Sp.Atk','Sp.Def','Speed']
        )
        self.power_score: float = (
            1.2 * self.stats.get('Attack', 0) +
            1.2 * self.stats.get('Sp.Atk', 0) +
            1.0 * self.stats.get('Defense', 0) +
            1.0 * self.stats.get('Sp.Def', 0) +
            0.8 * self.stats.get('HP', 0) +
            1.0 * self.stats.get('Speed', 0)
        )
        hp = self.stats.get('HP', 50)
        if hp < 60:
            self.retreat: int = 1
        elif hp < 100:
            self.retreat = 2
        else:
            self.retreat = 3
        self.rarity: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Types": self.types,
            "Stats": self.stats,
            "Total Stats": self.total_stats,
            "Power Score": round(self.power_score, 2),
            "Retreat Cost": self.retreat,
            "Rarity": self.rarity,
            "Abilities": self.abilities,
            # Limit skills to first two for display purposes
            "Skills": [sk.as_dict() for sk in self.skills[:2]],
            "Image": self.image,
        }


def draw_card(card: Dict[str, Any], width: int = 400, height: int = 560) -> Image.Image:
    """Render a single Pokémon card to a Pillow image.

    The layout follows a standard card format with a coloured header, artwork
    panel, type badges, skill descriptions and retreat cost icons.  The
    colours are derived from the primary type of the card.
    """
    card_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card_img)

    def text_wh(text: str, font: ImageFont.ImageFont) -> tuple:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)

    primary_type = card.get('Types')[0] if card.get('Types') else '노말'
    base_colour = TYPE_COLORS.get(primary_type, '#A8A77A')
    rarity = str(card.get('Rarity', 'Common'))
    rarity_levels = {'Common': 1, 'Uncommon': 2, 'Rare': 3, 'Epic': 4, 'Legendary': 5}
    level = rarity_levels.get(rarity, 1)

    # Border width and inner margin scale with rarity
    border_w = 8 + (level - 1) * 2
    inner_margin = border_w + 4
    glitter_density = (level - 1) * 20

    # Lighten the border colour relative to the base colour and rarity
    border_colour = lighten_color(base_colour, min(0.1 * (level - 1), 0.5))

    # Draw border and base rectangle
    draw.rectangle([0, 0, width, height], fill=border_colour)
    draw.rectangle([inner_margin, inner_margin, width - inner_margin, height - inner_margin], fill='#FDFDFD')

    # Sprinkle glitter in the background
    for _ in range(glitter_density):
        x = random.randint(inner_margin, width - inner_margin)
        y = random.randint(inner_margin, height - inner_margin)
        size = random.randint(1, 3)
        if random.random() < 0.5:
            colour = (255, 255, 255, random.randint(100, 200))
        else:
            colour = (255, 255, random.randint(200, 255), random.randint(100, 200))
        draw.ellipse([x, y, x + size, y + size], fill=colour)

    # Top bar for name and HP
    bar_h = 60
    bar_rect = [inner_margin + 2, inner_margin + 2, width - inner_margin - 2, inner_margin + bar_h]
    draw.rectangle(bar_rect, fill=base_colour)

    # Load fonts; fall back gracefully if not available
    try:
        font_title = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 24)
        font_subtitle = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 16)
        font_text = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 14)
    except Exception:
        try:
            font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
            font_subtitle = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
            font_text = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        except Exception:
            font_title = font_subtitle = font_text = ImageFont.load_default()

    # Draw name and HP on top bar
    draw.text((inner_margin + 10, inner_margin + 15), card.get('Name', 'Unknown'), font=font_title, fill='black')
    hp_value = (card.get('Stats') or {}).get('HP', 0)
    hp_text = f"HP {hp_value}"
    hp_w, _ = text_wh(hp_text, font_subtitle)
    draw.text((width - inner_margin - 10 - hp_w, inner_margin + 20), hp_text, font=font_subtitle, fill='black')

    # Artwork area
    art_top = inner_margin + bar_h + 6
    art_h = 240
    art_area = [inner_margin + 6, art_top, width - inner_margin - 6, art_top + art_h]
    if card.get('Image'):
        try:
            art_img = load_image_from_data_url(card['Image'])
            # Fit image into the artwork area while maintaining aspect ratio
            art_ratio = art_img.width / art_img.height
            target_w = art_area[2] - art_area[0]
            target_h = art_area[3] - art_area[1]
            target_ratio = target_w / target_h
            if art_ratio > target_ratio:
                new_w, new_h = target_w, int(target_w / art_ratio)
            else:
                new_h, new_w = target_h, int(target_h * art_ratio)
            art_resized = art_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x_off = art_area[0] + (target_w - new_w) // 2
            y_off = art_area[1] + (target_h - new_h) // 2
            card_img.paste(art_resized, (x_off, y_off), art_resized)
        except Exception:
            pass

    # Details area background tinted version of base colour
    details_top = art_area[3] + 8
    tinted_colour = lighten_color(base_colour, 0.7)
    draw.rectangle([inner_margin, details_top, width - inner_margin, height - inner_margin], fill=tinted_colour)

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

    # Skills section (limit to two skills)
    skills = card.get('Skills') or []
    if skills:
        y_pos = type_y + 22 + 10
        draw.text((inner_margin + 10, y_pos), 'Skills:', font=font_subtitle, fill='black')
        y_pos += 22
        for skill in skills[:2]:
            name = skill.get('Name', '?')
            stype = skill.get('Type', '?')
            draw.text((inner_margin + 20, y_pos), f"• {name} ({stype})", font=font_text, fill='black')
            y_pos += 16
            power = skill.get('Power')
            dmg_text = f"Damage: {power}" if power not in (None, -1) else "Damage: -"
            draw.text((inner_margin + 40, y_pos), dmg_text, font=font_text, fill='dimgray')
            y_pos += 18
            # Draw energy cost as coloured circles
            energy_cost = (
                skill.get('EnergyCost') or
                skill.get('Energy') or
                skill.get('Energy Cost') or
                skill.get('Cost')
            )
            try:
                energy_count = int(energy_cost)
            except Exception:
                energy_count = 0
            if energy_count > 0:
                icon_size = 12
                icon_x = inner_margin + 40
                icon_y = y_pos
                colour_key = stype if stype in TYPE_COLORS else '공통'
                for idx in range(energy_count):
                    et_colour = TYPE_COLORS.get(colour_key, '#A8A77A')
                    draw.ellipse([
                        icon_x + idx * (icon_size + 4), icon_y,
                        icon_x + idx * (icon_size + 4) + icon_size,
                        icon_y + icon_size
                    ], fill=et_colour, outline='black')
                y_pos += icon_size + 6
            else:
                y_pos += 6
            # Space before next skill
            y_pos += 8

    # Draw retreat cost at bottom of card
    bottom_y = height - inner_margin - 40
    retreat = card.get('Retreat Cost', 0) or 0
    label = 'Retreat:'
    lbl_w, _ = text_wh(label, font_text)
    rc_start = width - inner_margin - 10 - (retreat * 18)
    draw.text((rc_start - lbl_w - 4, bottom_y), label, font=font_text, fill='black')
    for i in range(retreat):
        cx = rc_start + i * 18
        cy = bottom_y + 5
        draw.ellipse([cx, cy, cx + 12, cy + 12], fill='gray', outline='black')

    return card_img


def generate_pokemon_cards(input_json_path: str) -> List[Dict[str, Any]]:
    """Load raw Pokémon specifications and assign rarity and derived stats.

    The returned list preserves the original ordering of the input JSON but
    populates the `rarity` field based on power ranking.  Legendary cards are
    the top 10 %, rare cards up to 40 %, and the remainder common.  The
    function will raise if the input file cannot be found or parsed.
    """
    if not os.path.isfile(input_json_path):
        raise FileNotFoundError(f"Input file '{input_json_path}' does not exist.")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # Generate card objects
    cards = [PokemonCard(entry) for entry in raw]
    # Sort by power score descending
    sorted_cards = sorted(cards, key=lambda c: c.power_score, reverse=True)
    total = len(sorted_cards)
    for idx, card in enumerate(sorted_cards):
        if idx < total * 0.1:
            card.rarity = 'Legendary'
        elif idx < total * 0.4:
            card.rarity = 'Rare'
        else:
            card.rarity = 'Common'
    # Return list of dictionaries in original order
    return [card.as_dict() for card in cards]


# -----------------------------------------------------------------------------
# Streamlit user interface
# -----------------------------------------------------------------------------

def build_app() -> None:
    """Construct the Streamlit UI for generating ideas and visualising cards."""
    st.title("Pokémon Card Generator & Visualiser")
    st.markdown(
        "Generate custom Pokémon card specifications from a dataset and render\n"
        "them as stylised trading cards.  Begin by optionally creating new ideas,\n"
        "then assign rarities and create artwork.  Finally select the cards\n"
        "you wish to see and click *Draw Selected Cards* to view them."
    )

    # Maintain generated card data in session state
    if 'cards_data' not in st.session_state:
        st.session_state.cards_data: Optional[List[Dict[str, Any]]] = None

    # Create tabs for each major step of the workflow.  Using tabs helps
    # organise the user interface so that ideation, card specification and
    # visualisation are clearly separated.
    tabs = st.tabs(["Ideation", "Generate Cards", "Visualise Cards"])

    # -------------------------------------------------------------------------
    # Tab 0 – Ideation 
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.header("Generate Pokémon Ideas")
        st.markdown(
            "Use the ideation script to produce brand new Pokémon concepts. "
            "This step is optional – you can also provide your own JSON file."
        )
        num_ideas = st.number_input(
            "Number of Pokémon ideas to generate", min_value=1, max_value=50, value=5,
            help="How many new Pokémon should the ideation script produce?"
        )
        default_idea_out = os.path.join(os.path.dirname(__file__), 'ideas', 'generated_pokemon.json')
        idea_output = st.text_input(
            "Path to save generated ideas", value=default_idea_out,
            help="Relative or absolute path where the ideation script will write JSON."
        )
        if st.button("Run Ideation"):
            out_dir = os.path.dirname(idea_output)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            try:
                script_path = os.path.join(os.path.dirname(__file__), 'perform_ideation_temp_free.py')
                if os.path.isfile(script_path):
                    cmd = [
                        'python',
                        script_path,
                        '--num_pokemon', str(int(num_ideas)),
                        '--output', idea_output,
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        st.error(f"Ideation script failed: {result.stderr.strip()}")
                    else:
                        st.success(f"Successfully generated {num_ideas} ideas to {idea_output}.")
                else:
                    dummy_char = {
                        "Name": "Testmon",
                        "Typing": ["노말"],
                        "Stats": {"HP": 50, "Attack": 50, "Defense": 50, "Sp.Atk": 50, "Sp.Def": 50, "Speed": 50},
                        "Abilities": ["Overgrow"],
                        "Signature Move": {"Name": "Test Move", "Type": "노말", "Power": 60, "Accuracy": 100, "PP": 20, "Effect": "None"},
                        "Movepool Highlights": ["Tackle", "Growl"],
                    }
                    with open(idea_output, 'w', encoding='utf-8') as f:
                        json.dump([dummy_char], f, ensure_ascii=False, indent=2)
                    st.info(
                        "Ideation script not found; wrote a dummy idea instead. "
                        "You can replace this file with your own ideas."
                    )
            except Exception as e:
                st.error(f"Failed to run ideation: {e}")
            # Display the newly generated ideas in the UI
            if os.path.isfile(idea_output):
                try:
                    with open(idea_output, 'r', encoding='utf-8') as f:
                        idea_data = json.load(f)
                    st.success(f"Loaded {len(idea_data)} ideas from {idea_output}.")
                    st.json(idea_data)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Tab 1 – Generate Card Specifications
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.header("Generate Card Specifications")
        default_path = os.path.join(os.path.dirname(__file__), 'ideas', 'i_cant_believe_its_not_better_image.json')
        input_path = st.text_input(
            "Path to raw Pokémon specification JSON", value=default_path,
            help="Relative or absolute path to the JSON file containing base Pokémon data."
        )
        if st.button("Generate Specifications"):
            try:
                cards = generate_pokemon_cards(input_path)
                st.session_state.cards_data = cards
                st.success(f"Generated {len(cards)} card specifications.")
                summary_records = [
                    {
                        'Name': c['Name'],
                        'Types': ', '.join(c.get('Types', [])),
                        'Power Score': c.get('Power Score'),
                        'Rarity': c.get('Rarity'),
                        'Retreat Cost': c.get('Retreat Cost'),
                    }
                    for c in cards
                ]
                summary_df = pd.DataFrame(summary_records)
                st.dataframe(summary_df, use_container_width=True)
                # Allow user to download the generated JSON
                json_str = json.dumps(cards, ensure_ascii=False, indent=2)
                st.download_button(
                    label="Download Cards JSON",
                    data=json_str,
                    file_name="pokemon_cards_output.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Failed to generate cards: {e}")

    # -------------------------------------------------------------------------
    # Tab 2 – Select & Visualise Cards
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.header("Select & Visualise Cards")
        cards_data: Optional[List[Dict[str, Any]]] = st.session_state.get('cards_data')
        if not cards_data:
            st.info("Please generate card specifications in the previous tab before proceeding.")
        else:
            # Provide selection of cards by name
            card_names = [card['Name'] for card in cards_data]
            selected_names = st.multiselect(
                "Select Pokémon to draw", card_names,
                help="Choose one or more cards from the generated list to render."
            )
            if selected_names and st.button("Draw Selected Cards"):
                # Render selected cards
                for card in cards_data:
                    if card['Name'] in selected_names:
                        img = draw_card(card)
                        # Convert to bytes for Streamlit
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        st.image(buf.getvalue(), caption=f"{card['Name']} ({card.get('Rarity', 'Unknown')})", use_column_width=False)


def main() -> None:
    """Entrypoint used by Streamlit when running ``streamlit run main.py``."""
    build_app()


if __name__ == '__main__':
    main()
