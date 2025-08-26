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

def draw_card(card: Dict[str, Any]) -> str:
    """Populates the HTML template with card data and returns the HTML string."""
    
    # Read the template file
    try:
        with open('card_template.html', 'r', encoding='utf-8') as f:
            html_template = f.read()
    except FileNotFoundError:
        return "<h3>Error: card_template.html not found.</h3><p>Please ensure the template file is in the same directory as the script.</p>"

    # --- Prepare data snippets for injection ---
    
    # Types
    types_html = ""
    for t in card.get('Types', []):
        color = TYPE_COLORS.get(t, '#666')
        types_html += f'<div class="type-badge" style="background-color: {color};">{t}</div>'

    # Skills - CORRECTED KEY ACCESS
    skills_html = ""
    for skill in card.get('Skills', []):
        # CORRECTED: Use 'Power', 'Name', and 'Effect' (capitalized)
        power = skill.get('Power')
        power_text = f" (Power: {power})" if power is not None else ""
        skill_name = skill.get('Name', 'Unknown')
        skill_effect = skill.get('Effect', 'No effect description.')
        
        skills_html += f"""
        <div class="skill">
            <span class="skill-name">• {skill_name}{power_text}:</span>
            <span>{skill_effect}</span>
        </div>
        """
        
    # Retreat Cost
    retreat_html = ""
    for _ in range(card.get('Retreat Cost', 0)):
        retreat_html += '<div class="retreat-icon"></div>'

    # Colors
    primary_type = card.get('Types')[0] if card.get('Types') else 'Normal'
    base_color = TYPE_COLORS.get(primary_type, '#A8A77A')
    # Create a tinted background color with transparency (alpha)
    tinted_color = f"{base_color}40" # Adding '40' creates ~25% opacity

    # --- Inject all data into the template ---
    html_content = html_template.replace('{{ NAME }}', card.get('Name', 'Unknown'))
    html_content = html_content.replace('{{ HP }}', str(card.get('Stats', {}).get('HP', 0)))
    html_content = html_content.replace('{{ RARITY }}', card.get('Rarity', 'Common'))
    html_content = html_content.replace('{{ IMAGE_DATA_URL }}', card.get('Image') or '')
    html_content = html_content.replace('{{ TYPES_HTML }}', types_html)
    html_content = html_content.replace('{{ SKILLS_HTML }}', skills_html)
    html_content = html_content.replace('{{ RETREAT_HTML }}', retreat_html)
    html_content = html_content.replace('{{ BASE_COLOR }}', base_color)
    html_content = html_content.replace('{{ BORDER_COLOR }}', base_color)
    html_content = html_content.replace('{{ TINTED_COLOR }}', tinted_color)
    

    return html_content
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
        "them as stylised trading cards.  Begin by optionally creating new ideas,\n"
        "then assign rarities and create artwork.  Finally select the cards\n"
        "you wish to see and click *Draw Selected Cards* to view them."
    )

    if 'cards_data' not in st.session_state:
        st.session_state.cards_data: Optional[List[Dict[str, Any]]] = None

    tabs = st.tabs(["1. Generate Pokémon Data", "2. Generate Pokémon Card"])
    
    # -------------------------------------------------------------------------
    # Tab 1 – Generate Pokémon (Formerly Ideation)
    # -------------------------------------------------------------------------
    with tabs[0]:
        with st.container(border=True):
            st.header("1. Generate New Pokémon Concepts")
            st.markdown(f"Select how many new Pokémon to generate from your `{os.path.abspath('./crawler/json')}` folder.")
            
            num_ideas = st.number_input("Number of Pokémon to generate", min_value=1, max_value=20, value=3)
            default_idea_out = os.path.join(os.path.dirname(__file__), 'generated_pokemon', 'new_creations.json')
            idea_output = st.text_input("Path to save final JSON file", value=default_idea_out)

            if st.button("✨ Run Generation Script", type="primary"):
                st.session_state.generated_ideas = [] # Clear previous results
                
                script_path = os.path.join(os.path.dirname(__file__), 'refactor_pokemon.py')
                cmd = ['python', script_path, '--count', str(num_ideas), '--output', idea_output]
                
                st.subheader("Generation Progress")
                progress_placeholder = st.empty()
                log_placeholder = st.expander("Show Live Logs", expanded=True)
                log_container = log_placeholder.empty()

                log_lines = []

                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, # This is the key change
                        text=True,
                        encoding='utf-8',
                        env={**os.environ, "PYTHONUTF8": "1"}
                    )

                    for line in iter(process.stdout.readline, ''):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Try to parse the line as JSON. If it works, it's a result.
                            new_pokemon = json.loads(line)
                            st.session_state.generated_ideas.append(new_pokemon)
                            
                            with progress_placeholder.container():
                                st.success(f"Generated {len(st.session_state.generated_ideas)} / {num_ideas} Pokémon...")
                                st.json(new_pokemon) # Display the latest one
                        except json.JSONDecodeError:
                            # If it's not JSON, it must be a log message.
                            log_lines.append(line)
                            log_container.code("\n".join(log_lines), language="log")

                    process.wait() # Wait for the process to finish
                    if process.returncode != 0:
                        st.error("The generation script encountered an error. Check logs above.")
                    else:
                        progress_placeholder.success(f"Generation complete! {len(st.session_state.generated_ideas)} Pokémon were created.")
                
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # -------------------------------------------------------------------------
    # Tab 2 – Generate Card Specifications, Select & Visualise Cards
    # -------------------------------------------------------------------------
    with tabs[1]:
        # --- Container 2: Finalize & Assign Rarity ---
        with st.container(border=True):
            st.subheader("2. Finalize Specs & Assign Rarity")
            st.markdown("This step loads a JSON file (like the one from the previous step), calculates power scores, and assigns a rarity to each Pokémon.")
            default_path = os.path.join(os.path.dirname(__file__), 'generated_pokemon', 'new_creations.json')
            input_path = st.text_input("Path to generated Pokémon JSON", value=default_path, key="finalize_input")
            
            if st.button("Generate Specifications", type="primary"):
                try:
                    cards = generate_pokemon_cards(input_path)
                    st.session_state.cards_data = cards
                    st.success(f"Processed {len(cards)} Pokémon and assigned rarities.")
                    summary_records = [{'Name': c['Name'], 'Types': ', '.join(c.get('Types', [])), 'Power Score': c.get('Power Score'), 'Rarity': c.get('Rarity')} for c in cards]
                    summary_df = pd.DataFrame(summary_records)
                    st.dataframe(summary_df, use_container_width=True)
                    json_str = json.dumps(cards, ensure_ascii=False, indent=2)
                    st.download_button(label="Download Finalized Card Data (JSON)", data=json_str, file_name="final_pokemon_cards.json", mime="application/json")
                except Exception as e:
                    st.error(f"Failed to process cards: {e}")

        st.divider()

        # --- Container 3: Draw Cards ---
        with st.container(border=True):
            st.subheader("3. Draw Pokémon Cards")
            cards_data: Optional[List[Dict[str, Any]]] = st.session_state.get('cards_data')
            if not cards_data:
                st.info("Please generate card specifications in Step 2 before proceeding.")
            else:
                # Define number of columns for the grid
                num_columns = 2
                
                # Create a list of columns
                cols = st.columns(num_columns)
                
                # Distribute cards into the columns
                for i, card in enumerate(cards_data):
                    with cols[i % num_columns]:
                        card_html = draw_card(card)
                        st.html(card_html)

def main() -> None:
    build_app()

if __name__ == '__main__':
    main()
