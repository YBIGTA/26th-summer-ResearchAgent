from bs4 import BeautifulSoup

from util import get_raw_data

import logging

logger = logging.getLogger(__name__)


def parse_pokemon_stage(pokemon_td) -> dict:
    """
    Extracts name, types, and image from an evolution (main column cell).
    """
    # Find the nested table within the column
    nested_table = pokemon_td.find("table")
    if not nested_table:
        return None
    
    tds = nested_table.find_all("td")

    # Find the evolution stage
    evolution_stage = tds[1].get_text().strip()
        
    # Find the name (it can be in a <strong> or <a> tag)
    name_tag = tds[2].find("span")
    name = name_tag.get_text(strip=True) if name_tag else "N/A"
    
    # Find all type spans
    type_tags = nested_table.select(".split-cell.text-white")
    types = [tag.get_text(strip=True) for tag in type_tags]
    
    # Find the image
    img_tag = nested_table.find("img")
    image_url = None
    if img_tag and img_tag.has_attr('data-src'):
        image_url = get_raw_data(img_tag['data-src'])
    
    return {
        "이름": name, 
        "타입": types, 
        "이미지": image_url, 
        "스테이지": evolution_stage
    }

def extract_evolutions(soup: BeautifulSoup) -> list:
    """
    Extract evolution chain(s) with level and image rawdata from the Pokémon page.
    """
    data = []
    
    evolution_header = soup.find("span", id="진화_단계")
    if not evolution_header:
        return []

    # Look for the evolution table
    evolution_table = evolution_header.find_parent("h3").find_next_sibling()
    if not evolution_table:
        return []
    elif evolution_table.name == "h4":
        evolution_table = evolution_table.find_next_sibling() # sometimes there is a subheading under 진화 단계
    elif evolution_table.name != "table":
        evolution_table.find("table") # if the next sibling is not a table, assume its a div with a table inside
    
    rows = evolution_table.find("tbody").find_all("tr", recursive=False)

    if len(rows) != 3: # check for simple evolution key case
        # -- SIMPLE EVOLUTION TREE PARSING --
        main_columns = evolution_table.find("tr").find_all("td", recursive=False)
        next_trigger = None

        for column in main_columns:
            # Check if the column is a Pokémon stage (contains a nested table)
            if column.find("table"):
                pokemon_data = parse_pokemon_stage(column)

                if next_trigger:
                    pokemon_data["진화 방법"] = next_trigger
                    next_trigger = None

                data.append(pokemon_data)
            # Check if the column is an evolution trigger (contains the arrow)
            elif "→" in column.get_text():
                trigger_text = column.get_text(strip=True)
                next_trigger = trigger_text[:-1].strip()

        return data
    
    # -- SPECIAL EVOLUTION TREE PARSING --
    logger.info("Found special evolution table, switching evolution tree parser")
    
    # 1. Get the base Pokémon from the first row
    base_pokemon_cell = rows[0].find("td")
    base_pokemon = parse_pokemon_stage(base_pokemon_cell)

    data.append(base_pokemon)
    
    # 2. Get the list of evolution conditions and results
    condition_cells = rows[1].find_all("td", recursive=False)
    result_cells = rows[2].find_all("td", recursive=False)

    # 3. Pair each condition with its result using zip()
    for condition_cell, result_cell in zip(condition_cells, result_cells):
        
        # Parse the evolved Pokémon
        evolved_pokemon = parse_pokemon_stage(result_cell)
        
        # Get the clean trigger text from the condition cell
        for unwanted_span in condition_cell.find_all("span", typeof="mw:File"):
            unwanted_span.decompose() # remove picture tags

        trigger_texts = []

        td_elements = condition_cell.find_all("td")
        if len(td_elements) == 0:
            trigger_text = " ".join([text for text in condition_cell.stripped_strings if text != '↓'])

            # If there are two '+', replace only the first one
            if trigger_text.count('+') == 2:
                trigger_text = trigger_text.replace('+', '', 1).strip()
        else:
            for td in td_elements: # may be within table, but is single row
                text = " ".join(td.stripped_strings)

                if text != "↓" and text != "→":
                    trigger_texts.append(text)

            # Using .stripped_strings joins text elements separated by <br>
            trigger_text = ", ".join(trigger_texts)
        
        # Create a complete evolution path entry
        evolved_pokemon["진화 방법"] = trigger_text
        data.append(evolved_pokemon)

    return data