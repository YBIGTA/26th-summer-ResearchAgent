from bs4 import BeautifulSoup

from util import get_raw_data


def parse_pokemon_stage(pokemon_td):
    """Extracts name, types, and image from a main column cell."""
    # Find the nested table within the column
    nested_table = pokemon_td.find("table")
    if not nested_table:
        return None
        
    # Find the name (it can be in a <strong> or <a> tag)
    name_tag = nested_table.find_all('td')[2].find('span')
    name = name_tag.get_text(strip=True) if name_tag else "N/A"
    
    # Find all type spans
    type_tags = nested_table.select(".split-cell.text-white")
    types = [tag.get_text(strip=True) for tag in type_tags]
    
    # Find the image
    img_tag = nested_table.find("img")
    image_url = None
    if img_tag and img_tag.has_attr('data-src'):
        image_url = get_raw_data(img_tag['data-src'])
    
    return {"이름": name, "타입": types, "이미지": image_url}

def extract_evolutions(soup: BeautifulSoup) -> list:
    """
    Extract evolution chain with level and image rawdata from the Pokémon page.
    """
    data = []
    
    # Evolution chain tables look like centered tables with arrows
    evolution_header = soup.find("span", id="진화_단계")

    if evolution_header:
        # 2. The main evolution table is the next sibling of the <h3> tag
        evolution_table = evolution_header.find_parent("h3").find_next_sibling("table")
        
        # 3. Find ONLY the direct children <td> tags of the first <tr>
        # This is the key fix: recursive=False stops it from going into the nested tables.
        main_columns = evolution_table.find("tr").find_all("td", recursive=False)
        
        for column in main_columns:
            # Check if the column is a Pokémon stage (contains a nested table)
            if column.find("table"):
                pokemon_data = parse_pokemon_stage(column)
                data.append(pokemon_data)
            # Check if the column is an evolution trigger (contains the arrow)
            elif "→" in column.get_text():
                trigger_text = column.get_text(strip=True)
                # Add the trigger info to the PREVIOUS Pokémon in the chain
                if data:
                    data[-1]['진화 방법'] = trigger_text[:-1]

    return data