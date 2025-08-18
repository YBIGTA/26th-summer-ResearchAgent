from bs4 import BeautifulSoup


def extract_abilities(soup: BeautifulSoup):
    """
    Extracts abilities data from the Pokémon page
    """
    stats_table = None
    stats_data = {}

    # --- 1. Find the specific header with table
    abilities_header = soup.find("span", id="종족값,_노력치")

    if abilities_header:
        # 2. The main evolution table is the next sibling of the <h3> tag
        stats_table = abilities_header.find_parent("h4").find_next_sibling("table")
    
    # --- 2. Find all table rows, skipping the first two header rows
    if stats_table:
        data_rows = stats_table.find('tbody').find_all('tr', recursive=False)[2:]

        for row in data_rows:
            # The first cell contains the stat name and base value
            first_cell = row.find('td')
            if not first_cell:
                continue

            # --- Handle the special case for the "총합" (Total) row ---
            if "총합" in first_cell.get_text():
                nested_ths = first_cell.find_all('th')
                if len(nested_ths) > 1:
                    total_value = int(nested_ths[1].get_text(strip=True))
                    stats_data['총합'] = {'종족값': total_value}
                continue # Move to the next row

            # --- Handle the regular stat rows (HP, Attack, etc.) ---
            # Get all column cells for the current row
            columns = row.find_all(['td', 'th'], recursive=False)
            if len(columns) < 4:
                continue
                
            # Column 0: Nested table with stat name and base stat
            nested_ths = columns[0].find_all('th')
            stat_name = nested_ths[0].get_text(strip=True).replace(':', '')
            base_stat = int(nested_ths[1].get_text(strip=True))
            
            # Column 1: Lv. 50 Range
            lv50_range = columns[1].get_text(strip=True)
            
            # Column 2: Lv. 100 Range
            lv100_range = columns[2].get_text(strip=True)
            
            # Column 3: Effort Value (EV)
            effort_value = int(columns[3].get_text(strip=True))
            
            # Assemble the dictionary for the current stat
            stats_data[stat_name] = {
                "종족값": base_stat,
                "능력치 범위": {
                    "Lv. 50일 때": lv50_range,
                    "Lv. 100일 때": lv100_range,
                },
                "노력치": effort_value
            }

    return stats_data
