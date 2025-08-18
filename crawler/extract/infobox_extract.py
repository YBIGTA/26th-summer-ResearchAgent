from bs4 import BeautifulSoup

from util import get_raw_data


def extract_infobox(soup: BeautifulSoup) -> dict:
    """
    Extracts infobox data from the Pokémon page
    """

    data = {}
    info_table = soup.select_one('table.body')
    if not info_table:
        return data

    # --- 1. Extract Main Image ---
    main_image_tag = soup.select_one('.infobox-pokemon .image img')
    if main_image_tag and main_image_tag.has_attr('data-src'):
        data['이미지'] = get_raw_data(main_image_tag['data-src'])

    # --- 2. Iterate through table rows for all other attributes ---
    rows = info_table.find_all('tr')
    for row in rows:
        headers = row.find_all('th')
        
        # If there are no headers in this row, it's likely a data row that
        # has already been processed by its preceding header row. Skip it.
        if not headers:
            continue
        
        header_texts = [h.get_text(strip=True) for h in headers]
        
        # --- SPECIAL HANDLING: POKEDEX NUMBERS ---
        # This section has a nested table that needs unique logic.
        if '보이기도감 번호' in header_texts:
            pokedex_entries = []

            target_tr = headers[0].find_parent('table').find_all('td')
            
            for cell in target_tr:
                text = cell.get_text()

                if(text[:3] == '칼로스'):
                    text = '칼로스#' + text[3:]
                pokedex_entries.append(text.replace('\n', ''))

            data['도감 번호'] = pokedex_entries
            continue # Done with this row, move to the next

        # --- GENERAL HANDLING FOR ALL OTHER ROWS ---
        data_row = row.find_next_sibling('tr')
        if not data_row:
            continue
        
        data_cells = data_row.find_all('td')
        if not data_cells:
            continue

        for i, header in enumerate(headers):
            key = header_texts[i]
            
            # ignore list
            if key == '외부 링크':
                continue

            if i < len(data_cells):
                value_cell = data_cells[i]

                # --- SPECIAL HANDLING: MEDIA EXTRACTION ---
                if key == '형태':
                    img_tag = value_cell.find('img')
                    if img_tag and img_tag.has_attr('data-src'):
                        data['형태'] = get_raw_data(img_tag['data-src'])
                elif key == '발자국':
                    img_tag = value_cell.find('img')
                    if img_tag and img_tag.has_attr('data-src'):
                        data['발자국'] = get_raw_data(img_tag['data-src'])
                elif key == '울음소리':
                    audio_tag = value_cell.find('audio')
                    if audio_tag and audio_tag.has_attr('src'):
                        data['울음소리'] = get_raw_data(audio_tag['src'])
                # --- DEFAULTS: TEXT ---
                else:
                    # Clean text to remove extra commas from newlines
                    raw_text = value_cell.get_text(strip=True, separator=',')
                    parts = [part.strip() for part in raw_text.split(',') if part.strip()]

                    if key == '성비':
                        gender_dict = {}
                        
                        # Loop through the list in steps of 2 (key, value)
                        if len(parts) % 2 == 0: # Ensure we have pairs
                            for j in range(0, len(parts), 2):
                                gender_key = parts[j].replace(':', '').strip()
                                gender_value = parts[j+1]
                                try:
                                    # Remove '%' and convert value to a float
                                    gender_dict[gender_key] = float(gender_value.replace('%', ''))
                                except ValueError:
                                    gender_dict[gender_key] = gender_value # Fallback
                        else:
                            # Handle cases like genderless Pokémon, e.g., ['없음']
                            gender_dict = ', '.join(parts) if parts else None

                        data[key] = gender_dict
                    elif key in ['LV.100 경험치량', '기초 친밀도', '포획률']:
                        try:
                            # 1. Remove commas and spaces from the string "1, 059, 860"
                            numeric_value = int(raw_text.replace(',', '').replace(' ', ''))
                            # 2. Convert the clean string "1059860" to the number 1059860
                            data[key] = numeric_value
                        except ValueError:
                            data[key] = raw_text # Fallback if conversion fails
                    elif key in ['포켓몬 교배']:
                        reproduction = {}

                        reproduction['알그룹'] = parts
                        reproduction['부화 걸음수'] = data_cells[i+1].get_text().replace('부화 걸음수', '').replace('\n', '')

                        data['포켓몬 교배'] = reproduction
                    else:                
                        if len(parts) == 1:
                            data[key] = parts[0]
                        else:            
                            data[key] = parts
    return data