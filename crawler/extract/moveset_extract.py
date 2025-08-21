import re
import os

from bs4 import BeautifulSoup
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

import logging

logger = logging.getLogger(__name__)
DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_moveset(soup: BeautifulSoup, driver: WebDriver):
    """
    Finds all generation tabs and extracts level-up and TM moves for each.
    """
    data = {}

    # --- 1. Find the moveset container
    # Find header with table
    moveset_header = soup.find("span", id="배우는_기술")
    tab_links = []

    if not moveset_header:
        return data
    
    try:
        moveset_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-tabview-id]"))
        )

        tab_links = moveset_container.find_elements(By.CSS_SELECTOR, "ul.tabs > li > a")
        gen_names = [link.text.strip() for link in tab_links if link.text.strip()]

        # The `data-tab` attribute in the parent `li` helps us target the content
        parent_lis = moveset_container.find_elements(By.CSS_SELECTOR, "ul.tabs > li")
        tab_ids = [li.get_attribute("data-tab") for li in parent_lis if li.get_attribute("data-tab")]

        if not gen_names or not tab_ids:
            logger.warning("No generation tabs were found.")
            return {}

        for i, gen_name in enumerate(gen_names):
            logger.info(f"Processing moves for: {gen_name}")

            current_tab_links = driver.find_elements(By.CSS_SELECTOR, "div[data-tabview-id] ul.tabs > li > a")
            if i >= len(current_tab_links):
                logger.error(f"Could not re-find tab for {gen_name}.")
                continue
            
            tab_to_click = current_tab_links[i]

            # --- 2. Click each tab to load its content via JavaScript
            driver.execute_script("arguments[0].click();", tab_to_click)

            tab_body_id = tab_ids[i]
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, f"div.tabBody[data-tab-body='{tab_body_id}'].selected div.mw-parser-output")
                )
            )

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Scope the search to only the active tab's content for accuracy
            active_tab_content = soup.select_one(f"div.tabBody[data-tab-body='{tab_body_id}'].selected")
            if not active_tab_content:
                logger.warning(f"Could not find content for active tab: {gen_name}")
                continue

            gen_moves = {}

            # --- 3. Find and parse tables within the active tab's content
            # Find and parse "Level Up" moves
            levelup_header = active_tab_content.find("span", id="레벨업으로_배우는_기술")
            if levelup_header:
                move_table = levelup_header.find_parent(['h3', 'h4']).find_next_sibling('table')
                gen_moves['레벨업으로 배우는 기술'] = parse_moves_table(move_table)

            # Find and parse "TM/HM" moves
            tm_header = active_tab_content.find('span', id='기술/비전머신으로_배우는_기술')
            if tm_header:
                move_table = tm_header.find_parent(['h3', 'h4']).find_next_sibling('table')
                gen_moves['기술머신으로 배우는 기술'] = parse_moves_table(move_table)

            # Find and parse "Taught" moves
            tm_header = active_tab_content.find('span', id='가르침기술로_배우는_기술')
            if tm_header:
                move_table = tm_header.find_parent(['h3', 'h4']).find_next_sibling('table')
                gen_moves['가르침기술로 배우는 기술'] = parse_moves_table(move_table)

            # Find and parse "Hatched" moves
            tm_header = active_tab_content.find('span', id='교배로_배우는_기술')
            if tm_header:
                move_table = tm_header.find_parent(['h3', 'h4']).find_next_sibling('table')
                gen_moves['교배로 배우는 기술'] = parse_moves_table(move_table)
        
            ## Skip 이벤트로 배우는 기술

            data[gen_name] = gen_moves
        
    except (NoSuchElementException, TimeoutException):
        logger.warning("Could not find or interact with the moveset tabs on the page.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during moveset extraction: {e}", exc_info=True)

    logger.info("All moveset Scraping Complete")
    return data

def parse_moves_table(table_element):
    """
    A universal and robust parser for all known Pokémon move table layouts.
    It handles different header locations, nested tables, and special columns like '父' (Parent).
    """
    if not table_element:
        return []

    header_row = None
    th_tags = []
    header_keys = []
    data_start_index = 0

    # Find the innermost table that contains the actual data rows
    inner_table = table_element.find("table", class_=re.compile("sortable"))

    if inner_table == None:
        # Find all potential rows from either tbody or the table itself
        all_rows = table_element.find("tbody").find_all("tr", recursive=False)

        if not all_rows:
            return []

        # Identify the header row
        for i, row in enumerate(all_rows):
            th_tags = row.find_all("th", recursive=False)
            row_text = row.get_text()

            if len(th_tags) > 2 and "기술" in row_text and "타입" in row_text:
                header_row = row

                data_start_index = i + 1
                break # stop once we've found our header
    else:
        # Edge case for inner table with jquery sortable
        all_rows = inner_table.find("tbody").find_all("tr", recursive=False)

        if not all_rows:
            return []
        header = inner_table.find("thead")

        if header:
            header_row = header.find("tr")
        else:
            header_row = all_rows[0]
            data_start_index = 1

        th_tags = header_row.find_all("th", recursive=False)

    header_keys = []
    for h in th_tags:
        header_text = h.get_text(strip=True)
        span_count = int(h.get('colspan', 1))
        header_keys.extend([header_text] * span_count) # add based on colspan count

    moves_list = []
    # Process rows that match number of cells as the header row
    for row in all_rows[data_start_index:]:
        cells = row.find_all(["td", "th"], recursive=False)

        if len(cells) == len(header_keys):
            move_data = {}
            for key, cell in zip(header_keys, cells):
                # Special handling for colspan header keys
                if key in move_data:
                    # Special handling within special handling for "게임" fields
                    if key == "게임":
                        styles = cell.get("style").lower().split()

                        if not 'background:#FFF;' in styles and not 'white' in styles:
                            move_data[key] += f", {cell.get_text(strip=True)}" # only append if colored
                    else:
                        move_data[key] += f", {cell.get_text(strip=True)}" # append normally
                elif key == "게임":
                    styles = cell.get("style").lower().split()

                    if not 'background:#FFF;' in styles and not 'white' in styles:
                        move_data[key] = cell.get_text(strip=True) # only set if colored
                # Special handling for the '父' (Parent) column in egg move tables
                elif key == "父" or key == "부모":
                    parent_names = [a.get("title") for a in cell.find_all("a", title=True)]
                    move_data["부모"] = parent_names
                # Handle level integer data
                elif key == "LV" or key == "레벨":
                    cell_data = cell.get_text(strip=True)

                    try:
                        if cell_data == "최초" or cell_data == "진화" or cell_data == "떠올리기":
                            move_data["레벨"] = cell_data
                        else:
                            move_data[key] = int(cell_data)
                    except Exception:
                        logger.warning(f"Error handling level integer data {cell_data}")
                        move_data["레벨"] = cell_data
                # Handle other integer data
                elif key in ["위력", "PP"]:
                    cell_data = cell.get_text(strip=True)

                    
                    try:
                        if '—' in cell_data or '-' in cell_data:
                            move_data[key] = '-'
                        else:
                            move_data[key] = int(cell_data)
                    except ValueError:
                        # Try removing non integer chars
                        logger.warning(f"ValueError detected for {key} {cell_data} pair")
                        try:
                            result = re.sub(r"[^0-9\.]", "", cell_data)

                            move_data[key] = int(result)
                        except Exception:
                            logger.warning(f"Number filter failed, fallback to normal string")
                            move_data[key] = cell_data
                else:
                    move_data[key] = cell.get_text(strip=True)
            moves_list.append(move_data)

    return moves_list