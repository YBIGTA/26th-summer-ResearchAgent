import os
import json
import time
import sys
import logging
from typing import List, Dict

from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import requests
import urllib.parse

from web_crawler import get_driver, fetch_url, scrape_pokemon_data

# --- Configuration ---
NATIONAL_DEX_URL = "https://pokemon.fandom.com/ko/wiki/전국도감"
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = "pokedex_crawler.log"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s/%(name)s/%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("crawler.log", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # --- Setup ---
    os.makedirs(os.path.join(DIRECTORY, 'json'), exist_ok=True)

    driver = get_driver()
    
    logger.info("Fetching all Pokemon links from the National Pokedex...")
    all_pokemon_links = []
    try:
        soup = fetch_url(driver, NATIONAL_DEX_URL)
        
        pokemon_tables = soup.find_all("table")

        required_styles = [
            "border-radius: 10px"
        ]
        
        for table in pokemon_tables:
            style = table.get('style')
            if style:
                # Check if all the required style strings are in the style attribute
                if not all(s in style for s in required_styles):
                    continue

            rows = table.find("tbody").find_all("tr")
            for row in rows[1:]: # Skip header row
                cells = row.find_all("td")
                if len(cells) > 1:
                    link_tag = cells[3].find("a")
                    if link_tag and link_tag.has_attr("href"):
                        name = link_tag.get_text(strip=True)
                        # Create a valid filename by removing forbidden characters
                        url = "https://pokemon.fandom.com" + link_tag['href']
                        file_name = urllib.parse.unquote(url.split('/')[-1]) + '.json'
                        all_pokemon_links.append({"name": name, "file_name": file_name, "url": url})
                        
        logger.info(f"Found {len(all_pokemon_links)} Pokemon links.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Pokedex page: {e}")

    # --- Handle "Continue From" Feature ---
    if "--continue_from" in sys.argv:
        try:
            pokemon_name_to_find = sys.argv[sys.argv.index("--continue_from") + 1]
            start_index = next((i for i, link in enumerate(all_pokemon_links) if link['name'] == pokemon_name_to_find), -1)
            
            if start_index != -1:
                logger.info(f"Resuming scrape from Pokemon: {pokemon_name_to_find}")
                links_to_process = all_pokemon_links[start_index:]
            else:
                logger.warning(f"Could not find Pokemon '{pokemon_name_to_find}'. Starting from the beginning.")
        except IndexError:
            logger.error("--continue_from requires a Pokemon name. Starting from the beginning.")

    # --- Main Scraping Loop ---
    pokemon_url = None

    try:
        total_links = len(all_pokemon_links)
        for i, link_info in enumerate(all_pokemon_links):
            pokemon_name = link_info['name']
            pokemon_url = link_info['url']
            
            logger.info(f"[{i+1}/{total_links}] Processing: {pokemon_name}")

            # Scrape all data for the Pokémon using the imported function
            pokemon_data = scrape_pokemon_data(driver, pokemon_url)
            
            if pokemon_data:
                # Save the data to an individual file
                file_name = link_info['file_name']
                file_path = os.path.join(DIRECTORY, 'json', file_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(pokemon_data, f, ensure_ascii=False, indent=4)
                
                logger.info(f"Successfully saved data to {file_path}")
            else:
                logger.error(f"Failed to scrape data for {pokemon_name}.")
            
            time.sleep(1) # politeness delay

        logger.info("Job complete!")
        driver.quit()
    except Exception as e:
        logger.error(f'Error fetching URL={pokemon_url}: {e}')
        