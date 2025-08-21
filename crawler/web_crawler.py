import os 
import json
import time
import sys
from typing import Tuple

from bs4 import BeautifulSoup as bs
from selenium import webdriver as wb
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import urllib.parse
from pathlib import Path
import requests
import logging

from extract.infobox_extract import extract_infobox
from extract.explanations_extract import extract_explanations
from extract.evolutions_extract import extract_evolutions
from extract.abilities_extract import extract_abilities
from extract.moveset_extract import extract_moveset


logger = logging.getLogger('crawler')
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s/%(name)s/%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("crawler.log", mode='w'),
        logging.StreamHandler()
    ]
)

def get_driver() -> WebDriver:
    options = Options()
    options.add_argument("--headless")
    options.page_load_strategy = 'eager'

    service = Service(os.path.join(DIRECTORY, "chromedriver.exe"))
    driver = wb.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)

    return driver
    
def save_soup(url: str, filename: str="bulbasaur.html") -> None:
    """
    Optional helper: save a snapshot of the page for later local testing.
    """
    res = requests.get(url)
    res.raise_for_status()

    folder_name = 'save'
    full_path = os.path.join(DIRECTORY, folder_name, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(res.text)
    logger.info(f"Saved page to {full_path}")

def fetch_url(driver: WebDriver, url: str) -> bs:
    driver.get(url)
        
    # Wait up to 20 seconds for the main content area to be loaded.
    # 'div.mw-parser-output' is a stable container for all article content on this wiki.
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.mw-parser-output"))
    )
    
    # A small, optional buffer can still be helpful for very slow scripts
    time.sleep(1) 
    
    return bs(driver.page_source, "html.parser")

def scrape_pokemon_data(driver: WebDriver, url: str):
    """
    Scrapes all data from a single Pokémon page URL.
    """
    try:
        soup = fetch_url(driver, url)
        
        data = {}
        data['infobox'] = extract_infobox(soup)
        data['explanations'] = extract_explanations(soup)
        data['evolutions'] = extract_evolutions(soup)
        data['abilities'] = extract_abilities(soup)
        data['moveset'] = extract_moveset(soup, driver)
        
        return data
        
    except TimeoutException:
        logger.error(f"  - Timed out waiting for core page content at {url}")
        return None
    except Exception as e:
        logger.error(f"  - An unexpected error occurred while scraping {url}: {e}")
        return None

def print_json(data: dict, tab_space = "") -> None:
    """
    Pretty print JSON data.
    """
    for key, value in data.items():
        print(f"{tab_space}\"{key}\":", end=" ")

        if isinstance(value, str):
            print(f"\"{value[:100]}\"{(value[100:] and '..')}", end=",\n")
        elif isinstance(value, list):
            if len(value) == 0:
                print("  [],")
                continue
            
            print(f"{tab_space}[")

            tab_space += "  "
            
            for i in range(len(value) - 1):
                item = value[i]
                if isinstance(item, dict):
                    print(f"{tab_space}{{")
                    print_json(item, tab_space=tab_space + "  ")
                    print(f"{tab_space}}}", end="")
                else:
                    print(f"{tab_space}{str(item)}", end="")  # Print with comma except last item

                print(f",") 

            item = value[-1]
            if isinstance(item, dict):
                print(f"{tab_space}{{")
                print_json(item, tab_space=tab_space + "  ")
                print(f"{tab_space}}}")
            else:
                print(f"{tab_space}{str(item)}")   

            tab_space = tab_space[:-2]
            
            print(f"{tab_space}],")      
        elif isinstance(value, dict):
            print(f"{tab_space}{{")
            print_json(value, tab_space=tab_space + "  ")
            print(f"{tab_space}}}")
        else: # number, etc
            print(f"{value}", end=",\n")

def print_usage() -> None:
    print("Usage: python web_crawler.py [url] [infobox|explanations|evolutions|abilities|moveset|all]")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    url = sys.argv[1]
    choice = sys.argv[2]

    driver = get_driver()
    
    if "-local" in sys.argv:
        # Convert file to file:// URI
        absolute_path = Path(url).resolve()
        url = absolute_path.as_uri()

    logger.info("Fetching URL=" + url)
    soup = fetch_url(driver, url)

    if not "-local" in sys.argv and "--save" in sys.argv:
        idx = sys.argv.index("--save")
        if idx + 1 < len(sys.argv):
            save_file = sys.argv[idx + 1]

            save_soup(url, save_file)
        else:
            logger.error("Error: --save requires a filename")
            sys.exit(1)

    if choice == "infobox":
        print_json(extract_infobox(soup))
    elif choice == "explanations":
        print_json(extract_explanations(soup))
    elif choice == "evolutions":
        data = {}
        data["evolutions"] = extract_evolutions(soup)

        print_json(data)
    elif choice == "abilities":
        print_json(extract_abilities(soup))
    elif choice == "moveset":
        print_json(extract_moveset(soup, driver))
    elif choice == "all":
        data = scrape_pokemon_data(driver, url)

        folder_name = 'json'
        
        if not "-local" in sys.argv:
            file_name = urllib.parse.unquote(url.split('/')[-1]) + '.json'
        else:
            file_name = urllib.parse.unquote(url.split('.')[0]) + '.json'
        full_path = os.path.join(DIRECTORY, folder_name, file_name)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved page to {full_path}")
    else:
        print_usage()
        exit(1)
    
    logger.info("Job complete!")
    driver.quit()

