from bs4 import BeautifulSoup


def extract_explanations(soup: BeautifulSoup) -> dict:
    """
    Extract explanations from the Pokémon page
    """
    explanation_tables = None
    data = {}

    # --- 1. Find header with table
    explanations_header = soup.find("span", id="포켓몬_도감_설명")

    if explanations_header:
        # 2. The main evolution table is the next sibling of the <h3> tag
        explanation_tables = explanations_header.find_parent("h3").find_next_sibling("table").find_all("table")

    # --- 2. Extract data from table
    if explanation_tables:
        # Each generation is an outer table with a <th><small> like "1세대"
        for gen_table in explanation_tables:
            gen_name_tag = gen_table.select_one("th small")
            if not gen_name_tag:
                continue
            gen_name = gen_name_tag.get_text(strip=True)
            data[gen_name] = {}

            # Inside each generation, find nested inner tables (with game names + explanations)
            inner_tables = gen_table.select("td[colspan] table.roundy")
            for inner in inner_tables:
                rows = inner.select("tr")

                for row in rows:
                    ths = row.select("th.roundy")
                    tds = row.select("td.roundy")

                    # If explanation exists in this row
                    if tds:
                        explanation = tds[0].get_text(strip=True)
                        for th in ths:
                            game_name = th.get_text(strip=True)
                            data[gen_name][game_name] = explanation
                    else:
                        # Handle rows where explanation is on another row (rowspan)
                        for th in ths:
                            game_name = th.get_text(strip=True)
                            # Explanation might be in previous <td> with rowspan
                            # Find sibling <td> if available
                            td = row.find("td", class_="roundy")
                            if td:
                                explanation = td.get_text(strip=True)
                                data[gen_name][game_name] = explanation

    return data