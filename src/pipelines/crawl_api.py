#!/usr/bin/env python3
"""
Crawl Windows API documentation from learn.microsoft.com to build a dataset
of API descriptions for DawnGNN.

Website layout changes frequently, so this script uses multiple CSS selectors,
retry logic, and optional non-headless mode to remain robust.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

try:
    from dawngnn.utils.paths import get_project_root
except ImportError:
    import sys
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from utils.paths import get_project_root

PROJECT_ROOT = get_project_root()
BASE_URL = "https://learn.microsoft.com/en-us/windows/win32/api/"

TOC_SELECTORS = [
    "ul.tree.table-of-contents",
    "nav[aria-label='Table of contents'] ul",
    "div[class*='table-of-contents'] ul",
    "section[aria-label='Table of contents'] ul",
    "section[aria-label='Table of contents'] li",
]

NAV_LINK_SELECTORS = [
    "ul.margin-none.has-line-height-reset li a.has-external-link-indicator",
    "div[data-bi-name='toc'] li a",
    "nav[aria-label='Primary navigation'] li a",
    "nav[aria-label='Secondary navigation'] li a",
    "nav[data-bi-name='primaryNav'] li a",
    "div[class*='is-side-nav'] li a",
    "div[class*='sideRail'] li a",
]

COOKIE_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "button[data-event-property='accept']",
    "button[aria-label*='Accept']",
]


def wait_for_selector(driver, selectors, timeout=30):
    """Wait until any selector in `selectors` appears."""
    last_err = None
    for selector in selectors:
        try:
            return WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except TimeoutException as err:
            last_err = err
            continue
    raise last_err or TimeoutException("Could not find matching selector on page.")


def setup_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(90)
    return driver


def click_cookie_banner(driver):
    for selector in COOKIE_SELECTORS:
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            btn.click()
            time.sleep(1)
            return True
        except TimeoutException:
            continue
    return False


def open_with_retry(driver, url: str, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(1.5)
            return True
        except (TimeoutException, WebDriverException):
            if attempt == retries - 1:
                return False
            driver.execute_script("window.stop();")
            time.sleep(2)
    return False


def collect_primary_links(driver) -> list:
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for selector in NAV_LINK_SELECTORS:
        for a in soup.select(selector):
            href = a.get("href")
            if not href:
                continue
            full_url = urljoin(BASE_URL, href)
            links.append(full_url)
    unique = []
    seen = set()
    for url in links:
        if not url.startswith(BASE_URL):
            continue
        if "/api/" not in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def extract_headers_from_toc(soup: BeautifulSoup) -> set:
    """Extract Win32 header file names (.h) from table-of-contents elements."""
    headers = set()
    for selector in TOC_SELECTORS:
        for node in soup.select(selector):
            # When selector returns <li>, node can be <li> instead of <ul>
            if node.name == "li":
                text = node.get_text(strip=True).replace("\u200b", "")
                if text.endswith(".h"):
                    headers.add(text[:-2])
                continue
            for li in node.select("li"):
                text = li.get_text(strip=True).replace("\u200b", "")
                if text.endswith(".h"):
                    headers.add(text[:-2])
    return headers


def extract_headers_from_inline_lists(soup: BeautifulSoup) -> set:
    """
    Fallback extraction when no explicit TOC is present.
    Many pages list required headers using <code>filename.h</code>.
    """
    headers = set()
    for code_tag in soup.find_all("code"):
        text = code_tag.get_text(strip=True)
        if text.endswith(".h") and "/" not in text:
            headers.add(text[:-2])
    # Also check strong/emphasis text
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True).replace("\u200b", "")
        if text.endswith(".h"):
            candidate = text.split()[-1]
            if candidate.endswith(".h") and "/" not in candidate:
                headers.add(candidate[:-2])
    return headers


def extract_functions(driver, url):
    """Return dict {function_name: description} for a given Win32 header page."""
    print(f"\n[=] Extracting functions from: {url}")
    if not open_with_retry(driver, url):
        print("   -> Failed to load page, skipping.")
        return {}

    soup = BeautifulSoup(driver.page_source, "html.parser")
    tables = soup.find_all("table", attrs={"aria-label": "Functions"})
    print(f"   -> Found {len(tables)} function tables")

    data = {}
    for table in tables:
        for td in table.find_all("td"):
            a = td.find("a", href=True)
            if not a:
                continue
            name = a.get_text(strip=True)
            full_text = td.get_text(" ", strip=True)
            desc = full_text.replace(name, "", 1).strip()
            if name and desc:
                data[name] = desc
    return data


def crawl(args):
    """Execute the multi-stage documentation crawler."""
    driver = setup_driver(headless=not args.no_headless)
    try:
        if not open_with_retry(driver, BASE_URL):
            raise RuntimeError("Failed to open root documentation page.")

        click_cookie_banner(driver)
        wait_for_selector(driver, NAV_LINK_SELECTORS, timeout=60)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        primary_links = collect_primary_links(driver)
        print(f"Found {len(primary_links)} primary category links.")
        if args.max_primary > 0:
            primary_links = primary_links[:args.max_primary]

        final_urls = set()
        for link in primary_links:
            print(f"\nProcessing category: {link}")
            if not open_with_retry(driver, link):
                print("  -> Failed to load page, skipping.")
                continue
            try:
                wait_for_selector(driver, TOC_SELECTORS, timeout=50)
            except TimeoutException:
                print("  -> Table of contents not found, skipping.")
                continue

            sub_soup = BeautifulSoup(driver.page_source, "html.parser")
            headers = extract_headers_from_toc(sub_soup)
            if not headers:
                headers = extract_headers_from_inline_lists(sub_soup)
            if not headers:
                print("  -> No headers extracted (page may require login or has non-standard layout).")
                continue

            for header in sorted(headers):
                url = f"https://learn.microsoft.com/en-us/windows/win32/api/{header}/"
                final_urls.add(url)
                print("   ->", url)

        print("\n=== FINAL DISCOVERED URLS ===")
        for u in sorted(final_urls):
            print(u)
        print("\nTotal headers:", len(final_urls))

        all_functions = {}
        for url in sorted(final_urls):
            funcs = extract_functions(driver, url)
            all_functions.update(funcs)

        output_path = Path(args.output) if args.output else PROJECT_ROOT / "data/api_docs/api_docs.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_functions, f, indent=4, ensure_ascii=False)
        print("\n============================")
        print(f"SUCCESS! Saved {len(all_functions)} functions to {output_path}")
    finally:
        driver.quit()


def main():
    parser = argparse.ArgumentParser(description="Crawl Windows API documentation from Microsoft Learn.")
    parser.add_argument(
        "--no-headless", action="store_true",
        help="Display browser GUI (useful for debugging)."
    )
    parser.add_argument(
        "--max-primary", type=int, default=0,
        help="Limit number of primary category links to crawl (0 = all)."
    )
    parser.add_argument(
        "--output", type=str, default=str(PROJECT_ROOT / "data/api_docs/api_docs.json"),
        help="Path to save the output API documentation JSON file."
    )
    args = parser.parse_args()
    crawl(args)


if __name__ == "__main__":
    main()