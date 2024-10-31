from pymongo import MongoClient
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class JavaScriptContentFetcher:
    def __init__(self, main_url):
        self.main_url = main_url
        self.message_prefix_url = "https://newsweb.oslobors.no"
        self.hrefs = []

        # MongoDB connection
        mongo_uri = "mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT"
        self.client = MongoClient(mongo_uri)
        self.db = self.client.news
        self.collection = self.db.newsweb_pho

    def extract_datetime(self, text_content):
        # Regex patterns for date and time
        date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', text_content)
        time_match = re.search(r'(\d{2}:\d{2}:\d{2})', text_content)

        if not date_match or not time_match:
            return None

        datetime_str = f"{date_match.group(1)} {time_match.group(1)}"
        return datetime.strptime(datetime_str, "%d.%m.%Y %H:%M:%S")

    def fetch_newsweb_message_content(self, url):
        print(f"Starting Playwright for message content at URL: {url}")
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            page.wait_for_load_state("networkidle")
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text()
            browser.close()
        return text_content

    def fetch_newsweb_list_urls(self, main_url):
        print("Starting Playwright for list of URLs")
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(main_url)

            while True:
                page.wait_for_selector("table")
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')
                table = soup.find("table")
                if table:
                    message_links = table.find_all("a", href=lambda x: x and "/message/" in x)
                    hrefs = [link['href'] for link in message_links]
                    self.hrefs.extend(hrefs)

                    if len(hrefs) < 50:
                        break

                try:
                    next_button = page.locator("a:has-text('⟩')")
                    if next_button.count() > 0:
                        next_button.first.click()
                        page.wait_for_load_state("domcontentloaded")
                    else:
                        break
                except Exception as e:
                    print(f"Error during pagination: {e}")
                    break

            print(f"Total announcements found: {len(self.hrefs)}")
            browser.close()

    def save_to_mongodb(self, url, page_datetime, content):
        document = {
            "time": page_datetime,
            "url": url,
            "content": content
        }
        self.collection.insert_one(document)
        print(f"Saved to MongoDB: {url}")

    def process_href(self, href):
        full_url = self.message_prefix_url + href
        content = self.fetch_newsweb_message_content(full_url)
        if content:
            page_datetime = self.extract_datetime(content)
            self.save_to_mongodb(full_url, page_datetime, content)


if __name__ == "__main__":
    main_url = "https://newsweb.oslobors.no/search?category=&issuer=6357&fromDate=2014-01-01&toDate=2024-10-31&market=&messageTitle="
    fetcher = JavaScriptContentFetcher(main_url)

    # Step 1: Fetch all URLs to process
    fetcher.fetch_newsweb_list_urls(main_url)

    # Step 2: Use ThreadPoolExecutor to process URLs in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetcher.process_href, fetcher.hrefs)
