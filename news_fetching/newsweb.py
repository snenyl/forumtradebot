from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
from datetime import datetime

class JavaScriptContentFetcher:
    def __init__(self, url):
        self.url = url

    def extract_datetime(self, text_content):
        # Updated regex pattern for date and time with more flexibility
        date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', text_content)
        time_match = re.search(r'(\d{2}:\d{2}:\d{2})', text_content)

        if date_match:
            date_str = date_match.group(1)
            print("Date extracted:", date_str)  # Debugging output
        else:
            print("No date found")  # Debugging output
            return None  # If no date found, return None

        if time_match:
            time_str = time_match.group(1)
            print("Time extracted:", time_str)  # Debugging output
        else:
            print("No time found")  # Debugging output
            return None  # If no time found, return None

        # Combine date and time strings and convert to a datetime object
        datetime_str = f"{date_str} {time_str}"
        page_datetime = datetime.strptime(datetime_str, "%d.%m.%Y %H:%M:%S")
        return page_datetime

    def fetch_content(self):
        print("Starting Playwright")
        with sync_playwright() as p:
            print("Launching browser")
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            print(f"Navigating to {self.url}")
            page.goto(self.url)
            print("Waiting for JavaScript to load")
            # Wait until the network is idle to ensure all resources have loaded
            page.wait_for_load_state("networkidle")
            # Get page content
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text()
            browser.close()
        return text_content

# Standard entry point
if __name__ == "__main__":
    url = "https://newsweb.oslobors.no/message/506633"  # Replace with your target URL
    fetcher = JavaScriptContentFetcher(url)

    content = fetcher.fetch_content()
    if content:
        page_datetime = fetcher.extract_datetime(content)
        print(page_datetime)
        print(content)
