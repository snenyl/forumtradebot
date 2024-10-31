from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

class JavaScriptContentFetcher:
    def __init__(self, url):
        self.url = url

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
        print(content)
