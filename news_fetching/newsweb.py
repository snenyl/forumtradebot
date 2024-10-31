from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
from datetime import datetime

class JavaScriptContentFetcher:
    def __init__(self, main_url):
        self.main_url = main_url
        self.message_prefix_url = "https://newsweb.oslobors.no/"
        self.hrefs = []


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

    def fetch_newsweb_message_content(self, url):
        print("Starting Playwright")
        with sync_playwright() as p:
            print("Launching browser")
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            print(f"Navigating to {url}")
            page.goto(url)
            print("Waiting for JavaScript to load")
            # Wait until the network is idle to ensure all resources have loaded
            page.wait_for_load_state("networkidle")
            # Get page content
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text()
            browser.close()
        return text_content

    def fetch_newsweb_list_urls(self, main_url):
        print("Starting Playwright")
        with sync_playwright() as p:
            print("Launching browser")
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            print(f"Navigating to {main_url}")
            page.goto(main_url)

            # Infinite loop to handle pagination
            while True:
                # Wait until any table element is loaded
                print("Waiting for any table element to load")
                page.wait_for_selector("table")

                # Extract HTML content for parsing
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')

                # Find the first table element in the HTML
                table = soup.find("table")
                if table:
                    print("Found the table.")

                    # Find all 'a' tags within the table that have href containing "/message/"
                    message_links = table.find_all("a", href=lambda x: x and "/message/" in x)

                    # Extract href values and print them
                    hrefs = [link['href'] for link in message_links]
                    print(f"Extracted {len(hrefs)} hrefs:")
                    print(hrefs)

                    if len(hrefs) < 50:
                        break # Break out of loop when on last page (Not bulletproof but good enough)

                else:
                    print("Table not found!")

                # Attempt to locate and click the next page button
                try:
                    print("Looking for next page button...")
                    next_button = page.locator("a:has-text('⟩')")
                    if next_button.count() > 0:
                        print("Next page button found. Clicking...")
                        next_button.first.click()
                        page.wait_for_load_state("domcontentloaded")  # Wait for the new page to load
                    else:
                        print("Next page button not found. Ending pagination.")
                        break
                except Exception as e:
                    print(f"Error while trying to navigate to the next page: {e}")
                    break

            # Close the browser after pagination is complete
            browser.close()
            print("Browser closed.")
        None

# Standard entry point
if __name__ == "__main__":
    url = "https://newsweb.oslobors.no/message/506633"  # Replace with your target URL
    main_url = "https://newsweb.oslobors.no/search?category=&issuer=12711&fromDate=2014-10-01&toDate=2024-10-31&market=&messageTitle="  # Replace with your target URL

    fetcher = JavaScriptContentFetcher(main_url)

    fetcher.fetch_newsweb_list_urls(main_url)

    # content = fetcher.fetch_newsweb_message_content(url)
    # if content:
    #     page_datetime = fetcher.extract_datetime(content)
    #     print(page_datetime)
    #     print(content)
