from playwright.sync_api import sync_playwright

def click_at_coordinates(x, y):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Launch browser in non-headless mode to see actions
        page = browser.new_page()
        
        # Go to a page, replace 'https://example.com' with the target URL
        page.goto("https://example.com")
        
        # Click at specified coordinates (x, y) on the page
        page.mouse.click(x, y)
        
        # Close the browser
        browser.close()

# Example coordinates
click_at_coordinates(100, 200)

