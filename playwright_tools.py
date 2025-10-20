import logging
from typing import Any, Dict, List, Optional
from playwright.sync_api import (
    sync_playwright,
    Playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEFAULT_TIMEOUT = 5000


class PlaywrightController:
    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        self.headless = headless
        self.browser_type = browser_type
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._dom_cache: Dict[str, List[Dict[str, Any]]] = {}

    def launch(self) -> None:
        logging.info(f"Launching browser ({self.browser_type}, headless={self.headless})")
        self.playwright = sync_playwright().start()
        browser_launcher = getattr(self.playwright, self.browser_type, None)
        if browser_launcher is None:
            raise RuntimeError(f"Unsupported browser type: {self.browser_type}")
        self.browser = browser_launcher.launch(headless=self.headless)
        logging.info("Browser launched successfully.")

    def new_context(self, storage_state: Optional[str] = None) -> None:
        if not self.browser:
            raise RuntimeError("Browser not launched. Call launch() first.")
        if storage_state:
            logging.info(f"Creating new browser context with storage state: {storage_state}")
            self.context = self.browser.new_context(storage_state=storage_state)
        else:
            logging.info("Creating new browser context (no storage state).")
            self.context = self.browser.new_context()
        self.page = self.context.new_page()

    def open_page(self, url: str, wait_until: str = "load") -> None:
        if not self.page:
            raise RuntimeError("No page available. Call new_context() first.")
        logging.info(f"Opening page: {url}")
        self.page.goto(url, wait_until=wait_until)
        logging.info(f"Page loaded: {self.page.url}")

    def describe_page(self) -> List[Dict[str, Any]]:
        return self.get_page_elements()

    def get_page_elements(self) -> List[Dict[str, Any]]:
        """
        Extract a compact summary of actionable elements on the current page.
        Filters by inputs, buttons, links, selects, textareas and elements that have data-testid or id.
        """
        if not self.page:
            return []

        url = self.page.url or ""
        if url in self._dom_cache:
            logging.debug(f"Using cached DOM for {url}")
            return self._dom_cache[url]

        logging.info("Scraping page elements for LLM context.")
        query = "[data-testid], input, button, a, select, textarea"
        try:
            elements = self.page.query_selector_all(query)
        except Exception as e:
            logging.warning(f"Failed to query elements: {e}")
            return []

        summary: List[Dict[str, Any]] = []
        for el in elements:
            try:
                tag = el.evaluate("e => e.tagName.toLowerCase()")
                text = el.inner_text().strip()[:200]
                id_ = el.get_attribute("id")
                data_testid = el.get_attribute("data-testid")
                name = el.get_attribute("name")
                placeholder = el.get_attribute("placeholder")
                aria = el.get_attribute("aria-label")
                xpath = el.evaluate(
                    "e => { let idx=0; let elCur = e; while(elCur.previousElementSibling){ idx++; elCur = elCur.previousElementSibling; } return e.tagName.toLowerCase() + '[' + (idx+1) + ']'; }"
                )
                visible = el.is_visible()
                summary.append(
                    {
                        "tag": tag,
                        "text": text,
                        "id": id_,
                        "data-testid": data_testid,
                        "name": name,
                        "placeholder": placeholder,
                        "aria-label": aria,
                        "xpath_hint": xpath,
                        "visible": visible,
                    }
                )
            except Exception:
                continue

        self._dom_cache[url] = summary
        logging.info(f"Collected {len(summary)} elements from {url}")
        return summary

    def fill(self, selector: str, text: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        if not self.page:
            raise RuntimeError("No page available. Call new_context() first.")
        logging.info(f"Filling '{selector}' with '{text}'")
        try:
            locator = self.page.locator(selector)
            locator.wait_for(state="visible", timeout=timeout)
            locator.fill(text)
        except Exception as e:
            logging.warning(f"Failed to fill {selector}: {e}")

    def click(self, selector: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        if not self.page:
            raise RuntimeError("No page available. Call new_context() first.")
        logging.info(f"Clicking element '{selector}'")
        try:
            locator = self.page.locator(selector)
            locator.wait_for(state="visible", timeout=timeout)
            locator.click()
        except Exception as e:
            logging.warning(f"Click failed for {selector}: {e}")

    def assert_text(self, selector: str, expected: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
        if not self.page:
            return False
        logging.info(f"Asserting that '{selector}' contains text '{expected}'")
        locator = self.page.locator(selector)
        try:
            locator.wait_for(state="visible", timeout=timeout)
            actual = locator.inner_text()
            result = expected.strip() in (actual or "")
            if not result:
                logging.warning(f"Assertion failed: expected '{expected}', got '{actual}'")
            return result
        except TimeoutError:
            logging.warning(f"Timeout waiting for selector {selector}")
            return False
        except Exception as e:
            logging.warning(f"Assertion error: {e}")
            return False

    def wait_for(self, selector: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        if not self.page:
            raise RuntimeError("No page available. Call new_context() first.")
        logging.info(f"Waiting for selector '{selector}' to appear.")
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
        except Exception as e:
            logging.warning(f"Wait failed for {selector}: {e}")

    def save_screenshot(self, path: str) -> None:
        if self.page:
            try:
                self.page.screenshot(path=path)
                logging.info(f"Screenshot saved: {path}")
            except Exception as e:
                logging.warning(f"Failed to save screenshot: {e}")

    def save_storage_state(self, path: str) -> None:
        if self.context:
            try:
                self.context.storage_state(path=path)
                logging.info(f"Storage state saved: {path}")
            except Exception as e:
                logging.warning(f"Failed to save storage state: {e}")

    def pause_for_manual_login(self) -> None:
        logging.info("Manual login pause. Perform login in browser, then press Enter to continue...")
        input()

    def close(self) -> None:
        logging.info("Closing Playwright session.")
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            logging.info("Playwright session closed cleanly.")
        except Exception as e:
            logging.warning(f"Error during browser close: {e}")