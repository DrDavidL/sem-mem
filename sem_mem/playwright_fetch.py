"""
Playwright-based web fetching for sem-mem.

Provides robust content extraction using headless browser automation:
- JavaScript rendering for dynamic content
- Better handling of modern SPAs
- Parallel fetching of multiple URLs
- Automatic retry with fallback

Usage:
    from sem_mem.playwright_fetch import PlaywrightFetcher, fetch_urls_parallel

    # Single fetch
    fetcher = PlaywrightFetcher()
    result = await fetcher.fetch("https://example.com")

    # Parallel fetch (from search results)
    results = await fetch_urls_parallel(["url1", "url2", "url3"])
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from urllib.parse import urlparse

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

# Playwright is optional - graceful degradation to requests-based fetcher
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
    PlaywrightTimeout = TimeoutError


@dataclass
class PlaywrightFetchResult:
    """Result from fetching a URL with Playwright."""
    url: str
    title: str
    content: str
    content_type: str
    success: bool
    error: Optional[str] = None
    status_code: Optional[int] = None
    fetch_time_ms: Optional[float] = None


class PlaywrightFetcher:
    """
    Fetch and extract content from URLs using Playwright.

    Features:
    - Headless browser for JavaScript rendering
    - Parallel fetching with configurable concurrency
    - Smart content extraction (main content detection)
    - Automatic timeout and error handling

    Security:
    - Validates URL scheme (http/https only)
    - Configurable domain allowlist/blocklist
    - Request timeout
    - Content size limits
    """

    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    DEFAULT_BLOCKED_DOMAINS = {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "internal",
        "intranet",
    }

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        max_content_length: int = 500_000,
        timeout_ms: int = 30_000,
        max_concurrent: int = 5,
        headless: bool = True,
    ):
        """
        Initialize Playwright fetcher.

        Args:
            allowed_domains: If set, only these domains are allowed (whitelist mode)
            blocked_domains: Domains to block (added to defaults)
            max_content_length: Maximum content size in characters
            timeout_ms: Page load timeout in milliseconds
            max_concurrent: Maximum concurrent page loads
            headless: Run browser in headless mode
        """
        self.allowed_domains = set(allowed_domains) if allowed_domains else None
        self.blocked_domains = self.DEFAULT_BLOCKED_DOMAINS.copy()
        if blocked_domains:
            self.blocked_domains.update(blocked_domains)
        self.max_content_length = max_content_length
        self.timeout_ms = timeout_ms
        self.max_concurrent = max_concurrent
        self.headless = headless

        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate URL for safety."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False, "Invalid URL format"

        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid scheme: {parsed.scheme}. Only http/https allowed."

        domain = parsed.netloc.lower()
        if not domain:
            return False, "No domain specified"

        domain_no_port = domain.split(":")[0]

        if domain_no_port in self.blocked_domains:
            return False, f"Domain blocked: {domain_no_port}"

        if re.match(r"^(10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)", domain_no_port):
            return False, "Private IP addresses not allowed"

        if self.allowed_domains is not None:
            if domain_no_port not in self.allowed_domains:
                return False, f"Domain not in allowlist: {domain_no_port}"

        return True, ""

    def _extract_text_from_html(self, html: str, url: str) -> tuple[str, str]:
        """Extract main text content from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                         "form", "button", "iframe", "noscript", "svg", "path"]):
            tag.decompose()

        main_content = None
        for selector in ["main", "article", '[role="main"]', ".content",
                        "#content", ".post-content", ".entry-content",
                        ".article-content", ".page-content"]:
            found = soup.select_one(selector)
            if found:
                main_content = found
                break

        if main_content is None:
            main_content = soup.body or soup

        text = main_content.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        return title, text

    async def _ensure_browser(self) -> Browser:
        """Ensure browser is started and return it."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not installed. Run: pip install playwright && playwright install chromium")

        if self._browser is None or not self._browser.is_connected():
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ]
            )
        return self._browser

    async def _get_context(self) -> BrowserContext:
        """Get or create browser context."""
        browser = await self._ensure_browser()
        if self._context is None:
            self._context = await browser.new_context(
                user_agent=self.USER_AGENT,
                viewport={"width": 1920, "height": 1080},
                java_script_enabled=True,
            )
        return self._context

    async def fetch(self, url: str) -> PlaywrightFetchResult:
        """
        Fetch and extract content from a URL using Playwright.

        Args:
            url: URL to fetch

        Returns:
            PlaywrightFetchResult with content or error
        """
        import time
        start_time = time.time()

        is_valid, error = self._validate_url(url)
        if not is_valid:
            return PlaywrightFetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                success=False,
                error=error,
            )

        if not PLAYWRIGHT_AVAILABLE:
            return PlaywrightFetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                success=False,
                error="Playwright not installed",
            )

        page: Optional[Page] = None
        try:
            context = await self._get_context()
            page = await context.new_page()

            response = await page.goto(
                url,
                timeout=self.timeout_ms,
                wait_until="domcontentloaded",
            )

            if response is None:
                return PlaywrightFetchResult(
                    url=url,
                    title="",
                    content="",
                    content_type="",
                    success=False,
                    error="No response received",
                )

            status_code = response.status
            if status_code >= 400:
                return PlaywrightFetchResult(
                    url=url,
                    title="",
                    content="",
                    content_type="",
                    success=False,
                    error=f"HTTP {status_code}",
                    status_code=status_code,
                )

            # Wait a bit for JS to render
            await asyncio.sleep(0.5)

            # Try to wait for main content to appear
            try:
                await page.wait_for_selector("main, article, .content, #content", timeout=2000)
            except PlaywrightTimeout:
                pass  # Content selectors not found, continue anyway

            html = await page.content()
            title, text = self._extract_text_from_html(html, url)

            # Apply content length limit
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + f"\n\n[Truncated - {len(text)} total chars]"

            fetch_time = (time.time() - start_time) * 1000

            return PlaywrightFetchResult(
                url=url,
                title=title or url,
                content=text,
                content_type="html",
                success=True,
                status_code=status_code,
                fetch_time_ms=fetch_time,
            )

        except PlaywrightTimeout:
            return PlaywrightFetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                success=False,
                error=f"Page load timed out after {self.timeout_ms}ms",
            )
        except Exception as e:
            return PlaywrightFetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                success=False,
                error=str(e),
            )
        finally:
            if page:
                await page.close()

    async def fetch_many(self, urls: List[str]) -> List[PlaywrightFetchResult]:
        """
        Fetch multiple URLs in parallel with controlled concurrency.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of PlaywrightFetchResult in same order as input URLs
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_with_semaphore(url: str) -> PlaywrightFetchResult:
            async with semaphore:
                return await self.fetch(url)

        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(PlaywrightFetchResult(
                    url=urls[i],
                    title="",
                    content="",
                    content_type="",
                    success=False,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def close(self):
        """Close browser and clean up resources."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience functions for common workflows

async def fetch_urls_parallel(
    urls: List[str],
    max_concurrent: int = 5,
    timeout_ms: int = 30_000,
    max_content_length: int = 500_000,
) -> List[PlaywrightFetchResult]:
    """
    Fetch multiple URLs in parallel using Playwright.

    This is the main entry point for batch fetching URLs from search results.

    Args:
        urls: List of URLs to fetch
        max_concurrent: Maximum concurrent fetches
        timeout_ms: Timeout per page in milliseconds
        max_content_length: Max content length per page

    Returns:
        List of PlaywrightFetchResult in same order as input

    Example:
        # After getting search results
        search_results = exa_search.search("Python asyncio tutorial")
        urls = [r.link for r in search_results.results]
        contents = await fetch_urls_parallel(urls)
    """
    async with PlaywrightFetcher(
        max_concurrent=max_concurrent,
        timeout_ms=timeout_ms,
        max_content_length=max_content_length,
    ) as fetcher:
        return await fetcher.fetch_many(urls)


def format_playwright_result_for_context(result: PlaywrightFetchResult, max_chars: int = 5000) -> str:
    """Format a Playwright fetch result as context for the LLM."""
    if not result.success:
        return f"Failed to fetch {result.url}: {result.error}"

    content = result.content[:max_chars]
    if len(result.content) > max_chars:
        content += f"\n\n[Content truncated - {len(result.content)} total chars]"

    return f"""Content from {result.url}:
Title: {result.title}

{content}"""


def format_multiple_results_for_context(
    results: List[PlaywrightFetchResult],
    max_chars_per_result: int = 3000,
    max_total_chars: int = 15000,
) -> str:
    """
    Format multiple fetch results as context for the LLM.

    Args:
        results: List of PlaywrightFetchResult
        max_chars_per_result: Max characters per individual result
        max_total_chars: Max total characters across all results

    Returns:
        Formatted string with all successful results
    """
    formatted_parts = []
    total_chars = 0

    for result in results:
        if not result.success:
            formatted_parts.append(f"[Failed: {result.url} - {result.error}]")
            continue

        # Calculate how much space we have left
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            formatted_parts.append(f"[Skipped remaining results due to length limit]")
            break

        chars_for_this = min(max_chars_per_result, remaining)
        content = result.content[:chars_for_this]
        if len(result.content) > chars_for_this:
            content += f"\n[Truncated]"

        part = f"""--- {result.title} ---
URL: {result.url}

{content}"""
        formatted_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(formatted_parts)


# Synchronous wrapper for non-async code

def fetch_url_sync(url: str, timeout_ms: int = 30_000) -> PlaywrightFetchResult:
    """
    Synchronous wrapper for fetching a single URL.

    Use this in synchronous code that can't use async/await.
    """
    return asyncio.run(_fetch_url_sync_impl(url, timeout_ms))


async def _fetch_url_sync_impl(url: str, timeout_ms: int) -> PlaywrightFetchResult:
    async with PlaywrightFetcher(timeout_ms=timeout_ms) as fetcher:
        return await fetcher.fetch(url)


def fetch_urls_sync(urls: List[str], max_concurrent: int = 5, timeout_ms: int = 30_000) -> List[PlaywrightFetchResult]:
    """
    Synchronous wrapper for fetching multiple URLs in parallel.

    Use this in synchronous code that can't use async/await.
    """
    return asyncio.run(fetch_urls_parallel(urls, max_concurrent=max_concurrent, timeout_ms=timeout_ms))
