"""
Web search integration for sem-mem.

Supports multiple backends:
1. Exa (recommended) - AI-native search with structured results, great for real-time data
2. Tavily - AI-native search optimized for LLM applications
3. Google Programmable Search Engine (PSE) - requires API key + Engine ID
4. OpenAI web_search_preview tool - requires OpenAI API with Responses API support

Web fetching:
- Playwright-based fetching (default) - JavaScript rendering, parallel fetching
- Requests-based fallback - simpler, no browser dependency

Configuration via environment variables:
- EXA_API_KEY: Exa API key (get from https://exa.ai)
- TAVILY_API_KEY: Tavily API key (get from https://tavily.com)
- GOOGLE_PSE_API_KEY: Google Custom Search API key
- GOOGLE_PSE_ENGINE_ID: Programmable Search Engine ID (cx parameter)
- WEB_SEARCH_BACKEND: "exa", "tavily", "google_pse", or "openai" (default: auto-detect)
- WEB_FETCH_USE_PLAYWRIGHT: "true" to use Playwright (default), "false" for requests
"""

import asyncio
import os
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse
import json

import requests
from bs4 import BeautifulSoup

# Import Playwright fetcher (optional)
try:
    from .playwright_fetch import (
        PlaywrightFetcher,
        PlaywrightFetchResult,
        fetch_urls_parallel,
        fetch_urls_sync,
        format_playwright_result_for_context,
        format_multiple_results_for_context,
        PLAYWRIGHT_AVAILABLE,
    )
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightFetcher = None
    PlaywrightFetchResult = None
    fetch_urls_parallel = None
    fetch_urls_sync = None
    format_playwright_result_for_context = None
    format_multiple_results_for_context = None


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    link: str
    snippet: str


@dataclass
class SearchResponse:
    """Response from a web search."""
    results: List[SearchResult]
    query: str
    backend: str


class GooglePSESearch:
    """
    Google Programmable Search Engine integration.

    Requires:
    - GOOGLE_PSE_API_KEY: API key from Google Cloud Console
    - GOOGLE_PSE_ENGINE_ID: Search engine ID from PSE control panel

    Get these from:
    - API key: https://console.cloud.google.com/apis/credentials
    - Engine ID: https://programmablesearchengine.google.com/
    """

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        engine_id: Optional[str] = None,
    ):
        """
        Initialize Google PSE search.

        Args:
            api_key: Google API key (or set GOOGLE_PSE_API_KEY env var)
            engine_id: PSE engine ID (or set GOOGLE_PSE_ENGINE_ID env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_PSE_API_KEY")
        self.engine_id = engine_id or os.getenv("GOOGLE_PSE_ENGINE_ID")

    def is_configured(self) -> bool:
        """Check if Google PSE is properly configured."""
        return bool(self.api_key and self.engine_id)

    def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs,
    ) -> SearchResponse:
        """
        Perform a web search using Google PSE.

        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            **kwargs: Additional parameters (e.g., dateRestrict, siteSearch)

        Returns:
            SearchResponse with results

        Raises:
            ValueError: If not configured
            requests.HTTPError: If API request fails
        """
        if not self.is_configured():
            raise ValueError(
                "Google PSE not configured. Set GOOGLE_PSE_API_KEY and GOOGLE_PSE_ENGINE_ID "
                "environment variables or pass api_key and engine_id to constructor."
            )

        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": min(num_results, 10),  # Google PSE max is 10
            **kwargs,
        }

        response = requests.get(self.BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        results = []
        for item in data.get("items", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
            ))

        return SearchResponse(
            results=results,
            query=query,
            backend="google_pse",
        )


class ExaSearch:
    """
    Exa AI-native search integration.

    Exa provides structured, high-quality search results optimized for AI applications.
    Great for real-time data like weather, stock prices, news, etc.

    Requires:
    - EXA_API_KEY: API key from https://exa.ai

    Features:
    - Neural search (semantic understanding)
    - Auto-prompt optimization
    - Content extraction (highlights/full text)
    """

    BASE_URL = "https://api.exa.ai/search"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Exa search.

        Args:
            api_key: Exa API key (or set EXA_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("EXA_API_KEY")

    def is_configured(self) -> bool:
        """Check if Exa is properly configured."""
        return bool(self.api_key)

    def search(
        self,
        query: str,
        num_results: int = 5,
        use_autoprompt: bool = True,
        include_highlights: bool = True,
        **kwargs,
    ) -> SearchResponse:
        """
        Perform a web search using Exa.

        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            use_autoprompt: Let Exa optimize the query for better results
            include_highlights: Include relevant text highlights from pages
            **kwargs: Additional Exa parameters

        Returns:
            SearchResponse with results

        Raises:
            ValueError: If not configured
            requests.HTTPError: If API request fails
        """
        if not self.is_configured():
            raise ValueError(
                "Exa not configured. Set EXA_API_KEY environment variable "
                "or pass api_key to constructor. Get a key at https://exa.ai"
            )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "query": query,
            "numResults": min(num_results, 10),
            "useAutoprompt": use_autoprompt,
            "contents": {
                "highlights": include_highlights,
            },
            **kwargs,
        }

        response = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()

        data = response.json()

        results = []
        for item in data.get("results", []):
            # Exa provides highlights as a list - join them for snippet
            highlights = item.get("highlights", [])
            snippet = " ... ".join(highlights) if highlights else item.get("text", "")[:300]

            results.append(SearchResult(
                title=item.get("title", ""),
                link=item.get("url", ""),
                snippet=snippet,
            ))

        return SearchResponse(
            results=results,
            query=query,
            backend="exa",
        )


class TavilySearch:
    """
    Tavily AI-native search integration.

    Tavily provides search results optimized for LLM applications with
    high-quality content extraction and summarization.

    Requires:
    - TAVILY_API_KEY: API key from https://tavily.com

    Features:
    - AI-optimized search results
    - Content extraction with context
    - Answer generation (optional)
    """

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily search.

        Args:
            api_key: Tavily API key (or set TAVILY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def is_configured(self) -> bool:
        """Check if Tavily is properly configured."""
        return bool(self.api_key)

    def search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
        **kwargs,
    ) -> SearchResponse:
        """
        Perform a web search using Tavily.

        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            search_depth: "basic" (fast) or "advanced" (more thorough)
            include_raw_content: Include full page content (increases response size)
            **kwargs: Additional Tavily parameters

        Returns:
            SearchResponse with results

        Raises:
            ValueError: If not configured
            requests.HTTPError: If API request fails
        """
        if not self.is_configured():
            raise ValueError(
                "Tavily not configured. Set TAVILY_API_KEY environment variable "
                "or pass api_key to constructor. Get a key at https://tavily.com"
            )

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(num_results, 10),
            "search_depth": search_depth,
            "include_raw_content": include_raw_content,
            **kwargs,
        }

        response = requests.post(self.BASE_URL, json=payload, timeout=15)
        response.raise_for_status()

        data = response.json()

        results = []
        for item in data.get("results", []):
            # Tavily provides content field with extracted text
            snippet = item.get("content", "")[:500]  # Limit snippet length

            results.append(SearchResult(
                title=item.get("title", ""),
                link=item.get("url", ""),
                snippet=snippet,
            ))

        return SearchResponse(
            results=results,
            query=query,
            backend="tavily",
        )


def format_search_results_for_context(response: SearchResponse, max_results: int = 5) -> str:
    """
    Format search results as context for the LLM.

    Args:
        response: SearchResponse from a search backend
        max_results: Maximum results to include

    Returns:
        Formatted string for injection into prompt context
    """
    if not response.results:
        return f"Web search for '{response.query}' returned no results."

    lines = [f"Web search results for '{response.query}':"]
    lines.append("")

    for i, result in enumerate(response.results[:max_results], 1):
        lines.append(f"{i}. **{result.title}**")
        lines.append(f"   {result.link}")
        lines.append(f"   {result.snippet}")
        lines.append("")

    return "\n".join(lines)


class WebSearchManager:
    """
    Unified web search manager that selects the best available backend.

    Priority (auto-detect):
    1. Exa (if configured) - best for real-time data, AI-optimized
    2. Tavily (if configured) - AI-native search for LLM apps
    3. Google PSE (if configured) - good general search
    4. OpenAI web_search_preview tool - fallback
    """

    def __init__(
        self,
        exa_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        google_engine_id: Optional[str] = None,
        preferred_backend: Optional[str] = None,
    ):
        """
        Initialize web search manager.

        Args:
            exa_api_key: Exa API key
            tavily_api_key: Tavily API key
            google_api_key: Google PSE API key
            google_engine_id: Google PSE engine ID
            preferred_backend: Force a specific backend ("exa", "tavily", "google_pse", or "openai")
        """
        self.exa = ExaSearch(api_key=exa_api_key)
        self.tavily = TavilySearch(api_key=tavily_api_key)
        self.google_pse = GooglePSESearch(
            api_key=google_api_key,
            engine_id=google_engine_id,
        )
        self.preferred_backend = preferred_backend or os.getenv("WEB_SEARCH_BACKEND")

    def get_available_backend(self) -> Optional[str]:
        """
        Get the best available search backend.

        Returns:
            Backend name or None if none available
        """
        # Check preferred backend first
        if self.preferred_backend:
            if self.preferred_backend == "exa" and self.exa.is_configured():
                return "exa"
            elif self.preferred_backend == "tavily" and self.tavily.is_configured():
                return "tavily"
            elif self.preferred_backend == "google_pse" and self.google_pse.is_configured():
                return "google_pse"
            elif self.preferred_backend == "openai":
                return "openai"

        # Auto-detect: prefer Exa > Tavily > Google PSE > OpenAI
        if self.exa.is_configured():
            return "exa"
        if self.tavily.is_configured():
            return "tavily"
        if self.google_pse.is_configured():
            return "google_pse"

        # Fall back to OpenAI (always "available" but may not work for all providers)
        return "openai"

    def is_exa_available(self) -> bool:
        """Check if Exa is available."""
        return self.exa.is_configured()

    def is_tavily_available(self) -> bool:
        """Check if Tavily is available."""
        return self.tavily.is_configured()

    def is_google_pse_available(self) -> bool:
        """Check if Google PSE is available."""
        return self.google_pse.is_configured()

    def search(
        self,
        query: str,
        num_results: int = 5,
        backend: Optional[str] = None,
    ) -> Optional[SearchResponse]:
        """
        Perform a web search using the best available backend.

        Args:
            query: Search query
            num_results: Number of results
            backend: Override backend selection

        Returns:
            SearchResponse or None if search fails
        """
        use_backend = backend or self.get_available_backend()

        if use_backend == "exa":
            try:
                return self.exa.search(query, num_results=num_results)
            except Exception as e:
                print(f"Exa search failed: {e}")
                # Try fallback to Tavily, then Google PSE
                if self.tavily.is_configured():
                    try:
                        return self.tavily.search(query, num_results=num_results)
                    except Exception as e2:
                        print(f"Tavily fallback also failed: {e2}")
                if self.google_pse.is_configured():
                    try:
                        return self.google_pse.search(query, num_results=num_results)
                    except Exception as e2:
                        print(f"Google PSE fallback also failed: {e2}")
                return None

        if use_backend == "tavily":
            try:
                return self.tavily.search(query, num_results=num_results)
            except Exception as e:
                print(f"Tavily search failed: {e}")
                # Try fallback to Google PSE
                if self.google_pse.is_configured():
                    try:
                        return self.google_pse.search(query, num_results=num_results)
                    except Exception as e2:
                        print(f"Google PSE fallback also failed: {e2}")
                return None

        if use_backend == "google_pse":
            try:
                return self.google_pse.search(query, num_results=num_results)
            except Exception as e:
                print(f"Google PSE search failed: {e}")
                return None

        # OpenAI backend is handled directly in chat_with_memory via tools parameter
        return None

    def get_openai_tool_config(self) -> Optional[Dict[str, Any]]:
        """
        Get OpenAI tool configuration for web search.

        Returns:
            Tool config dict for OpenAI Responses API, or None
        """
        backend = self.get_available_backend()

        if backend == "openai":
            return {"type": "web_search_preview"}

        # Exa, Tavily, and Google PSE don't use OpenAI tools
        return None


# =============================================================================
# Web Fetch - URL Content Extraction
# =============================================================================

@dataclass
class WebFetchResult:
    """Result from fetching a URL."""
    url: str
    title: str
    content: str
    content_type: str
    success: bool
    error: Optional[str] = None


class WebFetcher:
    """
    Fetch and extract content from URLs.

    Supports:
    - HTML pages (extracts main content as text)
    - Plain text files
    - JSON (pretty-printed)

    Security:
    - Validates URL scheme (http/https only)
    - Configurable domain allowlist/blocklist
    - Request timeout
    - Content size limits
    """

    # User-Agent to identify as a bot (respectful crawling)
    USER_AGENT = "SemMemBot/1.0 (Semantic Memory Agent; +https://github.com/)"

    # Default blocked domains (privacy, security)
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
        max_content_length: int = 500_000,  # 500KB default
        timeout: int = 15,
    ):
        """
        Initialize web fetcher.

        Args:
            allowed_domains: If set, only these domains are allowed (whitelist mode)
            blocked_domains: Domains to block (added to defaults)
            max_content_length: Maximum content size in bytes
            timeout: Request timeout in seconds
        """
        self.allowed_domains = set(allowed_domains) if allowed_domains else None
        self.blocked_domains = self.DEFAULT_BLOCKED_DOMAINS.copy()
        if blocked_domains:
            self.blocked_domains.update(blocked_domains)
        self.max_content_length = max_content_length
        self.timeout = timeout

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """
        Validate URL for safety.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return False, "Invalid URL format"

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid scheme: {parsed.scheme}. Only http/https allowed."

        # Check domain
        domain = parsed.netloc.lower()
        if not domain:
            return False, "No domain specified"

        # Remove port from domain for checking
        domain_no_port = domain.split(":")[0]

        # Check blocklist
        if domain_no_port in self.blocked_domains:
            return False, f"Domain blocked: {domain_no_port}"

        # Check for private IP patterns
        if re.match(r"^(10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)", domain_no_port):
            return False, "Private IP addresses not allowed"

        # Check allowlist if set
        if self.allowed_domains is not None:
            if domain_no_port not in self.allowed_domains:
                return False, f"Domain not in allowlist: {domain_no_port}"

        return True, ""

    def _extract_text_from_html(self, html: str, url: str) -> tuple[str, str]:
        """
        Extract main text content from HTML.

        Returns:
            Tuple of (title, content)
        """
        soup = BeautifulSoup(html, "html.parser")

        # Get title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                         "form", "button", "iframe", "noscript"]):
            tag.decompose()

        # Try to find main content
        main_content = None

        # Look for common main content containers
        for selector in ["main", "article", '[role="main"]', ".content",
                        "#content", ".post-content", ".entry-content"]:
            found = soup.select_one(selector)
            if found:
                main_content = found
                break

        # Fall back to body
        if main_content is None:
            main_content = soup.body or soup

        # Extract text
        text = main_content.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        return title, text

    def fetch(self, url: str) -> WebFetchResult:
        """
        Fetch and extract content from a URL.

        Args:
            url: URL to fetch

        Returns:
            WebFetchResult with content or error
        """
        # Validate URL
        is_valid, error = self._validate_url(url)
        if not is_valid:
            return WebFetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                success=False,
                error=error,
            )

        try:
            # Make request
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                stream=True,  # Don't download everything at once
            )
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_content_length:
                return WebFetchResult(
                    url=url,
                    title="",
                    content="",
                    content_type="",
                    success=False,
                    error=f"Content too large: {content_length} bytes (max: {self.max_content_length})",
                )

            # Read content with size limit
            content_bytes = b""
            for chunk in response.iter_content(chunk_size=8192):
                content_bytes += chunk
                if len(content_bytes) > self.max_content_length:
                    return WebFetchResult(
                        url=url,
                        title="",
                        content="",
                        content_type="",
                        success=False,
                        error=f"Content exceeded max size during download",
                    )

            content_type = response.headers.get("Content-Type", "").lower()

            # Handle different content types
            if "application/json" in content_type:
                try:
                    data = json.loads(content_bytes.decode("utf-8"))
                    return WebFetchResult(
                        url=url,
                        title=url,
                        content=json.dumps(data, indent=2),
                        content_type="json",
                        success=True,
                    )
                except json.JSONDecodeError:
                    pass

            if "text/plain" in content_type:
                return WebFetchResult(
                    url=url,
                    title=url,
                    content=content_bytes.decode("utf-8", errors="ignore"),
                    content_type="text",
                    success=True,
                )

            # Default: treat as HTML
            html = content_bytes.decode("utf-8", errors="ignore")
            title, text = self._extract_text_from_html(html, url)

            return WebFetchResult(
                url=url,
                title=title or url,
                content=text,
                content_type="html",
                success=True,
            )

        except requests.Timeout:
            return WebFetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                success=False,
                error=f"Request timed out after {self.timeout}s",
            )
        except requests.RequestException as e:
            return WebFetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                success=False,
                error=str(e),
            )


def format_fetch_result_for_context(result: WebFetchResult, max_chars: int = 5000) -> str:
    """
    Format a web fetch result as context for the LLM.

    Args:
        result: WebFetchResult from fetcher
        max_chars: Maximum characters to include

    Returns:
        Formatted string for injection into prompt context
    """
    if not result.success:
        return f"Failed to fetch {result.url}: {result.error}"

    content = result.content[:max_chars]
    if len(result.content) > max_chars:
        content += f"\n\n[Content truncated - {len(result.content)} total chars]"

    return f"""Content from {result.url}:
Title: {result.title}

{content}"""


def get_web_fetch_tool_definition(api_format: str = "responses") -> Dict[str, Any]:
    """
    Get the OpenAI function tool definition for web_fetch.

    This allows the LLM to proactively request URL content when needed.

    Args:
        api_format: Either "responses" (OpenAI Responses API) or "completions" (Chat Completions).
                   The Responses API uses a flat structure while Chat Completions nests under "function".

    Returns:
        Tool definition dict for the specified API format.
    """
    description = (
        "Fetch and extract content from a URL. Use this when you need to retrieve "
        "the actual content of a webpage to answer a question. This is useful for "
        "getting current data like stock prices, weather, news articles, documentation, "
        "or any other web content. The URL should be a complete http/https URL."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The complete URL to fetch (must start with http:// or https://)",
            },
        },
        "required": ["url"],
    }

    if api_format == "responses":
        # OpenAI Responses API format (flat structure)
        return {
            "type": "function",
            "name": "web_fetch",
            "description": description,
            "parameters": parameters,
            "strict": False,
        }
    else:
        # Chat Completions API format (nested under "function")
        return {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": description,
                "parameters": parameters,
            },
        }


# =============================================================================
# Search + Fetch Workflow
# =============================================================================

def should_use_playwright() -> bool:
    """Check if Playwright should be used for fetching."""
    env_setting = os.getenv("WEB_FETCH_USE_PLAYWRIGHT", "true").lower()
    return PLAYWRIGHT_AVAILABLE and env_setting != "false"


class SearchAndFetch:
    """
    Combined search and fetch workflow.

    This class provides a unified interface for:
    1. Searching the web using Exa, Tavily, or Google PSE
    2. Fetching the full content of search result URLs in parallel using Playwright

    Example:
        saf = SearchAndFetch()
        results = saf.search_and_fetch("Python asyncio best practices", num_results=5)
        context = saf.format_for_context(results)
    """

    def __init__(
        self,
        search_manager: Optional[WebSearchManager] = None,
        use_playwright: Optional[bool] = None,
        max_concurrent_fetches: int = 5,
        fetch_timeout_ms: int = 30_000,
        max_content_per_page: int = 10_000,
    ):
        """
        Initialize search and fetch workflow.

        Args:
            search_manager: WebSearchManager instance (created if None)
            use_playwright: Use Playwright for fetching (auto-detect if None)
            max_concurrent_fetches: Max parallel Playwright page loads
            fetch_timeout_ms: Timeout per page in milliseconds
            max_content_per_page: Max content chars per fetched page
        """
        self.search_manager = search_manager or WebSearchManager()
        self.use_playwright = use_playwright if use_playwright is not None else should_use_playwright()
        self.max_concurrent_fetches = max_concurrent_fetches
        self.fetch_timeout_ms = fetch_timeout_ms
        self.max_content_per_page = max_content_per_page

        # Fallback to requests-based fetcher
        self._requests_fetcher = WebFetcher(
            max_content_length=max_content_per_page,
            timeout=fetch_timeout_ms // 1000,
        )

    def search(self, query: str, num_results: int = 5) -> Optional[SearchResponse]:
        """Perform web search."""
        return self.search_manager.search(query, num_results=num_results)

    def fetch_urls(self, urls: List[str]) -> List[WebFetchResult]:
        """
        Fetch multiple URLs in parallel.

        Uses Playwright if available, falls back to requests.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of WebFetchResult (or PlaywrightFetchResult converted)
        """
        if self.use_playwright and PLAYWRIGHT_AVAILABLE and fetch_urls_sync:
            # Use Playwright for parallel fetching
            playwright_results = fetch_urls_sync(
                urls,
                max_concurrent=self.max_concurrent_fetches,
                timeout_ms=self.fetch_timeout_ms,
            )
            # Convert to WebFetchResult for compatibility
            results = []
            for pr in playwright_results:
                results.append(WebFetchResult(
                    url=pr.url,
                    title=pr.title,
                    content=pr.content[:self.max_content_per_page],
                    content_type=pr.content_type,
                    success=pr.success,
                    error=pr.error,
                ))
            return results
        else:
            # Fallback to sequential requests-based fetching
            results = []
            for url in urls:
                result = self._requests_fetcher.fetch(url)
                results.append(result)
            return results

    async def fetch_urls_async(self, urls: List[str]) -> List[WebFetchResult]:
        """
        Async version of fetch_urls for use in async code.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of WebFetchResult
        """
        if self.use_playwright and PLAYWRIGHT_AVAILABLE and fetch_urls_parallel:
            playwright_results = await fetch_urls_parallel(
                urls,
                max_concurrent=self.max_concurrent_fetches,
                timeout_ms=self.fetch_timeout_ms,
            )
            results = []
            for pr in playwright_results:
                results.append(WebFetchResult(
                    url=pr.url,
                    title=pr.title,
                    content=pr.content[:self.max_content_per_page],
                    content_type=pr.content_type,
                    success=pr.success,
                    error=pr.error,
                ))
            return results
        else:
            # Run requests-based fetching in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: [self._requests_fetcher.fetch(url) for url in urls]
            )
            return results

    def search_and_fetch(
        self,
        query: str,
        num_results: int = 5,
        fetch_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Search the web and optionally fetch full content of results.

        This is the main workflow: search -> get URLs -> fetch all in parallel.

        Args:
            query: Search query
            num_results: Number of search results
            fetch_results: Whether to fetch full content of result URLs

        Returns:
            Dict with:
                - search_response: SearchResponse from search backend
                - fetch_results: List of WebFetchResult (if fetch_results=True)
                - backend: Search backend used
        """
        search_response = self.search(query, num_results=num_results)

        result = {
            "search_response": search_response,
            "fetch_results": [],
            "backend": search_response.backend if search_response else None,
        }

        if search_response and fetch_results and search_response.results:
            urls = [r.link for r in search_response.results]
            result["fetch_results"] = self.fetch_urls(urls)

        return result

    async def search_and_fetch_async(
        self,
        query: str,
        num_results: int = 5,
        fetch_results: bool = True,
    ) -> Dict[str, Any]:
        """Async version of search_and_fetch."""
        # Search is currently sync, run in executor
        loop = asyncio.get_event_loop()
        search_response = await loop.run_in_executor(
            None,
            lambda: self.search(query, num_results=num_results)
        )

        result = {
            "search_response": search_response,
            "fetch_results": [],
            "backend": search_response.backend if search_response else None,
        }

        if search_response and fetch_results and search_response.results:
            urls = [r.link for r in search_response.results]
            result["fetch_results"] = await self.fetch_urls_async(urls)

        return result

    def format_for_context(
        self,
        results: Dict[str, Any],
        include_snippets: bool = True,
        include_full_content: bool = True,
        max_chars_per_result: int = 3000,
        max_total_chars: int = 15000,
    ) -> str:
        """
        Format search and fetch results as context for the LLM.

        Args:
            results: Output from search_and_fetch()
            include_snippets: Include search result snippets
            include_full_content: Include fetched full content
            max_chars_per_result: Max chars per individual result
            max_total_chars: Max total chars

        Returns:
            Formatted context string
        """
        parts = []
        total_chars = 0
        search_response = results.get("search_response")
        fetch_results = results.get("fetch_results", [])

        if not search_response or not search_response.results:
            return "No search results found."

        parts.append(f"Web search results for '{search_response.query}' (via {search_response.backend}):\n")

        # Create a map of URL -> fetched content
        fetch_map = {fr.url: fr for fr in fetch_results if fr.success}

        for i, sr in enumerate(search_response.results, 1):
            if total_chars >= max_total_chars:
                parts.append("\n[Additional results truncated due to length]")
                break

            part_lines = [f"\n{i}. **{sr.title}**"]
            part_lines.append(f"   URL: {sr.link}")

            if include_snippets and sr.snippet:
                snippet = sr.snippet[:500]
                part_lines.append(f"   Snippet: {snippet}")

            if include_full_content and sr.link in fetch_map:
                fetched = fetch_map[sr.link]
                remaining = min(max_chars_per_result, max_total_chars - total_chars - 200)
                if remaining > 100:
                    content = fetched.content[:remaining]
                    if len(fetched.content) > remaining:
                        content += "\n   [Content truncated]"
                    part_lines.append(f"\n   Full content:\n   {content}")

            part = "\n".join(part_lines)
            parts.append(part)
            total_chars += len(part)

        return "\n".join(parts)


# Convenience function for simple use case
def search_and_fetch_urls(
    query: str,
    num_results: int = 5,
    use_playwright: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Simple function to search and fetch URLs.

    Args:
        query: Search query
        num_results: Number of results to fetch
        use_playwright: Use Playwright (auto-detect if None)

    Returns:
        Dict with search_response and fetch_results

    Example:
        results = search_and_fetch_urls("Python asyncio tutorial")
        for fr in results["fetch_results"]:
            if fr.success:
                print(f"{fr.title}: {len(fr.content)} chars")
    """
    saf = SearchAndFetch(use_playwright=use_playwright)
    return saf.search_and_fetch(query, num_results=num_results)


async def search_and_fetch_urls_async(
    query: str,
    num_results: int = 5,
    use_playwright: Optional[bool] = None,
) -> Dict[str, Any]:
    """Async version of search_and_fetch_urls."""
    saf = SearchAndFetch(use_playwright=use_playwright)
    return await saf.search_and_fetch_async(query, num_results=num_results)
