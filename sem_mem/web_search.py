"""
Web search integration for sem-mem.

Supports multiple backends:
1. Exa (recommended) - AI-native search with structured results, great for real-time data
2. Google Programmable Search Engine (PSE) - requires API key + Engine ID
3. OpenAI web_search_preview tool - requires OpenAI API with Responses API support

Configuration via environment variables:
- EXA_API_KEY: Exa API key (get from https://exa.ai)
- GOOGLE_PSE_API_KEY: Google Custom Search API key
- GOOGLE_PSE_ENGINE_ID: Programmable Search Engine ID (cx parameter)
- WEB_SEARCH_BACKEND: "exa", "google_pse", or "openai" (default: auto-detect)
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json

import requests


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
    2. Google PSE (if configured) - good general search
    3. OpenAI web_search_preview tool - fallback
    """

    def __init__(
        self,
        exa_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        google_engine_id: Optional[str] = None,
        preferred_backend: Optional[str] = None,
    ):
        """
        Initialize web search manager.

        Args:
            exa_api_key: Exa API key
            google_api_key: Google PSE API key
            google_engine_id: Google PSE engine ID
            preferred_backend: Force a specific backend ("exa", "google_pse", or "openai")
        """
        self.exa = ExaSearch(api_key=exa_api_key)
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
            elif self.preferred_backend == "google_pse" and self.google_pse.is_configured():
                return "google_pse"
            elif self.preferred_backend == "openai":
                return "openai"

        # Auto-detect: prefer Exa > Google PSE > OpenAI
        if self.exa.is_configured():
            return "exa"
        if self.google_pse.is_configured():
            return "google_pse"

        # Fall back to OpenAI (always "available" but may not work for all providers)
        return "openai"

    def is_exa_available(self) -> bool:
        """Check if Exa is available."""
        return self.exa.is_configured()

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

        # Exa and Google PSE don't use OpenAI tools
        return None
