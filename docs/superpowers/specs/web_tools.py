"""
Web Search Tool — Provider Chain with Automatic Fallback

Architecture:
  Agent sees: web_search(query) → structured results
  Internally: Tavily → DuckDuckGo → raw HTTP fetch

Each provider returns the same WebSearchResult format.
Fallback is transparent — the agent never knows which backend served the results.

Dependencies:
  Required: requests (or httpx)
  Optional: tavily-python (for Tavily provider)
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


# ============================================================
# Unified result format
# ============================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str                 # Short description
    content: str = ""            # Extracted page content (if available)
    score: float = 0.0           # Relevance score (0-1, provider-dependent)
    source: str = ""             # Which provider returned this


@dataclass
class WebSearchResponse:
    query: str
    results: list[SearchResult]
    provider: str                # Which provider succeeded
    fallback_used: bool = False  # Whether we fell back from a higher-priority provider
    error_log: list[str] = field(default_factory=list)  # Errors from failed providers

    def to_tool_result(self, max_results: int = 5) -> str:
        """Format as the string returned to the agent via tool_result."""
        items = self.results[:max_results]
        if not items:
            return f"[web_search] No results found for: {self.query}"

        parts = []
        for i, r in enumerate(items, 1):
            entry = f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}"
            if r.content:
                # Truncate content to avoid context bloat
                preview = r.content[:600].strip()
                if len(r.content) > 600:
                    preview += "..."
                entry += f"\n    Content: {preview}"
            parts.append(entry)

        header = f"[web_search] {len(items)} results for \"{self.query}\""
        if self.fallback_used:
            header += f" (via {self.provider}, fallback)"
        else:
            header += f" (via {self.provider})"

        return header + "\n\n" + "\n\n".join(parts)


# ============================================================
# Provider interface
# ============================================================

class SearchProvider(ABC):
    """Base class for search providers."""

    name: str = "base"
    requires_api_key: bool = False

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute search. Raise Exception on failure."""
        ...

    def is_available(self) -> bool:
        """Quick check: is this provider configured and likely to work?"""
        return True


# ============================================================
# Provider 1: Tavily (highest quality, paid)
# ============================================================

class TavilyProvider(SearchProvider):
    """
    Tavily Search API.
    Returns structured results with extracted page content.
    Free tier: 1000 requests/month.
    """

    name = "tavily"
    requires_api_key = True

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._get_key_from_env()

    def _get_key_from_env(self) -> Optional[str]:
        import os
        return os.environ.get("TAVILY_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("Tavily API key not configured")

        import requests

        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "search_depth": "basic",
            },
            timeout=15,
        )

        if resp.status_code == 429:
            raise RuntimeError("Tavily rate limit exceeded")
        if resp.status_code == 401:
            raise RuntimeError("Tavily API key invalid or quota exhausted")
        resp.raise_for_status()

        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:300],
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                source="tavily",
            ))
        return results


# ============================================================
# Provider 2: DuckDuckGo (free, no API key)
# ============================================================

class DuckDuckGoProvider(SearchProvider):
    """
    DuckDuckGo Instant Answer + HTML scrape.
    Free, no API key required.
    Uses the duckduckgo-search library if available,
    otherwise falls back to the Instant Answer API.
    """

    name = "duckduckgo"
    requires_api_key = False

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        # Try the duckduckgo-search library first (best results)
        try:
            return self._search_via_library(query, max_results)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"duckduckgo-search library failed: {e}")

        # Fallback to Instant Answer API (limited but always available)
        return self._search_via_instant_api(query, max_results)

    def _search_via_library(self, query: str, max_results: int) -> list[SearchResult]:
        """Uses duckduckgo-search package: pip install duckduckgo-search"""
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    content="",  # DDG doesn't extract full content
                    source="duckduckgo",
                ))
        if not results:
            raise RuntimeError("DuckDuckGo returned no results")
        return results

    def _search_via_instant_api(self, query: str, max_results: int) -> list[SearchResult]:
        """DuckDuckGo Instant Answer API — limited but no dependencies."""
        import requests

        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []

        # Abstract (direct answer)
        if data.get("Abstract"):
            results.append(SearchResult(
                title=data.get("Heading", query),
                url=data.get("AbstractURL", ""),
                snippet=data["Abstract"][:300],
                content=data["Abstract"],
                source="duckduckgo_instant",
            ))

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(SearchResult(
                    title=topic.get("Text", "")[:80],
                    url=topic.get("FirstURL", ""),
                    snippet=topic.get("Text", "")[:300],
                    content="",
                    source="duckduckgo_instant",
                ))

        if not results:
            raise RuntimeError("DuckDuckGo Instant API returned no results")
        return results[:max_results]


# ============================================================
# Provider 3: Raw HTTP Fetch (always available, lowest quality)
# ============================================================

class RawFetchProvider(SearchProvider):
    """
    Last resort: construct search URL, fetch results page, parse HTML.
    No API key, no dependencies beyond requests + basic HTML parsing.
    Quality is low but it never fails (network permitting).
    """

    name = "raw_fetch"
    requires_api_key = False

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        import requests

        # Use a lightweight search endpoint
        # (Google's HTML results page with minimal parsing)
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"
        }

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        results = self._parse_ddg_html(resp.text, max_results)
        if not results:
            # Truly last resort: return a single result pointing to search
            results = [SearchResult(
                title=f"Search results for: {query}",
                url=f"https://duckduckgo.com/?q={quote_plus(query)}",
                snippet=f"Direct search link. Raw fetch could not parse results.",
                source="raw_fetch_fallback",
            )]
        return results

    def _parse_ddg_html(self, html: str, max_results: int) -> list[SearchResult]:
        """Minimal HTML parsing without BeautifulSoup dependency."""
        import re

        results = []
        # DDG HTML results have a pattern: <a class="result__a" href="...">title</a>
        # and <a class="result__snippet">snippet</a>
        links = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</[a-z]',
            html, re.DOTALL
        )

        for i, (url, title) in enumerate(links[:max_results]):
            # Clean HTML tags from title and snippet
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = ""
            if i < len(snippets):
                clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

            # DDG HTML wraps the real URL in a redirect
            if "uddg=" in url:
                real_url = re.search(r'uddg=([^&]+)', url)
                if real_url:
                    from urllib.parse import unquote
                    url = unquote(real_url.group(1))

            if clean_title:
                results.append(SearchResult(
                    title=clean_title,
                    url=url,
                    snippet=clean_snippet[:300],
                    content="",
                    source="raw_fetch",
                ))
        return results


# ============================================================
# Provider Chain (the actual tool implementation)
# ============================================================

class WebSearchTool:
    """
    Web search tool with automatic provider fallback.

    Usage as a framework tool:
        tool = WebSearchTool()
        result_str = tool.execute({"query": "2024 电商增长率", "max_results": 5})
        # Returns formatted string for tool_result

    Provider chain: Tavily → DuckDuckGo → Raw Fetch
    Fallback is transparent to the agent.
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        providers: Optional[list[SearchProvider]] = None,
        cache_ttl_seconds: int = 300,        # Cache results for 5 minutes
        max_retries_per_provider: int = 1,
    ):
        # Default provider chain
        self.providers = providers or [
            TavilyProvider(api_key=tavily_api_key),
            DuckDuckGoProvider(),
            RawFetchProvider(),
        ]
        self.max_retries = max_retries_per_provider

        # Simple in-memory cache
        self._cache: dict[str, tuple[float, WebSearchResponse]] = {}
        self._cache_ttl = cache_ttl_seconds

    def execute(self, tool_input: dict) -> str:
        """
        Main entry point. Called by the agent framework when LLM uses web_search tool.

        Args:
            tool_input: {"query": "...", "max_results": 5}
        Returns:
            Formatted string for tool_result.
        """
        query = tool_input.get("query", "")
        max_results = tool_input.get("max_results", 5)

        if not query.strip():
            return "[web_search] Error: empty query"

        # Check cache
        cached = self._get_cached(query)
        if cached:
            return cached.to_tool_result(max_results)

        # Try providers in order
        response = self._search_with_fallback(query, max_results)

        # Cache successful result
        self._cache[query.lower().strip()] = (time.time(), response)

        return response.to_tool_result(max_results)

    def _search_with_fallback(self, query: str, max_results: int) -> WebSearchResponse:
        """Try each provider in order. Fall back on failure."""
        errors = []
        fallback_used = False

        for i, provider in enumerate(self.providers):
            if not provider.is_available():
                errors.append(f"{provider.name}: not configured (skipped)")
                if i == 0:
                    fallback_used = True
                continue

            for attempt in range(self.max_retries + 1):
                try:
                    results = provider.search(query, max_results)
                    if results:
                        return WebSearchResponse(
                            query=query,
                            results=results,
                            provider=provider.name,
                            fallback_used=fallback_used,
                            error_log=errors,
                        )
                    else:
                        errors.append(f"{provider.name}: returned 0 results")
                        break
                except Exception as e:
                    error_msg = f"{provider.name}: {type(e).__name__}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(f"Search provider failed: {error_msg}")
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (attempt + 1))  # Brief backoff
                    continue

            # This provider failed → next one is a fallback
            fallback_used = True

        # All providers failed
        return WebSearchResponse(
            query=query, results=[], provider="none",
            fallback_used=True, error_log=errors,
        )

    def _get_cached(self, query: str) -> Optional[WebSearchResponse]:
        key = query.lower().strip()
        if key in self._cache:
            ts, response = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return response
            else:
                del self._cache[key]
        return None

    # ── Tool definition for agent framework registration ──

    @staticmethod
    def tool_definition() -> dict:
        """Returns the tool schema to register with the agent."""
        return {
            "name": "web_search",
            "description": (
                "Search the web for current information. "
                "Returns structured results with titles, URLs, and content snippets. "
                "Use for: recent events, factual lookups, research, "
                "finding documentation, checking current data."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Be specific. "
                            "Good: '2024 China e-commerce market growth rate statistics'. "
                            "Bad: 'e-commerce growth'."
                        )
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


# ============================================================
# Companion tool: web_fetch (fetch a specific URL)
# ============================================================

class WebFetchTool:
    """
    Fetch and extract content from a specific URL.
    Companion to web_search — use when the agent knows the exact URL.
    """

    def execute(self, tool_input: dict) -> str:
        import requests

        url = tool_input.get("url", "")
        if not url:
            return "[web_fetch] Error: no URL provided"

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"},
                timeout=15,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                return f"[web_fetch] {url}\nJSON content:\n{resp.text[:3000]}"

            # Extract text from HTML
            text = self._extract_text(resp.text)
            if len(text) > 3000:
                text = text[:3000] + f"\n\n[...truncated, total {len(text)} chars]"

            return f"[web_fetch] {url}\n\n{text}"

        except requests.Timeout:
            return f"[web_fetch] Timeout fetching {url}"
        except requests.RequestException as e:
            return f"[web_fetch] Error fetching {url}: {e}"

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML. No BeautifulSoup dependency."""
        import re

        # Remove script and style blocks
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Convert block elements to newlines
        text = re.sub(r'<(?:p|div|h[1-6]|li|br|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)

        # Strip remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    @staticmethod
    def tool_definition() -> dict:
        return {
            "name": "web_fetch",
            "description": (
                "Fetch and read the content of a specific URL. "
                "Use when you have an exact URL from search results or the user. "
                "Returns extracted text content from the page."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }


# ============================================================
# Registration helper
# ============================================================

def create_web_tools(tavily_api_key: Optional[str] = None) -> dict:
    """
    Create web search and fetch tools, ready to register with the agent.

    Usage:
        web_tools = create_web_tools(tavily_api_key="tvly-xxx")
        session = Session.create(
            tools=[execute_sql, execute_code, **web_tools],
            ...
        )

    Or without Tavily (auto-fallback to free providers):
        web_tools = create_web_tools()  # No key needed
    """
    search_tool = WebSearchTool(tavily_api_key=tavily_api_key)
    fetch_tool = WebFetchTool()

    return {
        "web_search": {
            "definition": search_tool.tool_definition(),
            "executor": search_tool.execute,
        },
        "web_fetch": {
            "definition": fetch_tool.tool_definition(),
            "executor": fetch_tool.execute,
        },
    }
