"""WebSearchTool: web search with configurable provider chain and automatic fallback."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import quote_plus, unquote

import httpx

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

# ── Unified result format ──

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    score: float = 0.0
    source: str = ""


@dataclass
class WebSearchResponse:
    query: str
    results: list[SearchResult]
    provider: str
    fallback_used: bool = False
    error_log: list[str] = field(default_factory=list)

    def to_tool_result(self, max_results: int = 5) -> str:
        items = self.results[:max_results]
        if not items:
            return f"[web_search] No results found for: {self.query}"

        parts = []
        for i, r in enumerate(items, 1):
            entry = f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}"
            if r.content:
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


# ── Provider ABC ──

class SearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

    def is_available(self) -> bool:
        return True


# ── Provider: Tavily ──

class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("Tavily API key not configured")
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                    "search_depth": "basic",
                },
            )
            if resp.status_code == 429:
                raise RuntimeError("Tavily rate limit exceeded")
            if resp.status_code == 401:
                raise RuntimeError("Tavily API key invalid or quota exhausted")
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:300],
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                source="tavily",
            )
            for r in data.get("results", [])
        ]


# ── Provider: DuckDuckGo (library) ──

class DuckDuckGoProvider(SearchProvider):
    name = "duckduckgo"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            return await self._search_via_library(query, max_results)
        except ImportError:
            raise RuntimeError("duckduckgo-search package not installed")

    async def _search_via_library(self, query: str, max_results: int) -> list[SearchResult]:
        from duckduckgo_search import DDGS

        def _sync_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        raw = await asyncio.to_thread(_sync_search)
        if not raw:
            raise RuntimeError("DuckDuckGo returned no results")
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
                source="duckduckgo",
            )
            for r in raw
        ]


# ── Provider: Baidu (千帆 API) ──

class BaiduProvider(SearchProvider):
    name = "baidu"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("Baidu search API key not configured")
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://qianfan.baidubce.com/v2/ai_search/web_search",
                headers={
                    "X-Appbuilder-Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": [{"content": query, "role": "user"}],
                    "search_source": "baidu_search_v2",
                    "resource_type_filter": [{"type": "web", "top_k": max_results}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", r.get("content", ""))[:300],
                content=r.get("content", ""),
                score=r.get("rerank_score", 0.0),
                source="baidu",
            )
            for r in data.get("references", [])
        ]


# ── Provider: Baidu Raw (HTML scrape) ──

class BaiduRawProvider(SearchProvider):
    name = "baidu_raw"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        url = f"https://www.baidu.com/s?wd={quote_plus(query)}&rn={max_results}"
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
            })
            resp.raise_for_status()
        results = self._parse_baidu_html(resp.text, max_results)
        if not results:
            raise RuntimeError("Baidu raw: could not parse results")
        return results

    @staticmethod
    def _parse_baidu_html(html: str, max_results: int) -> list[SearchResult]:
        results = []
        blocks = re.findall(
            r'<h3[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>\s*</h3>'
            r'(.*?)(?=<h3[^>]*>\s*<a|$)',
            html, re.DOTALL,
        )
        for link, title_html, body_html in blocks[:max_results]:
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            snippet_match = re.search(
                r'(?:class="[^"]*(?:c-abstract|content-right)[^"]*"[^>]*>)(.*?)</(?:span|div)',
                body_html, re.DOTALL,
            )
            snippet = ""
            if snippet_match:
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()[:300]
            if title:
                results.append(SearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source="baidu_raw",
                ))
        return results


# ── Provider: DuckDuckGo Raw (HTML scrape) ──

class DuckDuckGoRawProvider(SearchProvider):
    name = "duckduckgo_raw"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)",
            })
            resp.raise_for_status()
        results = self._parse_ddg_html(resp.text, max_results)
        if not results:
            raise RuntimeError("DuckDuckGo raw: could not parse results")
        return results

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> list[SearchResult]:
        results = []
        links = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</[a-z]',
            html, re.DOTALL,
        )
        for i, (raw_url, title_html) in enumerate(links[:max_results]):
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()[:300]
            url = raw_url
            if "uddg=" in url:
                m = re.search(r'uddg=([^&]+)', url)
                if m:
                    url = unquote(m.group(1))
            if title:
                results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo_raw"))
        return results


# ── Provider registry ──

PROVIDER_REGISTRY: dict[str, type[SearchProvider]] = {
    "tavily": TavilyProvider,
    "duckduckgo": DuckDuckGoProvider,
    "baidu": BaiduProvider,
    "baidu_raw": BaiduRawProvider,
    "duckduckgo_raw": DuckDuckGoRawProvider,
}


def create_providers(
    provider_names: list[str],
    tavily_api_key: str = "",
    baidu_search_api_key: str = "",
) -> list[SearchProvider]:
    """Create provider instances from config names."""
    providers = []
    for name in provider_names:
        cls = PROVIDER_REGISTRY.get(name)
        if cls is None:
            logger.warning(f"Unknown search provider: {name}, skipping")
            continue
        if name == "tavily":
            providers.append(cls(api_key=tavily_api_key or None))
        elif name == "baidu":
            providers.append(cls(api_key=baidu_search_api_key or None))
        else:
            providers.append(cls())
    return providers


# ── WebSearchTool (Tool subclass) ──

class WebSearchTool(Tool):
    """Web search with configurable provider chain and automatic fallback."""

    def __init__(
        self,
        providers: list[SearchProvider],
        cache_ttl_seconds: int = 300,
    ):
        self._providers = providers
        self._cache: dict[str, tuple[float, WebSearchResponse]] = {}
        self._cache_ttl = cache_ttl_seconds

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information. "
            "Returns structured results with titles, URLs, and content snippets. "
            "Use for: recent events, factual lookups, research, "
            "finding documentation, checking current data."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Be specific. "
                        "Good: '2024 China e-commerce market growth rate statistics'. "
                        "Bad: 'e-commerce growth'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int = 5) -> ToolResult:  # type: ignore[override]
        if not query.strip():
            return ToolResult(success=False, content="", error="Empty search query")

        # Check cache
        cached = self._get_cached(query)
        if cached:
            return ToolResult(success=True, content=cached.to_tool_result(max_results))

        # Search with fallback
        response = await self._search_with_fallback(query, max_results)

        # Cache result
        self._cache[query.lower().strip()] = (time.time(), response)

        return ToolResult(success=True, content=response.to_tool_result(max_results))

    async def _search_with_fallback(self, query: str, max_results: int) -> WebSearchResponse:
        errors = []
        fallback_used = False

        for i, provider in enumerate(self._providers):
            if not provider.is_available():
                errors.append(f"{provider.name}: not configured (skipped)")
                if i == 0:
                    fallback_used = True
                continue

            try:
                results = await provider.search(query, max_results)
                if results:
                    return WebSearchResponse(
                        query=query, results=results, provider=provider.name,
                        fallback_used=fallback_used, error_log=errors,
                    )
                errors.append(f"{provider.name}: returned 0 results")
            except Exception as e:
                errors.append(f"{provider.name}: {type(e).__name__}: {e}")
                logger.warning(f"Search provider failed: {provider.name}: {e}")

            fallback_used = True

        return WebSearchResponse(query=query, results=[], provider="none",
                                  fallback_used=True, error_log=errors)

    def _get_cached(self, query: str) -> WebSearchResponse | None:
        key = query.lower().strip()
        if key in self._cache:
            ts, response = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return response
            del self._cache[key]
        return None
