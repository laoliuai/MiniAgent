# Web Search & Fetch Tools Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add WebSearchTool (with configurable provider chain) and WebFetchTool to MiniAgent.

**Architecture:** WebSearchTool uses a provider chain pattern — an ordered list of SearchProvider subclasses, each tried in sequence until one succeeds. Provider order is configurable via YAML. WebFetchTool is a standalone tool that fetches URLs via async httpx and extracts text from HTML.

**Tech Stack:** Python 3.10+, httpx (async), duckduckgo-search (optional), pydantic

**Reference:** `docs/superpowers/specs/web_tools.py`

---

## File Structure

### New Files
| File | Responsibility |
|---|---|
| `mini_agent/tools/web_search_tool.py` | SearchProvider ABC, 5 provider implementations, WebSearchTool |
| `mini_agent/tools/web_fetch_tool.py` | WebFetchTool (async URL fetch + HTML text extraction) |
| `tests/test_web_search.py` | WebSearchTool unit tests (mocked HTTP) |
| `tests/test_web_fetch.py` | WebFetchTool unit tests (mocked HTTP) |

### Modified Files
| File | Changes |
|---|---|
| `mini_agent/config.py` | Add WebSearchConfig, enable_web_search/enable_web_fetch flags |
| `mini_agent/tools/__init__.py` | Export WebSearchTool, WebFetchTool |
| `mini_agent/cli.py` | Load web tools in initialize_base_tools() |

---

## Task 1: WebSearchTool — Providers + Search Tool

**Files:**
- Create: `mini_agent/tools/web_search_tool.py`
- Test: `tests/test_web_search.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_web_search.py
"""Tests for WebSearchTool and providers."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_agent.tools.web_search_tool import (
    WebSearchTool,
    SearchResult,
    WebSearchResponse,
    SearchProvider,
    DuckDuckGoProvider,
    BaiduProvider,
    BaiduRawProvider,
    DuckDuckGoRawProvider,
    TavilyProvider,
)


# --- SearchResult / WebSearchResponse ---

class TestSearchResult:
    def test_defaults(self):
        r = SearchResult(title="T", url="http://x", snippet="S")
        assert r.content == ""
        assert r.score == 0.0
        assert r.source == ""


class TestWebSearchResponse:
    def test_to_tool_result_with_results(self):
        resp = WebSearchResponse(
            query="test",
            results=[SearchResult(title="A", url="http://a", snippet="snip A")],
            provider="mock",
        )
        text = resp.to_tool_result(max_results=5)
        assert "1 results" in text
        assert "A" in text
        assert "http://a" in text

    def test_to_tool_result_empty(self):
        resp = WebSearchResponse(query="test", results=[], provider="none")
        text = resp.to_tool_result()
        assert "No results" in text

    def test_to_tool_result_truncates(self):
        results = [SearchResult(title=f"R{i}", url=f"http://{i}", snippet=f"S{i}") for i in range(10)]
        resp = WebSearchResponse(query="q", results=results, provider="mock")
        text = resp.to_tool_result(max_results=3)
        assert "[3]" in text
        assert "[4]" not in text


# --- Provider availability ---

class TestProviderAvailability:
    def test_tavily_unavailable_without_key(self):
        p = TavilyProvider(api_key=None)
        assert p.is_available() is False

    def test_tavily_available_with_key(self):
        p = TavilyProvider(api_key="test-key")
        assert p.is_available() is True

    def test_baidu_unavailable_without_key(self):
        p = BaiduProvider(api_key=None)
        assert p.is_available() is False

    def test_baidu_available_with_key(self):
        p = BaiduProvider(api_key="test-key")
        assert p.is_available() is True

    def test_duckduckgo_always_available(self):
        assert DuckDuckGoProvider().is_available() is True

    def test_duckduckgo_raw_always_available(self):
        assert DuckDuckGoRawProvider().is_available() is True

    def test_baidu_raw_always_available(self):
        assert BaiduRawProvider().is_available() is True


# --- WebSearchTool schema ---

class TestWebSearchToolSchema:
    def test_name(self):
        tool = WebSearchTool(providers=[])
        assert tool.name == "web_search"

    def test_parameters_has_query(self):
        tool = WebSearchTool(providers=[])
        params = tool.parameters
        assert "query" in params["properties"]
        assert "query" in params["required"]

    def test_schema_format(self):
        tool = WebSearchTool(providers=[])
        schema = tool.to_schema()
        assert schema["name"] == "web_search"
        assert "input_schema" in schema


# --- WebSearchTool execute with mock provider ---

class FakeProvider(SearchProvider):
    name = "fake"

    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error

    async def search(self, query, max_results=5):
        if self._error:
            raise self._error
        return self._results

class FakeFailProvider(SearchProvider):
    name = "fail"

    async def search(self, query, max_results=5):
        raise RuntimeError("provider failed")


class TestWebSearchToolExecute:
    async def test_successful_search(self):
        results = [SearchResult(title="R1", url="http://r1", snippet="S1", source="fake")]
        tool = WebSearchTool(providers=[FakeProvider(results=results)])
        result = await tool.execute(query="test")
        assert result.success is True
        assert "R1" in result.content

    async def test_empty_query(self):
        tool = WebSearchTool(providers=[])
        result = await tool.execute(query="")
        assert result.success is False
        assert "empty" in result.error.lower()

    async def test_fallback_on_failure(self):
        good_results = [SearchResult(title="Good", url="http://good", snippet="OK", source="fake")]
        tool = WebSearchTool(providers=[
            FakeFailProvider(),
            FakeProvider(results=good_results),
        ])
        result = await tool.execute(query="test")
        assert result.success is True
        assert "Good" in result.content

    async def test_all_providers_fail(self):
        tool = WebSearchTool(providers=[FakeFailProvider(), FakeFailProvider()])
        result = await tool.execute(query="test")
        assert result.success is True  # Returns "no results" message, not error
        assert "No results" in result.content

    async def test_cache_hit(self):
        results = [SearchResult(title="Cached", url="http://c", snippet="C", source="fake")]
        provider = FakeProvider(results=results)
        tool = WebSearchTool(providers=[provider], cache_ttl_seconds=300)
        # First call
        await tool.execute(query="cache test")
        # Replace provider with failing one
        tool._providers = [FakeFailProvider()]
        # Second call should hit cache
        result = await tool.execute(query="cache test")
        assert result.success is True
        assert "Cached" in result.content

    async def test_max_results_param(self):
        results = [SearchResult(title=f"R{i}", url=f"http://{i}", snippet=f"S{i}") for i in range(10)]
        tool = WebSearchTool(providers=[FakeProvider(results=results)])
        result = await tool.execute(query="test", max_results=2)
        assert result.success is True
        assert "[2]" in result.content
        assert "[3]" not in result.content
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement web_search_tool.py**

```python
# mini_agent/tools/web_search_tool.py
"""WebSearchTool: web search with configurable provider chain and automatic fallback."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
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

    def __init__(self, api_key: Optional[str] = None):
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

    def __init__(self, api_key: Optional[str] = None):
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
        # Baidu results: <h3 class="..."><a href="...">title</a></h3>
        # Content in <span class="content-right_..."> or <div class="c-abstract">
        blocks = re.findall(
            r'<h3[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>\s*</h3>'
            r'(.*?)(?=<h3[^>]*>\s*<a|$)',
            html, re.DOTALL,
        )
        for link, title_html, body_html in blocks[:max_results]:
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            # Extract snippet from various Baidu result containers
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
            results = [SearchResult(
                title=f"Search results for: {query}",
                url=f"https://duckduckgo.com/?q={quote_plus(query)}",
                snippet="Direct search link. Raw fetch could not parse results.",
                source="duckduckgo_raw",
            )]
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
            # Resolve DDG redirect URL
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

    def _get_cached(self, query: str) -> Optional[WebSearchResponse]:
        key = query.lower().strip()
        if key in self._cache:
            ts, response = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return response
            del self._cache[key]
        return None
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_web_search.py -v`

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/web_search_tool.py tests/test_web_search.py
git commit -m "feat: add WebSearchTool with configurable provider chain (tavily/duckduckgo/baidu)"
```

---

## Task 2: WebFetchTool

**Files:**
- Create: `mini_agent/tools/web_fetch_tool.py`
- Test: `tests/test_web_fetch.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_web_fetch.py
"""Tests for WebFetchTool."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from mini_agent.tools.web_fetch_tool import WebFetchTool


class TestWebFetchToolSchema:
    def test_name(self):
        assert WebFetchTool().name == "web_fetch"

    def test_parameters_has_url(self):
        params = WebFetchTool().parameters
        assert "url" in params["properties"]
        assert "url" in params["required"]

    def test_schema_format(self):
        schema = WebFetchTool().to_schema()
        assert schema["name"] == "web_fetch"


class TestWebFetchToolExecute:
    async def test_empty_url(self):
        tool = WebFetchTool()
        result = await tool.execute(url="")
        assert result.success is False
        assert "url" in result.error.lower()

    async def test_successful_html_fetch(self):
        html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("mini_agent.tools.web_fetch_tool.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            tool = WebFetchTool()
            result = await tool.execute(url="http://example.com")
            assert result.success is True
            assert "Title" in result.content
            assert "Hello world" in result.content

    async def test_json_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"key": "value"}'
        mock_response.raise_for_status = MagicMock()

        with patch("mini_agent.tools.web_fetch_tool.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            tool = WebFetchTool()
            result = await tool.execute(url="http://api.example.com/data")
            assert result.success is True
            assert "key" in result.content

    async def test_content_truncation(self):
        long_text = "x" * 5000
        html = f"<html><body><p>{long_text}</p></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("mini_agent.tools.web_fetch_tool.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            tool = WebFetchTool()
            result = await tool.execute(url="http://example.com")
            assert result.success is True
            assert "truncated" in result.content


class TestHTMLExtraction:
    def test_removes_script_and_style(self):
        tool = WebFetchTool()
        html = "<html><script>var x=1;</script><style>.a{}</style><p>Hello</p></html>"
        text = tool._extract_text(html)
        assert "var x" not in text
        assert ".a{}" not in text
        assert "Hello" in text

    def test_converts_block_elements(self):
        tool = WebFetchTool()
        html = "<h1>Title</h1><p>Para</p><div>Div</div>"
        text = tool._extract_text(html)
        assert "Title" in text
        assert "Para" in text
```

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Implement web_fetch_tool.py**

```python
# mini_agent/tools/web_fetch_tool.py
"""WebFetchTool: fetch and extract content from URLs."""
from __future__ import annotations

import re

import httpx

from .base import Tool, ToolResult


class WebFetchTool(Tool):
    """Fetch and extract content from a specific URL."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch and read the content of a specific URL. "
            "Use when you have an exact URL from search results or the user. "
            "Returns extracted text content from the page."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str) -> ToolResult:  # type: ignore[override]
        if not url.strip():
            return ToolResult(success=False, content="", error="No URL provided")

        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"},
                )
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                text = resp.text[:3000]
                return ToolResult(success=True, content=f"[web_fetch] {url}\nJSON content:\n{text}")

            text = self._extract_text(resp.text)
            if len(text) > 3000:
                text = text[:3000] + f"\n\n[...truncated, total {len(text)} chars]"

            return ToolResult(success=True, content=f"[web_fetch] {url}\n\n{text}")

        except httpx.TimeoutException:
            return ToolResult(success=False, content="", error=f"Timeout fetching {url}")
        except httpx.HTTPError as e:
            return ToolResult(success=False, content="", error=f"Error fetching {url}: {e}")

    @staticmethod
    def _extract_text(html: str) -> str:
        """Extract readable text from HTML without external dependencies."""
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<(?:p|div|h[1-6]|li|br|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
```

- [ ] **Step 4: Run tests, verify they pass**

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/web_fetch_tool.py tests/test_web_fetch.py
git commit -m "feat: add WebFetchTool for async URL content extraction"
```

---

## Task 3: Config + CLI Integration

**Files:**
- Modify: `mini_agent/config.py`
- Modify: `mini_agent/tools/__init__.py`
- Modify: `mini_agent/cli.py`

- [ ] **Step 1: Add WebSearchConfig to config.py**

After `PathGuardConfig`, add:
```python
class WebSearchConfig(BaseModel):
    """Web search provider configuration"""
    providers: list[str] = ["duckduckgo", "duckduckgo_raw"]
    tavily_api_key: str = ""
    baidu_search_api_key: str = ""
    cache_ttl: int = 300
```

Add to `ToolsConfig`:
```python
    enable_web_search: bool = True
    enable_web_fetch: bool = True
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
```

Parse in `from_yaml()`:
```python
        web_search_data = tools_data.get("web_search", {})
        web_search_config = WebSearchConfig(**web_search_data)
```

Pass to `ToolsConfig(... web_search=web_search_config)`.

- [ ] **Step 2: Update tools/__init__.py exports**

Add `WebSearchTool` and `WebFetchTool` to exports.

- [ ] **Step 3: Update CLI initialize_base_tools()**

After MCP tools loading, add:
```python
    # 5. Web tools
    if config.tools.enable_web_search:
        from mini_agent.tools.web_search_tool import WebSearchTool, create_providers
        ws_config = config.tools.web_search
        providers = create_providers(
            ws_config.providers,
            tavily_api_key=ws_config.tavily_api_key,
            baidu_search_api_key=ws_config.baidu_search_api_key,
        )
        tools.append(WebSearchTool(providers=providers, cache_ttl_seconds=ws_config.cache_ttl))
        print(f"{Colors.GREEN}✅ Loaded WebSearch tool (providers: {ws_config.providers}){Colors.RESET}")

    if config.tools.enable_web_fetch:
        from mini_agent.tools.web_fetch_tool import WebFetchTool
        tools.append(WebFetchTool())
        print(f"{Colors.GREEN}✅ Loaded WebFetch tool{Colors.RESET}")
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -q`

- [ ] **Step 5: Commit**

```bash
git add mini_agent/config.py mini_agent/tools/__init__.py mini_agent/cli.py
git commit -m "feat: integrate web tools into config and CLI with configurable provider chain"
```
