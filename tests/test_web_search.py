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
