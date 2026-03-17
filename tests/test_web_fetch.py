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
