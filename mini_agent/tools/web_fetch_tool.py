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
                    text = resp.text
                    if len(text) > 3000:
                        text = text[:3000] + f"\n\n[...truncated, total {len(text)} chars]"
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
