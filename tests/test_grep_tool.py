"""Test cases for GrepTool."""

import tempfile
from pathlib import Path

import pytest

from mini_agent.tools.grep_tool import GrepTool


@pytest.fixture
def search_dir():
    """Create a temp directory with searchable files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "hello.py").write_text("def hello():\n    print('hello world')\n    return True\n")
        (Path(tmpdir) / "goodbye.py").write_text("def goodbye():\n    print('goodbye world')\n    return False\n")
        (Path(tmpdir) / "data.json").write_text('{"key": "value", "hello": "world"}\n')
        (Path(tmpdir) / "empty.txt").write_text("")
        yield tmpdir


@pytest.mark.asyncio
async def test_grep_basic_search(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello")
    assert result.success
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_grep_no_matches(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="nonexistent_string_xyz")
    assert result.success
    assert "No matches found" in result.content


@pytest.mark.asyncio
async def test_grep_glob_filter(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello", glob="*.py")
    assert result.success
    assert "hello" in result.content
    assert "data.json" not in result.content


@pytest.mark.asyncio
async def test_grep_files_only_mode(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello", output_mode="files_only")
    assert result.success
    assert "hello.py" in result.content


@pytest.mark.asyncio
async def test_grep_count_mode(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="world", output_mode="count")
    assert result.success


@pytest.mark.asyncio
async def test_grep_with_context(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="print", context=1)
    assert result.success
    assert "def" in result.content or "return" in result.content


@pytest.mark.asyncio
async def test_grep_max_results(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern=".", max_results=3)
    assert result.success


@pytest.mark.asyncio
async def test_grep_specific_path(search_dir):
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello", path="hello.py")
    assert result.success
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_grep_schema():
    tool = GrepTool()
    schema = tool.to_schema()
    assert schema["name"] == "grep"
    assert "input_schema" in schema
    assert "pattern" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["pattern"]

    openai_schema = tool.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "grep"
