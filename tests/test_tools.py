"""Test cases for tools."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mini_agent.tools import BashTool, EditTool, ReadTool, WriteTool


@pytest.mark.asyncio
async def test_read_tool():
    """Test read file tool."""
    print("\n=== Testing ReadTool ===")

    # Create a temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success, f"Read failed: {result.error}"
        # ReadTool now returns content with line numbers in format: "LINE_NUMBER|LINE_CONTENT"
        assert "Hello, World!" in result.content, f"Content mismatch: {result.content}"
        assert "|Hello, World!" in result.content, f"Expected line number format: {result.content}"
        print("✅ ReadTool test passed")
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_write_tool():
    """Test write file tool."""
    print("\n=== Testing WriteTool ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"

        tool = WriteTool()
        result = await tool.execute(path=str(file_path), content="Test content")

        assert result.success, f"Write failed: {result.error}"
        assert file_path.exists(), "File was not created"
        assert file_path.read_text() == "Test content", "Content mismatch"
        print("✅ WriteTool test passed")


@pytest.mark.asyncio
async def test_edit_tool():
    """Test edit file tool."""
    print("\n=== Testing EditTool ===")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        tool = EditTool()
        result = await tool.execute(
            path=temp_path, old_str="World", new_str="Agent"
        )

        assert result.success, f"Edit failed: {result.error}"
        content = Path(temp_path).read_text()
        assert content == "Hello, Agent!", f"Content mismatch: {content}"
        print("✅ EditTool test passed")
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_bash_tool():
    """Test bash command tool."""
    print("\n=== Testing BashTool ===")

    tool = BashTool()

    # Test successful command
    result = await tool.execute(command="echo 'Hello from bash'")
    assert result.success, f"Bash failed: {result.error}"
    assert "Hello from bash" in result.content, f"Output mismatch: {result.content}"
    print("✅ BashTool test passed")

    # Test failed command
    result = await tool.execute(command="exit 1")
    assert not result.success, "Command should have failed"
    print("✅ BashTool error handling test passed")


@pytest.mark.asyncio
async def test_read_tool_large_file_protection():
    """Test that large files get preview + hint instead of full content."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        # Write 2500 lines (exceeds MAX_LINES=2000)
        for i in range(2500):
            f.write(f"Line {i+1}: some content here\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success
        # Should show preview, not all 2500 lines
        assert "2500 lines" in result.content
        assert "Use offset/limit" in result.content
        # Should contain first ~100 lines but NOT line 2500
        assert "|Line 1:" in result.content
        assert "|Line 2500:" not in result.content
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_read_tool_large_file_with_offset():
    """Test that large files can be read with offset/limit."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for i in range(2500):
            f.write(f"Line {i+1}: content\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path, offset=2490, limit=10)

        assert result.success
        assert "|Line 2490:" in result.content
        assert "|Line 2499:" in result.content
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_read_tool_long_line_truncation():
    """Test that long lines get truncated."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("short line\n")
        f.write("x" * 5000 + "\n")  # Very long line
        f.write("another short line\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success
        assert "truncated" in result.content
        assert "5000 chars" in result.content
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_read_tool_metadata_header():
    """Test that output includes file metadata header."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello\nWorld\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success
        assert "[File:" in result.content
        assert "2 lines" in result.content
    finally:
        Path(temp_path).unlink()


async def main():
    """Run all tool tests."""
    print("=" * 80)
    print("Running Tool Tests")
    print("=" * 80)

    await test_read_tool()
    await test_read_tool_large_file_protection()
    await test_read_tool_large_file_with_offset()
    await test_read_tool_long_line_truncation()
    await test_read_tool_metadata_header()
    await test_write_tool()
    await test_edit_tool()
    await test_bash_tool()

    print("\n" + "=" * 80)
    print("All tool tests passed! ✅")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
