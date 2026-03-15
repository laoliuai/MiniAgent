from mini_agent.context.context_editor import ContextEditor, CONTEXT_EDITING_TOOLS
from mini_agent.context.block_store import BlockStore
from mini_agent.context.models import ContextBlock, Layer, BlockType, BlockStatus


def _setup_store():
    store = BlockStore()
    for i in range(1, 6):
        store.add(ContextBlock(
            id=f"turn_{i:03d}_0_tool", turn_id=i, block_type=BlockType.TOOL_CALL,
            layer=Layer.L1_WORKING, original_content=f"content {i}",
            working_content=f"content {i}", token_count=500, original_token_count=500))
    return store


def test_mark_obsolete():
    store = _setup_store()
    editor = ContextEditor(store)
    result = editor.execute("context_mark_obsolete", {"turn_ids": [1, 2], "reason": "superseded by new query"})
    assert "obsolete" in result.lower()
    assert store.get("turn_001_0_tool").status == BlockStatus.OBSOLETE
    assert store.get("turn_002_0_tool").status == BlockStatus.OBSOLETE
    assert store.get("turn_003_0_tool").status == BlockStatus.ACTIVE


def test_mark_obsolete_skips_pinned():
    store = _setup_store()
    store.get("turn_001_0_tool").status = BlockStatus.PINNED
    editor = ContextEditor(store)
    editor.execute("context_mark_obsolete", {"turn_ids": [1], "reason": "test"})
    assert store.get("turn_001_0_tool").status == BlockStatus.PINNED


def test_compress_to_conclusion():
    store = _setup_store()
    editor = ContextEditor(store)
    result = editor.execute("context_compress_to_conclusion", {
        "turn_ids": [1, 2, 3], "conclusion": "The optimal batch size is 64."})
    assert "compressed" in result.lower() or "Compressed" in result
    primary = store.get("turn_001_0_tool")
    assert "optimal batch size is 64" in primary.working_content
    assert store.get("turn_002_0_tool").status == BlockStatus.SUMMARIZED
    assert store.get("turn_003_0_tool").status == BlockStatus.SUMMARIZED


def test_pin():
    store = _setup_store()
    editor = ContextEditor(store)
    result = editor.execute("context_pin", {"turn_ids": [3], "reason": "critical constraint"})
    assert "pinned" in result.lower() or "Pinned" in result
    block = store.get("turn_003_0_tool")
    assert block.status == BlockStatus.PINNED
    assert block.layer == Layer.L0_CORE


def test_unknown_tool():
    store = BlockStore()
    editor = ContextEditor(store)
    result = editor.execute("context_unknown", {})
    assert "unknown" in result.lower() or "Unknown" in result


def test_edit_log_recorded():
    store = _setup_store()
    editor = ContextEditor(store)
    editor.execute("context_mark_obsolete", {"turn_ids": [1], "reason": "test"})
    editor.execute("context_pin", {"turn_ids": [2]})
    assert len(editor.edit_log) == 2
    assert editor.edit_log[0]["action"] == "mark_obsolete"
    assert editor.edit_log[1]["action"] == "pin"


def test_context_editing_tools_schema():
    assert len(CONTEXT_EDITING_TOOLS) == 3
    names = {t["name"] for t in CONTEXT_EDITING_TOOLS}
    assert names == {"context_mark_obsolete", "context_compress_to_conclusion", "context_pin"}
    for tool in CONTEXT_EDITING_TOOLS:
        assert "description" in tool
        assert "input_schema" in tool
