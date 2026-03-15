# mini_agent/context/context_manager.py
import json
import logging
from typing import Optional
from .config import ContextConfig
from .block_store import BlockStore
from .models import ContextBlock, Layer, BlockType, BlockStatus
from .token_counter import count_tokens
from .classifier import LayerClassifier
from .micro_compressor import MicroCompressor
from .auto_compressor import AutoCompressor
from .context_editor import ContextEditor, CONTEXT_EDITING_TOOLS, CONTEXT_EDITING_PROMPT
from .budget_assembler import BudgetAssembler

logger = logging.getLogger(__name__)


class ContextManager:
    """Facade that orchestrates the 5-stage context management pipeline."""

    def __init__(self, config: ContextConfig, llm_client, tool_registry: dict):
        self.config = config
        self.store = BlockStore()
        self.classifier = LayerClassifier(config)
        self.micro = MicroCompressor(config, tool_registry)
        self.auto = AutoCompressor(config, llm_client)
        self.editor = ContextEditor(self.store)
        self.assembler = BudgetAssembler(config)
        self.current_turn = 0
        self._tool_call_index = 0  # For unique IDs within a turn

    def init_system(self, system_prompt: str):
        """Initialize system block with context editing prompt if enabled."""
        content = system_prompt
        if self.config.enable_context_editing:
            content += "\n\n" + CONTEXT_EDITING_PROMPT
        self.store.add(ContextBlock(
            id="system",
            turn_id=0,
            block_type=BlockType.SYSTEM,
            layer=Layer.L0_CORE,
            status=BlockStatus.PINNED,
            original_content=content,
            working_content=content,
            token_count=count_tokens(content),
            original_token_count=count_tokens(content),
        ))

    def add_user_message(self, content: str):
        """Record user message. First message becomes USER_INTENT."""
        self.current_turn += 1
        self._tool_call_index = 0
        block_type = BlockType.USER_INTENT if self.current_turn == 1 else BlockType.USER_MESSAGE
        tokens = count_tokens(content)
        self.store.add(ContextBlock(
            id=f"turn_{self.current_turn:03d}_user",
            turn_id=self.current_turn,
            block_type=block_type,
            layer=Layer.L1_WORKING,
            original_content=content,
            working_content=content,
            token_count=tokens,
            original_token_count=tokens,
        ))

    def add_assistant_reply(self, content: str, thinking: Optional[str] = None):
        """Record assistant reply."""
        tokens = count_tokens(content)
        tags = ["has_thinking"] if thinking else []
        self.store.add(ContextBlock(
            id=f"turn_{self.current_turn:03d}_assistant",
            turn_id=self.current_turn,
            block_type=BlockType.ASSISTANT_REPLY,
            layer=Layer.L1_WORKING,
            original_content=content,
            working_content=content,
            token_count=tokens,
            original_token_count=tokens,
            tags=tags,
        ))

    def add_tool_call(self, tool_name: str, tool_input: dict, tool_result: str,
                      tool_call_id: Optional[str] = None):
        """Record tool_use + tool_result as a single TOOL_CALL block.

        tool_call_id: the original LLM-assigned tool_use ID (for API compatibility).
        tool_input: stored as full JSON (tool inputs are typically small).
        """
        tokens = count_tokens(tool_result)
        input_json = json.dumps(tool_input, ensure_ascii=False)
        self.store.add(ContextBlock(
            id=f"turn_{self.current_turn:03d}_{self._tool_call_index}_{tool_name}",
            turn_id=self.current_turn,
            block_type=BlockType.TOOL_CALL,
            layer=Layer.L1_WORKING,
            original_content=tool_result,
            working_content=tool_result,
            token_count=tokens,
            original_token_count=tokens,
            tool_name=tool_name,
            tool_input_summary=input_json,
            tool_call_id=tool_call_id,
        ))
        self._tool_call_index += 1

    async def process_and_assemble(self) -> tuple[list[dict], list[str]]:
        """Run stages 1→2→3, assemble final messages as dicts.

        Returns:
            (messages, events) — messages for LLM, events for logging.
        """
        events: list[str] = []
        all_blocks = self.store.all()

        # Stage 1: Classify
        self.classifier.classify(all_blocks, self.current_turn)

        # Stage 2: Micro compress — track what got compressed
        tokens_before = self.store.total_active_tokens()
        self.micro.compress(all_blocks, self.current_turn)
        tokens_after_micro = self.store.total_active_tokens()
        micro_saved = tokens_before - tokens_after_micro
        if micro_saved > 0:
            compressed_count = sum(
                1 for b in all_blocks
                if b.status == BlockStatus.MICRO_COMPRESSED
                and b.compression_history
                and b.compression_history[-1].get("stage") == "micro"
            )
            events.append(
                f"micro_compress: {compressed_count} blocks, saved {micro_saved} tokens"
            )

        # Stage 3: Auto compress (async — may call LLM)
        if self.auto.should_trigger(self.store):
            summary = await self.auto.compress(self.store, self.current_turn)
            if summary:
                events.append(
                    f"auto_compress: turns {summary.id}, "
                    f"freed ~{tokens_after_micro - self.store.total_active_tokens()} tokens (LLM summary)"
                )

        # Dynamic mode upgrade
        old_mode = self.config.mode
        self._maybe_upgrade_mode()
        if self.config.mode != old_mode:
            events.append(f"mode_upgrade: {old_mode} → {self.config.mode}")

        # Stage 5: Assemble
        messages, usage = self.assembler.assemble(self.store.all())
        total_used = sum(usage.values())
        events.append(
            f"assembled: L0={usage['L0']} L1={usage['L1']} L2={usage['L2']} L3={usage['L3']} "
            f"total={total_used}/{self.config.total_token_budget} "
            f"({total_used / self.config.total_token_budget:.0%})"
        )
        return messages, events

    def handle_context_tool(self, tool_name: str, tool_input: dict) -> str:
        """Route context_* tool calls to ContextEditor."""
        return self.editor.execute(tool_name, tool_input)

    def get_system_prompt_section(self) -> str:
        """Return context editing guidance for system prompt."""
        if self.config.enable_context_editing:
            return CONTEXT_EDITING_PROMPT
        return ""

    def get_context_tools(self) -> list[dict]:
        """Return context editing tool definitions."""
        if self.config.enable_context_editing:
            return CONTEXT_EDITING_TOOLS
        return []

    def get_status(self) -> dict:
        """Return monitoring/debugging info."""
        return {
            "total_blocks": len(self.store.all()),
            "active_blocks": len(self.store.active_blocks()),
            "total_active_tokens": self.store.total_active_tokens(),
            "budget_usage": (
                self.store.total_active_tokens() / self.config.total_token_budget
                if self.config.total_token_budget > 0 else 0
            ),
            "blocks_per_layer": {
                layer.name: len([b for b in self.store.active_blocks() if b.layer == layer])
                for layer in Layer
            },
            "edit_log": self.editor.edit_log,
            "layering_enabled": self.config.layering_enabled,
            "current_turn": self.current_turn,
        }

    def _maybe_upgrade_mode(self):
        """Auto-upgrade Claude Code → Hybrid when pressure builds."""
        usage_ratio = (
            self.store.total_active_tokens() / self.config.total_token_budget
            if self.config.total_token_budget > 0 else 0
        )
        # Only upgrade when layering is disabled (Claude Code mode) and pressure builds
        if (self.config.l1_window_turns > 1000
                and usage_ratio > 0.70
                and self.current_turn > 20):
            new_config = ContextConfig.from_mode(
                "hybrid", total_token_budget=self.config.total_token_budget
            )
            self.config = new_config
            # Propagate to ALL stages
            self.classifier.config = new_config
            self.micro.config = new_config
            self.auto.config = new_config
            self.assembler.config = new_config
            # Re-classify with new config
            self.classifier.classify(self.store.all(), self.current_turn)
