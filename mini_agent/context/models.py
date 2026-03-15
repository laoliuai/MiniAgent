from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Layer(Enum):
    L0_CORE = 0
    L1_WORKING = 1
    L2_REFERENCE = 2
    L3_ARCHIVE = 3


class BlockType(Enum):
    SYSTEM = "system"
    USER_INTENT = "user_intent"
    USER_MESSAGE = "user_message"
    ASSISTANT_REPLY = "assistant"
    TOOL_CALL = "tool_call"
    SUMMARY = "summary"
    PINNED = "pinned"


class BlockStatus(Enum):
    ACTIVE = "active"
    MICRO_COMPRESSED = "micro"
    SUMMARIZED = "summarized"
    OBSOLETE = "obsolete"
    PINNED = "pinned"


@dataclass
class ContextBlock:
    id: str
    turn_id: int
    block_type: BlockType
    layer: Layer
    status: BlockStatus = BlockStatus.ACTIVE

    original_content: str = ""
    working_content: str = ""

    token_count: int = 0
    original_token_count: int = 0

    tool_name: Optional[str] = None
    tool_input_summary: str = ""
    tool_call_id: Optional[str] = None

    depends_on: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    compression_history: list[dict] = field(default_factory=list)
