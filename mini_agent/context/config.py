from typing import Optional
from pydantic import BaseModel, computed_field

_PRESETS = {
    "claude_code": {
        "l0_budget_ratio": 0.15,
        "l1_budget_ratio": 0.85,
        "l2_budget_ratio": 0.0,
        "l3_budget_ratio": 0.0,
        "l1_window_turns": 999_999,
        "l2_window_turns": 999_999,
    },
    "hybrid": {
        "l0_budget_ratio": 0.15,
        "l1_budget_ratio": 0.40,
        "l2_budget_ratio": 0.30,
        "l3_budget_ratio": 0.15,
        "l1_window_turns": 8,
        "l2_window_turns": 30,
    },
    "full_layering": {
        "l0_budget_ratio": 0.15,
        "l1_budget_ratio": 0.30,
        "l2_budget_ratio": 0.35,
        "l3_budget_ratio": 0.20,
        "l1_window_turns": 5,
        "l2_window_turns": 15,
    },
}


class ContextConfig(BaseModel):
    """Central configuration for the context manager."""

    mode: str = "hybrid"
    total_token_budget: int = 80_000
    l0_budget_ratio: float = 0.15
    l1_budget_ratio: float = 0.40
    l2_budget_ratio: float = 0.30
    l3_budget_ratio: float = 0.15
    l1_window_turns: int = 8
    l2_window_turns: int = 30
    micro_compress_after_turns: int = 3
    tool_output_max_tokens_l1: int = 2000
    tool_output_max_tokens_l2: int = 200
    auto_compress_trigger_ratio: float = 0.85
    auto_compress_target_ratio: float = 0.60
    auto_compress_model: Optional[str] = None
    enable_context_editing: bool = True
    enable_prefix_caching: bool = True

    @computed_field
    @property
    def layering_enabled(self) -> bool:
        return self.l1_window_turns < 1000

    @computed_field
    @property
    def layer_budgets(self) -> dict[str, int]:
        if not self.layering_enabled:
            return {
                "L0": int(self.total_token_budget * self.l0_budget_ratio),
                "L1": int(self.total_token_budget * (1.0 - self.l0_budget_ratio)),
                "L2": 0,
                "L3": 0,
            }
        return {
            "L0": int(self.total_token_budget * self.l0_budget_ratio),
            "L1": int(self.total_token_budget * self.l1_budget_ratio),
            "L2": int(self.total_token_budget * self.l2_budget_ratio),
            "L3": int(self.total_token_budget * self.l3_budget_ratio),
        }

    @classmethod
    def from_mode(cls, mode: str, **overrides) -> "ContextConfig":
        if mode not in _PRESETS:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(_PRESETS.keys())}")
        defaults = {"mode": mode, **_PRESETS[mode]}
        defaults.update(overrides)
        return cls(**defaults)
