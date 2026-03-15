from mini_agent.context.config import ContextConfig


def test_default_is_hybrid():
    config = ContextConfig()
    assert config.mode == "hybrid"
    assert config.l1_window_turns == 8
    assert config.l2_window_turns == 30
    assert config.layering_enabled is True


def test_claude_code_mode():
    config = ContextConfig.from_mode("claude_code")
    assert config.l1_window_turns == 999_999
    assert config.l1_budget_ratio == 0.85
    assert config.l2_budget_ratio == 0.0
    assert config.layering_enabled is False


def test_hybrid_mode():
    config = ContextConfig.from_mode("hybrid")
    assert config.l1_window_turns == 8
    assert config.l1_budget_ratio == 0.40
    assert config.l2_budget_ratio == 0.30
    assert config.l3_budget_ratio == 0.15


def test_full_layering_mode():
    config = ContextConfig.from_mode("full_layering")
    assert config.l1_window_turns == 5
    assert config.l2_window_turns == 15
    assert config.l1_budget_ratio == 0.30


def test_layer_budgets_with_layering():
    config = ContextConfig.from_mode("hybrid", total_token_budget=100_000)
    budgets = config.layer_budgets
    assert budgets["L0"] == 15_000
    assert budgets["L1"] == 40_000
    assert budgets["L2"] == 30_000
    assert budgets["L3"] == 15_000


def test_layer_budgets_without_layering():
    config = ContextConfig.from_mode("claude_code", total_token_budget=100_000)
    budgets = config.layer_budgets
    assert budgets["L0"] == 15_000
    assert budgets["L1"] == 85_000
    assert budgets["L2"] == 0
    assert budgets["L3"] == 0


def test_from_mode_with_overrides():
    config = ContextConfig.from_mode("hybrid", total_token_budget=120_000,
                                     enable_context_editing=False)
    assert config.total_token_budget == 120_000
    assert config.enable_context_editing is False
    assert config.l1_window_turns == 8


def test_enable_context_editing_default_true():
    config = ContextConfig()
    assert config.enable_context_editing is True


def test_context_config_in_project_config():
    from mini_agent.config import Config
    # Config requires llm.api_key — check that context field exists on the class
    assert "context" in Config.model_fields
    # Verify default value
    assert Config.model_fields["context"].default == ContextConfig()
