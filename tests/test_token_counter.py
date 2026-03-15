from mini_agent.context.token_counter import count_tokens


def test_count_tokens_basic():
    tokens = count_tokens("Hello, world!")
    assert tokens > 0
    assert isinstance(tokens, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_count_tokens_long_text():
    text = "word " * 1000
    tokens = count_tokens(text)
    assert 800 < tokens < 1500


def test_count_tokens_cjk():
    text = "你好世界" * 100
    tokens = count_tokens(text)
    assert tokens > 0
