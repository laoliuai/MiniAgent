from typing import Optional

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            _encoder = False
    return _encoder


def count_tokens(text: Optional[str]) -> int:
    """Count tokens. Uses tiktoken if available, char heuristic as fallback."""
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder and encoder is not False:
        return len(encoder.encode(text))

    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                    or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    ascii_chars = len(text) - cjk_chars
    return int(ascii_chars / 4 + cjk_chars / 1.5)
