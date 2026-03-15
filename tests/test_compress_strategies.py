import json
from mini_agent.context.compress_strategies import (
    SqlResultStrategy, CodeOutputStrategy, FileReadStrategy,
    SearchResultStrategy, PassThroughStrategy, DefaultTruncateStrategy,
)


class TestSqlResultStrategy:
    def test_light_keeps_sample_rows(self):
        strategy = SqlResultStrategy()
        data = json.dumps({
            "columns": ["id", "name", "value"],
            "rows": [{"id": i, "name": f"item{i}", "value": i * 10} for i in range(100)]
        })
        result = strategy.compress(data, "execute_sql", 2000, "light")
        parsed = json.loads(result)
        assert parsed["total_rows"] == 100
        assert len(parsed["sample_rows"]) == 10
        assert "columns" in parsed

    def test_aggressive_stats_only(self):
        strategy = SqlResultStrategy()
        data = json.dumps({"rows": [{"id": i, "val": i * 10} for i in range(50)]})
        result = strategy.compress(data, "execute_sql", 200, "aggressive")
        assert "[SQL result summary]" in result
        assert "50 rows" in result

    def test_invalid_json_fallback(self):
        strategy = SqlResultStrategy()
        result = strategy.compress("not json", "execute_sql", 200, "light")
        assert isinstance(result, str)


class TestCodeOutputStrategy:
    def test_light_head_tail(self):
        strategy = CodeOutputStrategy()
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines)
        result = strategy.compress(content, "execute_code", 2000, "light")
        assert "line 0" in result
        assert "line 99" in result
        assert "omitted" in result

    def test_aggressive_errors_and_tail(self):
        strategy = CodeOutputStrategy()
        lines = ["ok", "ok", "Error: something failed", "ok", "final"]
        content = "\n".join(lines)
        result = strategy.compress(content, "execute_code", 200, "aggressive")
        assert "Error: something failed" in result
        assert "final" in result


class TestFileReadStrategy:
    def test_light_keeps_head_and_structure(self):
        strategy = FileReadStrategy()
        content = "class Foo:\n    pass\n" + "\n".join(f"line {i}" for i in range(100))
        result = strategy.compress(content, "read_file", 2000, "light")
        assert "class Foo" in result
        assert "Structure:" in result

    def test_aggressive_structure_only(self):
        strategy = FileReadStrategy()
        content = "class Foo:\n    pass\ndef bar():\n    pass\n" + "x\n" * 200
        result = strategy.compress(content, "read_file", 200, "aggressive")
        assert "[file summary]" in result
        assert "Foo" in result


class TestSearchResultStrategy:
    def test_light_keeps_snippets(self):
        strategy = SearchResultStrategy()
        data = json.dumps([
            {"title": "Result 1", "url": "http://a.com", "snippet": "desc 1"},
            {"title": "Result 2", "url": "http://b.com", "snippet": "desc 2"},
        ])
        result = strategy.compress(data, "web_search", 2000, "light")
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["title"] == "Result 1"

    def test_aggressive_titles_only(self):
        strategy = SearchResultStrategy()
        data = json.dumps([{"title": "A"}, {"title": "B"}])
        result = strategy.compress(data, "web_search", 200, "aggressive")
        assert "[search summary]" in result


class TestPassThroughStrategy:
    def test_returns_unchanged(self):
        strategy = PassThroughStrategy()
        assert strategy.compress("hello", "write_file", 100, "aggressive") == "hello"


class TestDefaultTruncateStrategy:
    def test_short_content_unchanged(self):
        strategy = DefaultTruncateStrategy()
        assert strategy.compress("short", "x", 1000, "light") == "short"

    def test_long_content_truncated(self):
        strategy = DefaultTruncateStrategy()
        content = "x" * 10000
        result = strategy.compress(content, "x", 100, "light")
        assert len(result) < len(content)
        assert "omitted" in result
