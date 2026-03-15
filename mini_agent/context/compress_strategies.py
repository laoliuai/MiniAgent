import json
import re
from abc import ABC, abstractmethod


class ToolCompressStrategy(ABC):
    @abstractmethod
    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str: ...


class SqlResultStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return DefaultTruncateStrategy().compress(content, tool_name, max_tokens, level)
        rows = data.get("rows", [])
        columns = data.get("columns", list(rows[0].keys()) if rows else [])
        if level == "light":
            return json.dumps({"columns": columns, "total_rows": len(rows),
                               "sample_rows": rows[:10],
                               "note": f"[micro-compressed] Original {len(rows)} rows, showing first 10"},
                              ensure_ascii=False)
        else:
            stats = self._column_stats(rows, columns)
            return (f"[SQL result summary] {len(rows)} rows, {len(columns)} columns\n"
                    f"Columns: {', '.join(str(c) for c in columns)}\n"
                    f"Stats: {json.dumps(stats, ensure_ascii=False)}")

    def _column_stats(self, rows, columns):
        stats = {}
        for col in columns[:8]:
            values = [r.get(col) for r in rows if r.get(col) is not None]
            if not values: continue
            if isinstance(values[0], (int, float)):
                stats[col] = {"min": min(values), "max": max(values), "avg": round(sum(values)/len(values), 2)}
            elif isinstance(values[0], str):
                unique = set(values)
                stats[col] = {"unique": len(unique), "sample": list(unique)[:5]}
        return stats


class CodeOutputStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        lines = content.split("\n")
        errors = [l for l in lines if any(k in l.lower() for k in ["error", "exception", "traceback"])]
        if level == "light":
            head, tail = lines[:20], lines[-20:] if len(lines) > 40 else []
            omitted = max(0, len(lines) - 40)
            result = "\n".join(head)
            if omitted > 0:
                result += f"\n\n[...{omitted} lines omitted...]\n\n" + "\n".join(tail)
            return result
        else:
            parts = [f"[exec result] {len(lines)} lines"]
            if errors: parts.append("Errors: " + "; ".join(errors[:3]))
            parts.append("Tail:\n" + "\n".join(lines[-5:]))
            return "\n".join(parts)


class FileReadStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        lines = content.split("\n")
        structure = self._extract_structure(content)
        if level == "light":
            return "\n".join(lines[:30]) + f"\n\n[...{len(lines)} lines total...]\nStructure: {structure}"
        else:
            return (f"[file summary] {len(lines)} lines\nStructure: {structure}\n"
                    f"[Full content was analyzed in the assistant reply that followed this tool call]")

    def _extract_structure(self, content):
        classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
        functions = re.findall(r"^(?:def|function|const|async)\s+(\w+)", content, re.MULTILINE)
        return f"classes={classes}, functions={functions}"


class SearchResultStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            return DefaultTruncateStrategy().compress(content, tool_name, max_tokens, level)
        if level == "light":
            return json.dumps([{"title": r.get("title",""), "url": r.get("url",""),
                                "snippet": r.get("snippet","")[:200]} for r in results], ensure_ascii=False)
        else:
            titles = [r.get("title", "untitled") for r in results]
            return f"[search summary] {len(results)} results: " + "; ".join(titles)


class PassThroughStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        return content


class DefaultTruncateStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        max_chars = max_tokens * 4
        if len(content) <= max_chars: return content
        head = int(max_chars * 0.6)
        tail = int(max_chars * 0.3)
        omitted = len(content) - head - tail
        return content[:head] + f"\n\n[...{omitted} chars omitted...]\n\n" + content[-tail:]
