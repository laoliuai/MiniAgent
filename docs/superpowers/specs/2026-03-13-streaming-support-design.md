# Streaming Support Design Spec

## Overview

Add streaming support to Mini Agent across all layers: LLM client, Agent core loop, and consumer layers (CLI, ACP). The design follows a bottom-up approach with two layers of stream types, keeping each layer's responsibilities cleanly separated.

## Design Decisions

- **`Agent.run()` stays unchanged** (returns `str`), new `Agent.run_stream()` returns `AsyncGenerator[StreamEvent, None]`
- **`run()` internally wraps `run_stream()`** — single source of truth for the agent loop
- **No config switch** — the underlying LLM calls are always streaming; callers choose `run()` vs `run_stream()`
- **LLM client exposes both `generate()` and `generate_stream()`** — `generate()` is a convenience wrapper that collects the stream into a complete `LLMResponse`
- **Print logic moves out of Agent** into consumers (CLI, ACP) — Agent becomes a pure business engine
- **Retry only covers connection establishment**, not mid-stream failures

## Type Definitions

### LLM Layer: `LLMStreamChunk`

Protocol-level stream type. Handles Anthropic/OpenAI SSE differences. Lives in `schema/schema.py`.

```python
class LLMStreamChunkType(str, Enum):
    TEXT_DELTA = "text_delta"           # Incremental text content
    THINKING_DELTA = "thinking_delta"   # Incremental thinking content
    TOOL_CALL_START = "tool_call_start" # Tool call begins (id, name available)
    TOOL_CALL_DELTA = "tool_call_delta" # Tool call argument JSON fragment
    TOOL_CALL_END = "tool_call_end"     # Tool call arguments complete
    USAGE = "usage"                     # Token usage (at stream end)
    DONE = "done"                       # Stream finished

class LLMStreamChunk(BaseModel):
    type: LLMStreamChunkType
    content: str | None = None          # For text/thinking deltas
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: str | None = None   # JSON fragment for TOOL_CALL_DELTA
    usage: TokenUsage | None = None     # For USAGE type only
    finish_reason: str | None = None    # For DONE type only
```

### Agent Layer: `StreamEvent`

Business-level stream type. Adds Agent concepts (steps, tool execution). Lives in `schema/schema.py`.

```python
class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    STEP_START = "step_start"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    STEP_COMPLETE = "step_complete"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

class StreamEvent(BaseModel):
    type: StreamEventType
    content: str | None = None          # Delta text content
    step: int | None = None             # Current step number
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments: dict | None = None  # Complete tool arguments
    tool_result: ToolResult | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
```

Key distinction: `LLMStreamChunk` is protocol-level (handles JSON fragment assembly), `StreamEvent` is business-level (steps, tool execution — concepts only Agent has).

## LLM Client Layer

### `LLMClientBase` Changes

- `generate_stream()` becomes the **only abstract method** subclasses must implement
- `generate()` moves from abstract to **concrete base class method** that collects `generate_stream()` results

```python
class LLMClientBase(ABC):
    async def generate(self, messages, tools=None) -> LLMResponse:
        """Collect stream into complete LLMResponse. Not abstract — lives in base."""
        text, thinking = "", ""
        tool_calls = []
        current_tool = {}
        usage, finish_reason = None, "stop"

        async for chunk in self.generate_stream(messages, tools):
            match chunk.type:
                case LLMStreamChunkType.TEXT_DELTA:
                    text += chunk.content
                case LLMStreamChunkType.THINKING_DELTA:
                    thinking += chunk.content
                case LLMStreamChunkType.TOOL_CALL_START:
                    current_tool = {"id": chunk.tool_call_id, "name": chunk.tool_name, "arguments_json": ""}
                case LLMStreamChunkType.TOOL_CALL_DELTA:
                    current_tool["arguments_json"] += chunk.tool_arguments
                case LLMStreamChunkType.TOOL_CALL_END:
                    tool_calls.append(ToolCall(
                        id=current_tool["id"], type="function",
                        function=FunctionCall(name=current_tool["name"],
                                              arguments=json.loads(current_tool["arguments_json"]))
                    ))
                case LLMStreamChunkType.USAGE:
                    usage = chunk.usage
                case LLMStreamChunkType.DONE:
                    finish_reason = chunk.finish_reason or "stop"

        return LLMResponse(
            content=text,
            thinking=thinking if thinking else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
        )

    @abstractmethod
    async def generate_stream(self, messages, tools=None) -> AsyncGenerator[LLMStreamChunk, None]:
        """Subclasses implement this."""
        ...
```

### `AnthropicClient.generate_stream()`

Uses `client.messages.stream()` from the Anthropic SDK:

```python
async def generate_stream(self, messages, tools=None):
    request_params = self._prepare_request(messages, tools)
    stream = await self._make_stream_request_with_retry(request_params)

    async with stream as s:
        async for event in s:
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    yield LLMStreamChunk(type=TOOL_CALL_START,
                                         tool_call_id=event.content_block.id,
                                         tool_name=event.content_block.name)
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield LLMStreamChunk(type=TEXT_DELTA, content=event.delta.text)
                elif event.delta.type == "thinking_delta":
                    yield LLMStreamChunk(type=THINKING_DELTA, content=event.delta.thinking)
                elif event.delta.type == "input_json_delta":
                    yield LLMStreamChunk(type=TOOL_CALL_DELTA,
                                         tool_arguments=event.delta.partial_json)
            elif event.type == "content_block_stop":
                # Yield TOOL_CALL_END if the stopped block was a tool_use
                yield LLMStreamChunk(type=TOOL_CALL_END, ...)
            elif event.type == "message_delta":
                yield LLMStreamChunk(type=USAGE, usage=..., finish_reason=...)
            elif event.type == "message_stop":
                yield LLMStreamChunk(type=DONE)
```

### `OpenAIClient.generate_stream()`

Uses `stream=True` with the OpenAI SDK:

```python
async def generate_stream(self, messages, tools=None):
    request_params = self._prepare_request(messages, tools)
    stream = await self._make_stream_request_with_retry(request_params)

    current_tool_calls = {}  # index -> {id, name, arguments}
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            if delta.content:
                yield LLMStreamChunk(type=TEXT_DELTA, content=delta.content)
            # reasoning_content for thinking (OpenAI-compatible models)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield LLMStreamChunk(type=THINKING_DELTA, content=delta.reasoning_content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {"id": tc.id, "name": tc.function.name, "arguments": ""}
                        yield LLMStreamChunk(type=TOOL_CALL_START,
                                             tool_call_id=tc.id, tool_name=tc.function.name)
                    if tc.function.arguments:
                        current_tool_calls[idx]["arguments"] += tc.function.arguments
                        yield LLMStreamChunk(type=TOOL_CALL_DELTA,
                                             tool_arguments=tc.function.arguments)
        if chunk.choices and chunk.choices[0].finish_reason:
            # Emit TOOL_CALL_END for all accumulated tool calls
            for info in current_tool_calls.values():
                yield LLMStreamChunk(type=TOOL_CALL_END,
                                     tool_call_id=info["id"], tool_name=info["name"])
            if chunk.usage:
                yield LLMStreamChunk(type=USAGE, usage=TokenUsage(...))
            yield LLMStreamChunk(type=DONE, finish_reason=chunk.choices[0].finish_reason)
```

### Retry Strategy

Retry wraps **connection establishment only**:

```python
async def _make_stream_request_with_retry(self, params):
    """Create the stream object. This step is retryable."""
    if self.retry_config.enabled:
        retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
        return await retry_decorator(self._make_stream_request)(params)
    return await self._make_stream_request(params)
```

Once the stream object is obtained, iteration happens outside the retry boundary. Mid-stream failures propagate as exceptions to the caller.

### `LLMClient` Wrapper

Transparently forwards the new method:

```python
class LLMClient:
    async def generate_stream(self, messages, tools=None):
        async for chunk in self._client.generate_stream(messages, tools):
            yield chunk
```

## Agent Layer

### `Agent.run_stream()`

The core agent loop, yielding `StreamEvent`:

```python
async def run_stream(self, cancel_event=None) -> AsyncGenerator[StreamEvent, None]:
    if cancel_event is not None:
        self.cancel_event = cancel_event
    self.logger.start_new_run()

    for step in range(self.max_steps):
        # Cancellation check
        if self._check_cancelled():
            self._cleanup_incomplete_messages()
            yield StreamEvent(type=StreamEventType.CANCELLED)
            return

        yield StreamEvent(type=StreamEventType.STEP_START, step=step + 1)

        # Token management
        await self._summarize_messages()

        # Stream LLM response
        text, thinking = "", ""
        tool_calls = []
        current_tool_json = {}
        tool_list = list(self.tools.values())
        self.logger.log_request(messages=self.messages, tools=tool_list)

        try:
            async for chunk in self.llm.generate_stream(self.messages, tool_list):
                match chunk.type:
                    case LLMStreamChunkType.TEXT_DELTA:
                        text += chunk.content
                        yield StreamEvent(type=StreamEventType.TEXT_DELTA,
                                          content=chunk.content, step=step + 1)
                    case LLMStreamChunkType.THINKING_DELTA:
                        thinking += chunk.content
                        yield StreamEvent(type=StreamEventType.THINKING_DELTA,
                                          content=chunk.content, step=step + 1)
                    case LLMStreamChunkType.TOOL_CALL_START:
                        current_tool_json = {
                            "id": chunk.tool_call_id,
                            "name": chunk.tool_name,
                            "arguments_json": ""
                        }
                    case LLMStreamChunkType.TOOL_CALL_DELTA:
                        current_tool_json["arguments_json"] += chunk.tool_arguments
                    case LLMStreamChunkType.TOOL_CALL_END:
                        tc = ToolCall(id=current_tool_json["id"], type="function",
                                      function=FunctionCall(
                                          name=current_tool_json["name"],
                                          arguments=json.loads(current_tool_json["arguments_json"])))
                        tool_calls.append(tc)
                    case LLMStreamChunkType.USAGE:
                        self.api_total_tokens = chunk.usage.total_tokens
        except Exception as e:
            yield StreamEvent(type=StreamEventType.ERROR, content=str(e))
            return

        # Append complete assistant message to history
        assistant_msg = Message(role="assistant", content=text,
                                thinking=thinking or None,
                                tool_calls=tool_calls or None)
        self.messages.append(assistant_msg)
        self.logger.log_response(content=text, thinking=thinking or None,
                                  tool_calls=tool_calls or None, finish_reason="stop")

        # No tool calls → task complete
        if not tool_calls:
            yield StreamEvent(type=StreamEventType.STEP_COMPLETE, step=step + 1)
            yield StreamEvent(type=StreamEventType.DONE, content=text)
            return

        # Cancellation check before tool execution
        if self._check_cancelled():
            self._cleanup_incomplete_messages()
            yield StreamEvent(type=StreamEventType.CANCELLED)
            return

        # Execute tool calls
        for tool_call in tool_calls:
            yield StreamEvent(type=StreamEventType.TOOL_CALL_START,
                              tool_name=tool_call.function.name,
                              tool_call_id=tool_call.id,
                              tool_arguments=tool_call.function.arguments,
                              step=step + 1)

            if tool_call.function.name not in self.tools:
                result = ToolResult(success=False, content="",
                                    error=f"Unknown tool: {tool_call.function.name}")
            else:
                try:
                    tool = self.tools[tool_call.function.name]
                    result = await tool.execute(**tool_call.function.arguments)
                except Exception as e:
                    import traceback
                    result = ToolResult(success=False, content="",
                                        error=f"Tool execution failed: {e}\n{traceback.format_exc()}")

            self.logger.log_tool_result(tool_name=tool_call.function.name,
                                         arguments=tool_call.function.arguments,
                                         result_success=result.success,
                                         result_content=result.content if result.success else None,
                                         result_error=result.error if not result.success else None)

            yield StreamEvent(type=StreamEventType.TOOL_CALL_RESULT,
                              tool_name=tool_call.function.name,
                              tool_call_id=tool_call.id,
                              tool_result=result,
                              step=step + 1)

            self.messages.append(Message(
                role="tool",
                content=result.content if result.success else f"Error: {result.error}",
                tool_call_id=tool_call.id,
                name=tool_call.function.name))

            # Cancellation check after each tool
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                yield StreamEvent(type=StreamEventType.CANCELLED)
                return

        yield StreamEvent(type=StreamEventType.STEP_COMPLETE, step=step + 1)

    # Max steps exceeded
    yield StreamEvent(type=StreamEventType.ERROR,
                      content=f"Task couldn't be completed after {self.max_steps} steps.")
```

### `Agent.run()` — Wrapper

```python
async def run(self, cancel_event=None) -> str:
    final_text = ""
    async for event in self.run_stream(cancel_event):
        if event.type == StreamEventType.DONE:
            final_text = event.content or ""
        elif event.type == StreamEventType.ERROR:
            return event.content or "Unknown error"
        elif event.type == StreamEventType.CANCELLED:
            return "Task cancelled by user."
    return final_text
```

## Consumer Layer

### CLI (`cli.py`)

Replaces the current `agent.run()` call with event-driven rendering:

```python
async for event in agent.run_stream(cancel_event):
    match event.type:
        case StreamEventType.STEP_START:
            # Print step header box
            ...
        case StreamEventType.THINKING_DELTA:
            sys.stdout.write(f"{Colors.DIM}{event.content}{Colors.RESET}")
            sys.stdout.flush()
        case StreamEventType.TEXT_DELTA:
            sys.stdout.write(event.content)
            sys.stdout.flush()
        case StreamEventType.TOOL_CALL_START:
            print(f"\n🔧 Tool Call: {event.tool_name}")
            print(f"   Arguments: {json.dumps(event.tool_arguments, indent=2)}")
        case StreamEventType.TOOL_CALL_RESULT:
            if event.tool_result.success:
                print(f"✓ Result: {event.tool_result.content[:300]}")
            else:
                print(f"✗ Error: {event.tool_result.error}")
        case StreamEventType.STEP_COMPLETE:
            print(f"⏱️  Step {event.step} completed")
        case StreamEventType.DONE:
            pass
        case StreamEventType.ERROR:
            print(f"❌ Error: {event.content}")
        case StreamEventType.CANCELLED:
            print("⚠️  Task cancelled by user.")
```

### ACP Server (`acp/__init__.py`)

`_run_turn()` consumes `run_stream()` for incremental editor updates:

```python
async def _run_turn(self, state, session_id):
    async for event in state.agent.run_stream():
        match event.type:
            case StreamEventType.TEXT_DELTA:
                await self._send(session_id, update_agent_message(text_block(event.content)))
            case StreamEventType.THINKING_DELTA:
                await self._send(session_id, update_agent_thought(text_block(event.content)))
            case StreamEventType.TOOL_CALL_START:
                label = f"🔧 {event.tool_name}()"
                await self._send(session_id, start_tool_call(event.tool_call_id, label,
                                 kind="execute", raw_input=event.tool_arguments))
            case StreamEventType.TOOL_CALL_RESULT:
                status = "completed" if event.tool_result.success else "failed"
                text = event.tool_result.content if event.tool_result.success else event.tool_result.error
                await self._send(session_id, update_tool_call(event.tool_call_id, status=status,
                                 content=[tool_content(text_block(text))], raw_output=text))
            case StreamEventType.DONE:
                return "end_turn"
            case StreamEventType.ERROR:
                return "refusal"
            case StreamEventType.CANCELLED:
                return "cancelled"
    return "max_turn_requests"
```

## Public SDK Surface

`__init__.py` adds new exports:

```python
__all__ = [
    # Existing
    "Agent", "LLMClient", "LLMProvider",
    "Message", "LLMResponse", "ToolCall", "FunctionCall",
    # New — streaming
    "LLMStreamChunk", "LLMStreamChunkType",
    "StreamEvent", "StreamEventType",
]
```

## Files Changed

| File | Change |
|------|--------|
| `schema/schema.py` | Add `LLMStreamChunk`, `LLMStreamChunkType`, `StreamEvent`, `StreamEventType` |
| `llm/base.py` | `generate()` becomes concrete (collects stream); add abstract `generate_stream()` |
| `llm/anthropic_client.py` | Implement `generate_stream()` via `messages.stream()`; remove `generate()` override |
| `llm/openai_client.py` | Implement `generate_stream()` via `stream=True`; remove `generate()` override |
| `llm/llm_wrapper.py` | Add `generate_stream()` passthrough |
| `agent.py` | Add `run_stream()`; `run()` wraps `run_stream()`; remove all print logic |
| `cli.py` | Consume `run_stream()` events for terminal output |
| `acp/__init__.py` | `_run_turn()` consumes `run_stream()` |
| `__init__.py` | Export new types |

## Unchanged

- Configuration system (no stream toggle needed)
- Tool system (tool execution is not streaming)
- Retry core code (same mechanism, different wrapping point)
- Logger (still records complete requests/responses inside Agent)
