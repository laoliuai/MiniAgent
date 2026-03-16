"""Tests for per-call model override in LLM clients."""
import inspect
import pytest
from collections.abc import AsyncGenerator

from mini_agent.llm.base import LLMClientBase
from mini_agent.schema import LLMStreamChunk, LLMStreamChunkType, Message


class FakeClientWithModelCapture(LLMClientBase):
    """Fake client that captures the model parameter passed to generate_stream."""

    def __init__(self):
        super().__init__(api_key="fake", api_base="http://fake", model="default-model")
        self.captured_model = None

    async def generate_stream(self, messages, tools=None, model=None):
        self.captured_model = model or self.model
        yield LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="ok")
        yield LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop")

    def _prepare_request(self, messages, tools=None):
        return {}

    def _convert_messages(self, messages):
        return None, []


class TestModelOverrideSignature:
    def test_generate_stream_accepts_model_param(self):
        sig = inspect.signature(LLMClientBase.generate_stream)
        assert "model" in sig.parameters
        assert sig.parameters["model"].default is None

    def test_generate_accepts_model_param(self):
        sig = inspect.signature(LLMClientBase.generate)
        assert "model" in sig.parameters
        assert sig.parameters["model"].default is None


class TestModelOverrideBehavior:
    async def test_model_none_uses_default(self):
        client = FakeClientWithModelCapture()
        await client.generate([Message(role="user", content="hi")], model=None)
        assert client.captured_model == "default-model"

    async def test_model_override_forwarded(self):
        client = FakeClientWithModelCapture()
        await client.generate(
            [Message(role="user", content="hi")], model="custom-model"
        )
        assert client.captured_model == "custom-model"

    async def test_generate_stream_model_override(self):
        client = FakeClientWithModelCapture()
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="hi")], model="stream-model"
        ):
            chunks.append(chunk)
        assert client.captured_model == "stream-model"
