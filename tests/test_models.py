"""Tests for modernized models with pattern matching and enhanced features."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models import (
    MessageRole,
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaShowRequest,
    RequestMetadata,
    ResponseFormat,
)


class TestMessageRole:
    """Test MessageRole enum with pattern matching features."""

    def test_from_string_basic(self):
        """Test basic string conversion."""
        assert MessageRole.from_string("user") == MessageRole.USER
        assert MessageRole.from_string("assistant") == MessageRole.ASSISTANT
        assert MessageRole.from_string("system") == MessageRole.SYSTEM
        assert MessageRole.from_string("tool") == MessageRole.TOOL

    def test_from_string_aliases(self):
        """Test alias support in from_string."""
        assert MessageRole.from_string("human") == MessageRole.USER
        assert MessageRole.from_string("ai") == MessageRole.ASSISTANT
        assert MessageRole.from_string("bot") == MessageRole.ASSISTANT
        assert MessageRole.from_string("sys") == MessageRole.SYSTEM
        assert MessageRole.from_string("function") == MessageRole.TOOL

    def test_from_string_case_insensitive(self):
        """Test case insensitive conversion."""
        assert MessageRole.from_string("USER") == MessageRole.USER
        assert MessageRole.from_string("Assistant") == MessageRole.ASSISTANT
        assert MessageRole.from_string("SYSTEM") == MessageRole.SYSTEM

    def test_from_string_invalid(self):
        """Test invalid role raises error."""
        with pytest.raises(ValueError, match="Invalid message role"):
            MessageRole.from_string("invalid")

    def test_is_user_role(self):
        """Test user role detection."""
        assert MessageRole.USER.is_user_role()
        assert MessageRole.SYSTEM.is_user_role()
        assert not MessageRole.ASSISTANT.is_user_role()
        assert not MessageRole.TOOL.is_user_role()

    def test_is_assistant_role(self):
        """Test assistant role detection."""
        assert MessageRole.ASSISTANT.is_assistant_role()
        assert not MessageRole.USER.is_assistant_role()
        assert not MessageRole.SYSTEM.is_assistant_role()
        assert not MessageRole.TOOL.is_assistant_role()

    def test_get_display_name(self):
        """Test display name generation."""
        assert MessageRole.USER.get_display_name() == "User"
        assert MessageRole.ASSISTANT.get_display_name() == "Assistant"
        assert MessageRole.SYSTEM.get_display_name() == "System"
        assert MessageRole.TOOL.get_display_name() == "Tool"


class TestRequestMetadata:
    """Test RequestMetadata dataclass with modern features."""

    def test_default_creation(self):
        """Test default metadata creation."""
        metadata = RequestMetadata()
        assert metadata.request_id.startswith("req_")
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.user_agent is None
        assert metadata.client_ip is None

    def test_custom_creation(self):
        """Test metadata creation with custom values."""
        metadata = RequestMetadata(
            user_agent="test-agent",
            client_ip="127.0.0.1",
            correlation_id="test-correlation",
            session_id="test-session",
        )
        assert metadata.user_agent == "test-agent"
        assert metadata.client_ip == "127.0.0.1"
        assert metadata.correlation_id == "test-correlation"
        assert metadata.session_id == "test-session"

    def test_age_seconds(self):
        """Test age calculation."""
        metadata = RequestMetadata()
        age = metadata.age_seconds()
        assert age >= 0
        assert age < 1  # Should be very small for new metadata

    def test_is_expired(self):
        """Test expiration check."""
        metadata = RequestMetadata()
        assert not metadata.is_expired(max_age_seconds=300.0)
        assert metadata.is_expired(max_age_seconds=0.0)

    def test_to_dict(self):
        """Test dictionary conversion."""
        metadata = RequestMetadata(user_agent="test")
        result = metadata.to_dict()

        assert "request_id" in result
        assert "timestamp" in result
        assert "age_seconds" in result
        assert result["user_agent"] == "test"
        assert isinstance(result["timestamp"], str)  # ISO format


class TestOllamaChatMessage:
    """Test OllamaChatMessage with enhanced features."""

    def test_basic_creation(self):
        """Test basic message creation."""
        msg = OllamaChatMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.images is None

    def test_content_length(self):
        """Test content length property."""
        msg = OllamaChatMessage(role=MessageRole.USER, content="Hello world")
        assert msg.content_length == 11

    def test_is_empty(self):
        """Test empty message detection."""
        empty_msg = OllamaChatMessage(role=MessageRole.USER, content="")
        whitespace_msg = OllamaChatMessage(role=MessageRole.USER, content="   ")
        normal_msg = OllamaChatMessage(role=MessageRole.USER, content="Hello")

        assert empty_msg.is_empty()
        assert whitespace_msg.is_empty()
        assert not normal_msg.is_empty()

    def test_string_representation(self):
        """Test string representation."""
        msg = OllamaChatMessage(role=MessageRole.USER, content="Hello world")
        str_repr = str(msg)
        assert "user:" in str_repr.lower()
        assert "Hello world" in str_repr

    def test_long_content_truncation(self):
        """Test long content truncation in string representation."""
        long_content = "x" * 150
        msg = OllamaChatMessage(role=MessageRole.USER, content=long_content)
        str_repr = str(msg)
        assert "..." in str_repr
        assert len(str_repr) < len(long_content) + 20  # Should be truncated


class TestOllamaChatRequest:
    """Test OllamaChatRequest with enhanced validation."""

    def test_basic_creation(self):
        """Test basic request creation."""
        messages = [OllamaChatMessage(role=MessageRole.USER, content="Hello")]
        req = OllamaChatRequest(model="test-model", messages=messages)
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.stream is False

    def test_message_count(self):
        """Test message count property."""
        messages = [
            OllamaChatMessage(role=MessageRole.USER, content="Hello"),
            OllamaChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        req = OllamaChatRequest(model="test", messages=messages)
        assert req.message_count == 2

    def test_total_content_length(self):
        """Test total content length calculation."""
        messages = [
            OllamaChatMessage(role=MessageRole.USER, content="Hello"),  # 5 chars
            OllamaChatMessage(role=MessageRole.ASSISTANT, content="Hi"),  # 2 chars
        ]
        req = OllamaChatRequest(model="test", messages=messages)
        assert req.total_content_length == 7

    def test_get_messages_by_role(self):
        """Test filtering messages by role."""
        messages = [
            OllamaChatMessage(role=MessageRole.SYSTEM, content="System"),
            OllamaChatMessage(role=MessageRole.USER, content="User1"),
            OllamaChatMessage(role=MessageRole.ASSISTANT, content="Assistant"),
            OllamaChatMessage(role=MessageRole.USER, content="User2"),
        ]
        req = OllamaChatRequest(model="test", messages=messages)

        user_messages = req.get_messages_by_role(MessageRole.USER)
        assert len(user_messages) == 2
        assert all(msg.role == MessageRole.USER for msg in user_messages)

        system_messages = req.get_messages_by_role(MessageRole.SYSTEM)
        assert len(system_messages) == 1
        assert system_messages[0].content == "System"

    def test_validate_conversation_flow(self):
        """Test conversation flow validation."""
        # Valid: single system message
        req1 = OllamaChatRequest(
            model="test",
            messages=[OllamaChatMessage(role=MessageRole.SYSTEM, content="System")],
        )
        assert req1.validate_conversation_flow()

        # Valid: system followed by user
        req2 = OllamaChatRequest(
            model="test",
            messages=[
                OllamaChatMessage(role=MessageRole.SYSTEM, content="System"),
                OllamaChatMessage(role=MessageRole.USER, content="User"),
            ],
        )
        assert req2.validate_conversation_flow()

        # Valid: starting with user
        req3 = OllamaChatRequest(
            model="test",
            messages=[OllamaChatMessage(role=MessageRole.USER, content="User")],
        )
        assert req3.validate_conversation_flow()

        # Invalid: starting with assistant
        req4 = OllamaChatRequest(
            model="test",
            messages=[
                OllamaChatMessage(role=MessageRole.ASSISTANT, content="Assistant")
            ],
        )
        assert not req4.validate_conversation_flow()

    def test_get_conversation_summary(self):
        """Test conversation summary generation."""
        messages = [
            OllamaChatMessage(role=MessageRole.SYSTEM, content="System"),
            OllamaChatMessage(role=MessageRole.USER, content="User1"),
            OllamaChatMessage(role=MessageRole.ASSISTANT, content="Assistant"),
            OllamaChatMessage(role=MessageRole.USER, content="User2"),
        ]
        req = OllamaChatRequest(model="test", messages=messages)
        summary = req.get_conversation_summary()

        assert summary["total_messages"] == 4
        assert summary["role_distribution"]["system"] == 1
        assert summary["role_distribution"]["user"] == 2
        assert summary["role_distribution"]["assistant"] == 1
        assert summary["has_system_context"] is True
        assert summary["is_valid_flow"] is True


class TestOllamaShowRequest:
    """Test OllamaShowRequest with pattern matching validation."""

    def test_valid_name_field(self):
        """Test request with valid name field."""
        req = OllamaShowRequest(name="test-model")
        assert req.name == "test-model"

    def test_valid_model_field(self):
        """Test request with valid model field."""
        req = OllamaShowRequest(model="test-model")
        assert req.model == "test-model"

    def test_both_fields_present(self):
        """Test request with both name and model fields."""
        req = OllamaShowRequest(name="test-name", model="test-model")
        assert req.name == "test-name"
        assert req.model == "test-model"

    def test_empty_request_body(self):
        """Test validation with empty request body."""
        with pytest.raises(ValueError, match="Request body cannot be empty"):
            OllamaShowRequest.model_validate(None)

    def test_invalid_data_type(self):
        """Test validation with invalid data type."""
        with pytest.raises(ValueError, match="Invalid request data type"):
            OllamaShowRequest.model_validate("invalid")


if __name__ == "__main__":
    pytest.main([__file__])
