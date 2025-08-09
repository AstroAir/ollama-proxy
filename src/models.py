from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Any, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Type Aliases for Better Code Clarity ---

ModelID: TypeAlias = str
RequestID: TypeAlias = str
Content: TypeAlias = str
TokenCount: TypeAlias = int

# --- Base Models with Modern Configuration ---


class BaseOllamaModel(BaseModel):
    """Base model with modern Pydantic v2 configuration and enhanced features."""

    model_config = ConfigDict(
        # Enable validation on assignment
        validate_assignment=True,
        # Use enum values instead of names
        use_enum_values=True,
        # Forbid extra fields for strict validation
        extra="forbid",
        # Validate default values
        validate_default=True,
        # Populate by name for API compatibility
        populate_by_name=True,
        # Enable frozen for immutable models where appropriate
        frozen=False,
        # Use slots for better performance
        arbitrary_types_allowed=True,
        # Validate return values
        validate_return=True,
    )


# --- Enums for better type safety ---


class MessageRole(StrEnum):
    """Message roles for chat conversations with enhanced validation and pattern matching support."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    @classmethod
    def from_string(cls, value: str) -> MessageRole:
        """Create MessageRole from string with validation using pattern matching."""
        normalized = value.lower().strip()
        match normalized:
            case "system" | "sys":
                return cls.SYSTEM
            case "user" | "human":
                return cls.USER
            case "assistant" | "ai" | "bot":
                return cls.ASSISTANT
            case "tool" | "function":
                return cls.TOOL
            case _:
                raise ValueError(
                    f"Invalid message role: {value}. Valid roles: {list(cls)}"
                )

    def is_user_role(self) -> bool:
        """Check if this is a user-initiated role using pattern matching."""
        match self:
            case MessageRole.USER | MessageRole.SYSTEM:
                return True
            case _:
                return False

    def is_assistant_role(self) -> bool:
        """Check if this is an assistant role."""
        return self == MessageRole.ASSISTANT

    def get_display_name(self) -> str:
        """Get human-readable display name for the role."""
        match self:
            case MessageRole.SYSTEM:
                return "System"
            case MessageRole.USER:
                return "User"
            case MessageRole.ASSISTANT:
                return "Assistant"
            case MessageRole.TOOL:
                return "Tool"
            case _:
                return self.value.title()


class ResponseFormat(StrEnum):
    """Response format types."""

    JSON = "json"


# --- Ollama API Schemas ---


class OllamaChatMessage(BaseOllamaModel):
    """Chat message with enhanced validation and modern features."""

    role: MessageRole
    content: str
    images: list[str] | None = Field(
        None, description="Base64 encoded images (future feature)"
    )

    def __str__(self) -> str:
        """String representation for logging."""
        return f"{self.role}: {self.content[:100]}{'...' if len(self.content) > 100 else ''}"

    @property
    def content_length(self) -> int:
        """Get content length for monitoring."""
        return len(self.content)

    def is_empty(self) -> bool:
        """Check if message has no meaningful content."""
        return not self.content.strip()


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestMetadata:
    """Metadata for tracking requests with modern dataclass features and enhanced functionality."""

    request_id: RequestID = field(
        default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}"
    )
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    user_agent: str | None = None
    client_ip: str | None = None
    correlation_id: str | None = None
    session_id: str | None = None

    def age_seconds(self) -> float:
        """Get request age in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def is_expired(self, max_age_seconds: float = 300.0) -> bool:
        """Check if request has expired based on age."""
        return self.age_seconds() > max_age_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "age_seconds": self.age_seconds(),
            "user_agent": self.user_agent,
            "client_ip": self.client_ip,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
        }


class OllamaChatRequest(BaseOllamaModel):
    """Enhanced chat request with validation, metadata support, and modern Python features."""

    model: ModelID = Field(..., min_length=1, description="Model identifier")
    messages: list[OllamaChatMessage] = Field(
        ..., min_length=1, description="Conversation messages"
    )
    format: ResponseFormat | None = Field(
        None, description="Response format specification"
    )
    options: dict[str, Any] | None = Field(
        None, description="Model-specific options like temperature"
    )
    template: str | None = Field(
        None, description="Custom prompt template (proxy may ignore)"
    )
    stream: bool = Field(False, description="Enable streaming response")
    keep_alive: str | None = Field(
        None, description="Keep-alive duration (proxy ignores)"
    )

    # Internal metadata (not part of API)
    metadata: RequestMetadata | None = Field(None, exclude=True)

    @property
    def message_count(self) -> TokenCount:
        """Get total number of messages."""
        return len(self.messages)

    @property
    def total_content_length(self) -> TokenCount:
        """Get total content length across all messages."""
        return sum(msg.content_length for msg in self.messages)

    def get_messages_by_role(self, role: MessageRole) -> list[OllamaChatMessage]:
        """Get all messages of a specific role using pattern matching."""
        return [msg for msg in self.messages if msg.role == role]

    def get_system_messages(self) -> list[OllamaChatMessage]:
        """Get all system messages."""
        return self.get_messages_by_role(MessageRole.SYSTEM)

    def get_user_messages(self) -> list[OllamaChatMessage]:
        """Get all user messages."""
        return self.get_messages_by_role(MessageRole.USER)

    def get_assistant_messages(self) -> list[OllamaChatMessage]:
        """Get all assistant messages."""
        return self.get_messages_by_role(MessageRole.ASSISTANT)

    def validate_conversation_flow(self) -> bool:
        """Validate that the conversation follows proper flow patterns."""
        if not self.messages:
            return False

        # Pattern match on conversation structure
        match (self.messages[0].role, len(self.messages)):
            case (MessageRole.SYSTEM, 1):
                return True  # Single system message is valid
            case (MessageRole.SYSTEM, count) if count > 1:
                # System message should be followed by user messages
                return self.messages[1].role == MessageRole.USER
            case (MessageRole.USER, _):
                return True  # Starting with user message is valid
            case _:
                return False  # Other patterns are invalid

    def get_conversation_summary(self) -> dict[str, Any]:
        """Get a summary of the conversation structure."""
        role_counts = {}
        for role in MessageRole:
            role_counts[role.value] = len(self.get_messages_by_role(role))

        return {
            "total_messages": self.message_count,
            "total_content_length": self.total_content_length,
            "role_distribution": role_counts,
            "has_system_context": bool(self.get_system_messages()),
            "is_valid_flow": self.validate_conversation_flow(),
        }


class OllamaChatResponseDelta(BaseModel):
    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT
    content: str


class OllamaChatStreamResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaChatResponseDelta | None = None  # Null in final 'done' message
    done: bool
    # Ollama includes more stats in the final message, omitted here


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaChatMessage  # Role is assistant here
    done: bool = True
    # Add missing stats fields (synthesized)
    total_duration: int | None = 0
    load_duration: int | None = 0
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = 0
    eval_count: int | None = 0
    eval_duration: int | None = 0
    # Ollama includes stats here, omitted


class OllamaTagDetails(BaseModel):
    # Placeholder, as we won't have real details from OpenRouter
    format: str | None = None
    family: str | None = None
    families: list[str] | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


class OllamaTagModel(BaseModel):
    name: str
    modified_at: str
    size: int | None = None
    digest: str | None = None
    details: OllamaTagDetails


class OllamaTagsResponse(BaseModel):
    models: list[OllamaTagModel]


# --- OpenRouter Schemas (Simplified, for internal use) ---


class OpenRouterModel(BaseModel):
    id: str
    name: str
    description: str | None = None
    pricing: dict[str, Any]
    context_length: int | None = None
    # Add other fields if needed


class OpenRouterModelsResponse(BaseModel):
    data: list[OpenRouterModel]


# --- /api/generate Schemas ---


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    suffix: str | None = None
    images: list[str] | None = None
    format: str | None = None
    options: dict | None = None
    system: str | None = None
    template: str | None = None
    stream: bool | None = True
    raw: bool | None = False
    keep_alive: str | None = None
    # Deprecated, but included for compatibility
    context: list[int] | None = None


class OllamaGenerateStreamResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = False
    # Include final stats when done=True
    context: list[int] | None = None  # Deprecated
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int | None = None
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = True
    context: list[int] | None = None  # Deprecated
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int | None = None
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


# --- /api/show Schemas ---


class OllamaShowRequest(BaseModel):
    # Allow 'name' to be optional, defaulting to None if missing
    name: str | None = None
    # Keep model field for backward compatibility or direct use if provided
    model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_name_or_model(cls, data: Any) -> Any:
        """Validate request data using modern pattern matching."""
        match data:
            case dict() as request_data:
                # Use pattern matching for validation logic
                match (request_data.get("name"), request_data.get("model")):
                    case (None, None):
                        # Neither field present - let Pydantic handle the validation
                        pass
                    case (str(name), _) if name.strip():
                        # Name is present and valid - prioritize it
                        pass
                    case (None, str(model)) if model.strip():
                        # Only model is present and valid
                        pass
                    case _:
                        # Invalid combination or empty strings
                        raise ValueError(
                            "Request must contain either 'name' or 'model' field with valid content."
                        )
                return request_data
            case None:
                raise ValueError("Request body cannot be empty.")
            case _:
                raise ValueError(
                    f"Invalid request data type: {type(data).__name__}")
        return data


class OllamaShowDetails(BaseModel):
    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: list[str] = []
    parameter_size: str = ""
    quantization_level: str = ""


class OllamaShowTensor(BaseModel):
    name: str = ""
    type: str = ""
    shape: list[int] = []


class OllamaShowModelInfo(BaseModel):
    # These fields are highly specific to Ollama's internal model representation
    # and cannot be accurately fetched from OpenRouter. Return empty dict.
    pass


class OllamaShowResponse(BaseModel):
    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: OllamaShowDetails = OllamaShowDetails()
    model_info: dict = {}
    tensors: list[OllamaShowTensor] = []
    # projector REMOVED to match Ollama
    # modified_at is not present in Ollama's /api/show, so remove it

    model_config = {
        "protected_namespaces": (),
    }


# --- /api/ps Schemas --- (List Running Models)


class OllamaPsModel(BaseModel):
    name: str
    model: str
    size: int | None = None
    digest: str | None = None
    details: OllamaShowDetails
    expires_at: str | None = None
    size_vram: int | None = None


class OllamaPsResponse(BaseModel):
    models: list[OllamaPsModel]


# --- /api/embed Schemas --- (Newer Embeddings Endpoint)


class OllamaEmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    truncate: bool | None = True  # Note: Currently ignored by proxy logic
    options: dict | None = None  # Note: Ignored by proxy
    keep_alive: str | None = None  # Note: Ignored by proxy


class OllamaEmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]
    # Ollama includes stats like prompt_eval_count, omitted here


# --- /api/embeddings Schemas --- (Older Embeddings Endpoint)


class OllamaEmbeddingsRequest(BaseModel):
    model: str
    prompt: str  # Equivalent to single input in /api/embed
    options: dict | None = None  # Note: Ignored by proxy
    keep_alive: str | None = None  # Note: Ignored by proxy


class OllamaEmbeddingsResponse(BaseModel):
    # Corresponds to the first embedding in /api/embed response
    embedding: list[float]
