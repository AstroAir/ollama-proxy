"""Enhanced configuration system for multi-provider support.

This module extends the existing configuration system to support multiple
AI providers, routing rules, and provider-specific settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import Environment, LogLevel, Settings
from .providers.base import ProviderCapability, ProviderConfig, ProviderType


class RoutingStrategy(StrEnum):
    """Strategies for routing requests to providers."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    COST_OPTIMIZED = "cost_optimized"
    CAPABILITY_BASED = "capability_based"
    MANUAL = "manual"


class FallbackStrategy(StrEnum):
    """Strategies for handling provider failures."""

    NONE = "none"
    NEXT_AVAILABLE = "next_available"
    RETRY_SAME = "retry_same"
    BEST_ALTERNATIVE = "best_alternative"


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """Settings for a specific provider."""

    enabled: bool = True
    api_key: str = ""
    base_url: str = ""
    timeout: int = 300
    max_retries: int = 3
    max_concurrent_requests: int = 100
    priority: int = 1  # Lower number = higher priority
    cost_per_token: float = 0.0  # For cost optimization
    custom_headers: Dict[str, str] = field(default_factory=dict)
    model_mapping: Dict[str, str] = field(default_factory=dict)

    def to_provider_config(self, provider_type: ProviderType) -> ProviderConfig:
        """Convert to ProviderConfig."""
        return ProviderConfig(
            provider_type=provider_type,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_concurrent_requests=self.max_concurrent_requests,
            custom_headers=self.custom_headers,
            model_mapping=self.model_mapping,
        )


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """Rule for routing requests to specific providers."""

    name: str
    condition: str  # e.g., "model.startswith('gpt')"
    provider_type: ProviderType
    priority: int = 1
    enabled: bool = True


class MultiProviderSettings(BaseSettings):
    """Enhanced settings with multi-provider support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="forbid",
        env_nested_delimiter="__",
    )

    # Inherit base settings
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=11434, ge=1, le=65535, description="Port to listen on")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Multi-provider settings
    routing_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.CAPABILITY_BASED,
        description="Strategy for routing requests to providers"
    )
    fallback_strategy: FallbackStrategy = Field(
        default=FallbackStrategy.NEXT_AVAILABLE,
        description="Strategy for handling provider failures"
    )
    enable_load_balancing: bool = Field(
        default=True,
        description="Enable load balancing across providers"
    )
    health_check_interval: int = Field(
        default=60,
        ge=10,
        description="Health check interval in seconds"
    )

    # Provider configurations
    openrouter_enabled: bool = Field(default=True, description="Enable OpenRouter provider")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    openrouter_timeout: int = Field(default=300, description="OpenRouter timeout")
    openrouter_priority: int = Field(default=1, description="OpenRouter priority")

    openai_enabled: bool = Field(default=False, description="Enable OpenAI provider")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    openai_timeout: int = Field(default=300, description="OpenAI timeout")
    openai_priority: int = Field(default=2, description="OpenAI priority")

    anthropic_enabled: bool = Field(default=False, description="Enable Anthropic provider")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic API base URL"
    )
    anthropic_timeout: int = Field(default=300, description="Anthropic timeout")
    anthropic_priority: int = Field(default=3, description="Anthropic priority")

    google_enabled: bool = Field(default=False, description="Enable Google provider")
    google_api_key: str = Field(default="", description="Google API key")
    google_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Google API base URL"
    )
    google_timeout: int = Field(default=300, description="Google timeout")
    google_priority: int = Field(default=4, description="Google priority")

    # Azure OpenAI provider settings
    azure_enabled: bool = Field(default=False, description="Enable Azure OpenAI provider")
    azure_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_base_url: str = Field(default="", description="Azure OpenAI endpoint URL")
    azure_timeout: int = Field(default=300, description="Azure timeout")
    azure_priority: int = Field(default=5, description="Azure priority")

    # AWS Bedrock provider settings
    aws_bedrock_enabled: bool = Field(default=False, description="Enable AWS Bedrock provider")
    aws_bedrock_access_key: str = Field(default="", description="AWS Access Key ID")
    aws_bedrock_secret_key: str = Field(default="", description="AWS Secret Access Key")
    aws_bedrock_region: str = Field(default="us-east-1", description="AWS region")
    aws_bedrock_session_token: str = Field(default="", description="AWS session token (optional)")
    aws_bedrock_timeout: int = Field(default=300, description="AWS Bedrock timeout")
    aws_bedrock_priority: int = Field(default=6, description="AWS Bedrock priority")

    # Local Ollama provider settings
    ollama_enabled: bool = Field(default=False, description="Enable local Ollama provider")
    ollama_api_key: str = Field(default="", description="Ollama API key (optional)")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_timeout: int = Field(default=300, description="Ollama timeout")
    ollama_priority: int = Field(default=7, description="Ollama priority")

    # Model filtering and routing
    models_filter_path: Optional[str] = Field(
        default="models-filter.txt",
        description="Path to model filter file"
    )
    routing_rules_path: Optional[str] = Field(
        default="routing-rules.json",
        description="Path to routing rules file"
    )

    def get_provider_settings(self) -> Dict[ProviderType, ProviderSettings]:
        """Get provider settings for all enabled providers."""
        providers = {}

        if self.openrouter_enabled and self.openrouter_api_key:
            providers[ProviderType.OPENROUTER] = ProviderSettings(
                enabled=True,
                api_key=self.openrouter_api_key,
                base_url=self.openrouter_base_url,
                timeout=self.openrouter_timeout,
                priority=self.openrouter_priority,
            )

        if self.openai_enabled and self.openai_api_key:
            providers[ProviderType.OPENAI] = ProviderSettings(
                enabled=True,
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                timeout=self.openai_timeout,
                priority=self.openai_priority,
            )

        if self.anthropic_enabled and self.anthropic_api_key:
            providers[ProviderType.ANTHROPIC] = ProviderSettings(
                enabled=True,
                api_key=self.anthropic_api_key,
                base_url=self.anthropic_base_url,
                timeout=self.anthropic_timeout,
                priority=self.anthropic_priority,
            )

        if self.google_enabled and self.google_api_key:
            providers[ProviderType.GOOGLE] = ProviderSettings(
                enabled=True,
                api_key=self.google_api_key,
                base_url=self.google_base_url,
                timeout=self.google_timeout,
                priority=self.google_priority,
            )

        if self.azure_enabled and self.azure_api_key and self.azure_base_url:
            providers[ProviderType.AZURE] = ProviderSettings(
                enabled=True,
                api_key=self.azure_api_key,
                base_url=self.azure_base_url,
                timeout=self.azure_timeout,
                priority=self.azure_priority,
            )

        if self.aws_bedrock_enabled and self.aws_bedrock_access_key and self.aws_bedrock_secret_key:
            custom_headers = {
                "aws_secret_access_key": self.aws_bedrock_secret_key,
                "aws_region": self.aws_bedrock_region,
            }
            if self.aws_bedrock_session_token:
                custom_headers["aws_session_token"] = self.aws_bedrock_session_token

            providers[ProviderType.AWS_BEDROCK] = ProviderSettings(
                enabled=True,
                api_key=self.aws_bedrock_access_key,
                base_url=f"https://bedrock-runtime.{self.aws_bedrock_region}.amazonaws.com",
                timeout=self.aws_bedrock_timeout,
                priority=self.aws_bedrock_priority,
                custom_headers=custom_headers,
            )

        if self.ollama_enabled:
            providers[ProviderType.OLLAMA] = ProviderSettings(
                enabled=True,
                api_key=self.ollama_api_key,
                base_url=self.ollama_base_url,
                timeout=self.ollama_timeout,
                priority=self.ollama_priority,
            )

        return providers

    def get_enabled_providers(self) -> List[ProviderType]:
        """Get list of enabled provider types."""
        return list(self.get_provider_settings().keys())

    def is_provider_enabled(self, provider_type: ProviderType) -> bool:
        """Check if a specific provider is enabled."""
        return provider_type in self.get_provider_settings()

    @field_validator("routing_strategy", mode="before")
    @classmethod
    def validate_routing_strategy(cls, v: str | RoutingStrategy) -> RoutingStrategy:
        """Validate routing strategy."""
        if isinstance(v, str):
            try:
                return RoutingStrategy(v.lower())
            except ValueError:
                raise ValueError(f"Invalid routing strategy: {v}")
        return v

    @field_validator("fallback_strategy", mode="before")
    @classmethod
    def validate_fallback_strategy(cls, v: str | FallbackStrategy) -> FallbackStrategy:
        """Validate fallback strategy."""
        if isinstance(v, str):
            try:
                return FallbackStrategy(v.lower())
            except ValueError:
                raise ValueError(f"Invalid fallback strategy: {v}")
        return v