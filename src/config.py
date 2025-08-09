"""Configuration management for ollama-proxy."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Self, TypeAlias

from pydantic import ConfigDict, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Type aliases for better code clarity
HostAddress: TypeAlias = str
PortNumber: TypeAlias = int
APIKey: TypeAlias = str
FilePath: TypeAlias = str | Path


class LogLevel(StrEnum):
    """Supported log levels with validation and pattern matching."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, value: str) -> LogLevel:
        """Create LogLevel from string with case-insensitive matching using pattern matching."""
        normalized = value.upper().strip()
        match normalized:
            case "DEBUG" | "DBG" | "D":
                return cls.DEBUG
            case "INFO" | "INFORMATION" | "I":
                return cls.INFO
            case "WARNING" | "WARN" | "W":
                return cls.WARNING
            case "ERROR" | "ERR" | "E":
                return cls.ERROR
            case "CRITICAL" | "CRIT" | "FATAL" | "C" | "F":
                return cls.CRITICAL
            case _:
                raise ValueError(
                    f"Invalid log level: {value}. Valid levels: {list(cls)}"
                )

    @property
    def numeric_level(self) -> int:
        """Get numeric level for comparison (lower is more verbose)."""
        match self:
            case LogLevel.DEBUG:
                return 10
            case LogLevel.INFO:
                return 20
            case LogLevel.WARNING:
                return 30
            case LogLevel.ERROR:
                return 40
            case LogLevel.CRITICAL:
                return 50
            case _:
                return 20  # Default to INFO level

    def is_enabled_for(self, target_level: LogLevel) -> bool:
        """Check if this level would be logged at the target level."""
        return self.numeric_level >= target_level.numeric_level


class Environment(StrEnum):
    """Application environment types with enhanced pattern matching support."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def from_string(cls, value: str) -> Environment:
        """Create Environment from string with flexible matching."""
        normalized = value.lower().strip()
        match normalized:
            case "development" | "dev" | "local":
                return cls.DEVELOPMENT
            case "staging" | "stage" | "test" | "testing":
                return cls.STAGING
            case "production" | "prod" | "live":
                return cls.PRODUCTION
            case _:
                # Default to development for unknown environments
                return cls.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if this is production environment."""
        return self == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if this is development environment."""
        return self == Environment.DEVELOPMENT

    @property
    def is_staging(self) -> bool:
        """Check if this is staging environment."""
        return self == Environment.STAGING

    @property
    def debug_enabled(self) -> bool:
        """Check if debug should be enabled by default for this environment."""
        match self:
            case Environment.DEVELOPMENT:
                return True
            case Environment.STAGING | Environment.PRODUCTION:
                return False
            case _:
                return False

    @property
    def log_level_default(self) -> LogLevel:
        """Get default log level for this environment."""
        match self:
            case Environment.DEVELOPMENT:
                return LogLevel.DEBUG
            case Environment.STAGING:
                return LogLevel.INFO
            case Environment.PRODUCTION:
                return LogLevel.WARNING
            case _:
                return LogLevel.INFO


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelFilter:
    """Model filter configuration with enhanced validation, caching, and pattern matching."""

    path: Path | None = None
    models: frozenset[str] = field(default_factory=frozenset)
    patterns: frozenset[str] = field(default_factory=frozenset)
    exclude_patterns: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_file(cls, path: FilePath) -> ModelFilter:
        """Create ModelFilter from file with enhanced error handling and pattern support."""
        filter_path = Path(path)
        if not filter_path.exists():
            raise FileNotFoundError(
                f"Model filter file not found: {filter_path}")

        try:
            models = set()
            patterns = set()
            exclude_patterns = set()

            with filter_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Pattern matching for different filter types
                    match line:
                        case line if line.startswith("!"):
                            # Exclusion pattern
                            exclude_patterns.add(line[1:])
                        case line if "*" in line or "?" in line:
                            # Wildcard pattern
                            patterns.add(line)
                        case _:
                            # Exact model name
                            models.add(line)

            return cls(
                path=filter_path,
                models=frozenset(models),
                patterns=frozenset(patterns),
                exclude_patterns=frozenset(exclude_patterns),
            )
        except Exception as e:
            raise ValueError(
                f"Could not load filter file {filter_path}: {e}") from e

    @classmethod
    def empty(cls) -> ModelFilter:
        """Create empty filter."""
        return cls()

    def is_allowed(self, model_name: str) -> bool:
        """Check if model is allowed by filter using pattern matching."""
        if not self.models and not self.patterns:
            return True  # Empty filter allows all

        # Check exclusion patterns first
        for exclude_pattern in self.exclude_patterns:
            if self._matches_pattern(model_name, exclude_pattern):
                return False

        # Check exact matches
        if model_name in self.models:
            return True

        # Check wildcard patterns
        for pattern in self.patterns:
            if self._matches_pattern(model_name, pattern):
                return True

        # If we have filters but no match, deny
        return not (self.models or self.patterns)

    def _matches_pattern(self, model_name: str, pattern: str) -> bool:
        """Check if model name matches a wildcard pattern."""
        import fnmatch

        return fnmatch.fnmatch(model_name, pattern)

    def get_statistics(self) -> dict[str, Any]:
        """Get filter statistics."""
        return {
            "exact_models": len(self.models),
            "patterns": len(self.patterns),
            "exclude_patterns": len(self.exclude_patterns),
            "total_rules": len(self.models)
            + len(self.patterns)
            + len(self.exclude_patterns),
            "has_filters": bool(self.models or self.patterns),
            "path": str(self.path) if self.path else None,
        }

    def __len__(self) -> int:
        """Get number of allowed models (exact matches only)."""
        return len(self.models)

    def __bool__(self) -> bool:
        """Check if filter has any rules."""
        return bool(self.models or self.patterns or self.exclude_patterns)


class Settings(BaseSettings):
    """Enhanced application settings with modern validation and computed fields."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="forbid",
        # Support for nested environment variables
        env_nested_delimiter="__",
    )

    # Core settings
    openrouter_api_key: str = Field(
        ..., description="OpenRouter API key", min_length=1, alias="OPENROUTER_API_KEY"
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
        pattern=r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$|^localhost$|^0\.0\.0\.0$",
    )
    port: int = Field(default=11434, ge=1, le=65535,
                      description="Port to listen on")

    # Environment and deployment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )

    # Model filtering
    models_filter_path: str | None = Field(
        default="models-filter.txt", description="Path to model filter file"
    )

    # Logging configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        default="%(asctime)s %(levelname)s %(name)s %(message)s",
        description="Log format string",
    )

    # OpenRouter specific settings
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
        pattern=r"^https?://.+",
    )
    openrouter_timeout: int = Field(
        default=300, ge=1, le=3600, description="OpenRouter request timeout in seconds"
    )

    # Performance settings
    max_concurrent_requests: int = Field(
        default=100, ge=1, le=1000, description="Maximum concurrent requests"
    )

    # Development settings
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(
        default=False, description="Enable auto-reload for development"
    )

    # Health check settings
    health_check_interval: int = Field(
        default=60, ge=10, description="Health check interval in seconds"
    )

    @computed_field
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.is_production

    @computed_field
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.is_development

    @computed_field
    def log_level_name(self) -> str:
        """Get log level as string."""
        return self.log_level.value

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> APIKey:
        """Validate API key format and content using pattern matching."""
        if not v or not v.strip():
            raise ValueError("OpenRouter API key cannot be empty")

        stripped = v.strip()

        # Allow test keys for testing environments
        if stripped.startswith("test-") or stripped == "test-key":
            return stripped

        # Pattern match on API key characteristics for production keys
        match stripped:
            case key if len(key) < 10:
                raise ValueError("OpenRouter API key appears to be too short")
            case key if key.startswith("sk-"):
                # OpenAI-style key format
                if len(key) < 20:
                    raise ValueError(
                        "OpenRouter API key appears to be too short")
            case key if key.startswith("or-"):
                # OpenRouter-specific key format
                if len(key) < 15:
                    raise ValueError(
                        "OpenRouter API key appears to be too short")
            case key if not key.replace("-", "").replace("_", "").isalnum():
                raise ValueError(
                    "OpenRouter API key contains invalid characters")
            case _:
                # Generic validation for other formats
                if len(stripped) < 10:
                    raise ValueError(
                        "OpenRouter API key appears to be too short")

        return stripped

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str | LogLevel) -> LogLevel:
        """Validate and convert log level using enhanced pattern matching."""
        match v:
            case LogLevel() as level:
                return level
            case str() as level_str:
                return LogLevel.from_string(level_str)
            case _:
                raise ValueError(f"Invalid log level type: {type(v)}")

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str | Environment) -> Environment:
        """Validate and convert environment using pattern matching."""
        match v:
            case Environment() as env:
                return env
            case str() as env_str:
                return Environment.from_string(env_str)
            case _:
                raise ValueError(f"Invalid environment type: {type(v)}")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> HostAddress:
        """Validate host address format."""
        if not v or not v.strip():
            raise ValueError("Host cannot be empty")

        stripped = v.strip()
        match stripped:
            case "localhost" | "0.0.0.0":
                return stripped
            case host if host.count(".") == 3:
                # Basic IPv4 validation
                parts = host.split(".")
                if all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
                    return stripped
                raise ValueError(f"Invalid IPv4 address: {host}")
            case _:
                # Allow other formats (hostnames, IPv6, etc.)
                return stripped

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> PortNumber:
        """Validate port number with enhanced checks."""
        match v:
            case port if 1 <= port <= 65535:
                return port
            case port if port <= 0:
                raise ValueError("Port number must be positive")
            case port if port > 65535:
                raise ValueError("Port number must be <= 65535")
            case _:
                raise ValueError(f"Invalid port number: {v}")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    # BaseSettings will automatically load from environment variables and .env file
    return Settings()  # type: ignore[call-arg]


def load_model_filter(settings: Settings) -> ModelFilter:
    """Load model filter from settings with enhanced error handling."""
    if not settings.models_filter_path:
        return ModelFilter.empty()

    filter_path = Path(settings.models_filter_path)
    if not filter_path.exists():
        # Return empty filter if file doesn't exist (non-fatal)
        return ModelFilter.empty()

    try:
        return ModelFilter.from_file(filter_path)
    except (FileNotFoundError, ValueError) as e:
        # Log warning but don't fail startup
        import structlog

        logger = structlog.get_logger(__name__)
        logger.warning(
            "Failed to load model filter, using empty filter",
            path=str(filter_path),
            error=str(e),
        )
        return ModelFilter.empty()


@dataclass(slots=True)
class AppState:
    """Enhanced application state container with modern patterns."""

    settings: Settings
    model_filter: ModelFilter = field(init=False)
    all_models: list[dict[str, Any]] = field(default_factory=list)
    ollama_to_openrouter_map: dict[str, str] = field(default_factory=dict)
    openrouter_to_ollama_map: dict[str, str] = field(default_factory=dict)
    openrouter_client: Any = field(default=None)

    # Health and monitoring
    startup_time: float = field(
        default_factory=lambda: __import__("time").time())
    last_model_refresh: float = field(default=0.0)
    request_count: int = field(default=0)
    error_count: int = field(default=0)

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass creation."""
        self.model_filter = load_model_filter(self.settings)

    @classmethod
    def create(cls, settings: Settings) -> Self:
        """Create AppState with proper initialization."""
        return cls(settings=settings)

    def update_models(
        self,
        models: list[dict[str, Any]],
        ollama_map: dict[str, str],
        openrouter_map: dict[str, str],
    ) -> None:
        """Update model mappings with timestamp tracking."""
        # Store the raw model data from OpenRouter API
        self.all_models = models

        # Update bidirectional mappings for name resolution
        # ollama_map: "gpt-4:latest" -> "openai/gpt-4"
        self.ollama_to_openrouter_map = ollama_map
        # openrouter_map: "openai/gpt-4" -> "gpt-4:latest"
        self.openrouter_to_ollama_map = openrouter_map

        # Track when models were last refreshed for cache invalidation
        self.last_model_refresh = __import__("time").time()

    def increment_request_count(self) -> None:
        """Increment request counter for monitoring."""
        self.request_count += 1

    def increment_error_count(self) -> None:
        """Increment error counter for monitoring."""
        self.error_count += 1

    @property
    def uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return __import__("time").time() - self.startup_time

    @property
    def model_count(self) -> int:
        """Get total number of available models."""
        return len(self.all_models)

    @property
    def filtered_model_count(self) -> int:
        """Get number of models after filtering."""
        if not self.model_filter:
            return self.model_count
        return len(
            [
                m
                for m in self.all_models
                if self.model_filter.is_allowed(m.get("id", ""))
            ]
        )

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "status": "healthy" if self.openrouter_client else "unhealthy",
            "uptime_seconds": self.uptime_seconds,
            "model_count": self.model_count,
            "filtered_model_count": self.filtered_model_count,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "last_model_refresh": self.last_model_refresh,
            "environment": self.settings.environment.value,
        }
