"""Utility functions for model name processing and mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelMapping:
    """Container for model mapping information."""

    ollama_name: str
    openrouter_id: str
    display_name: str | None = None
    context_length: int | None = None

    @classmethod
    def from_openrouter_model(cls, model: dict[str, Any]) -> ModelMapping:
        """Create mapping from OpenRouter model data."""
        ollama_name = openrouter_id_to_ollama_name(model["id"])
        return cls(
            ollama_name=ollama_name,
            openrouter_id=model["id"],
            display_name=model.get("name"),
            context_length=model.get("context_length"),
        )


def openrouter_id_to_ollama_name(model_id: str) -> str:
    """Convert OpenRouter model ID to Ollama-compatible name."""
    # Remove vendor prefix (e.g., 'openai/') and append ':latest'
    name = model_id.split("/")[-1]
    # Clean up the name to be more Ollama-like
    name = re.sub(r"[^a-zA-Z0-9\-_.]", "-", name)
    return f"{name}:latest"


def build_ollama_to_openrouter_map(models: list[dict[str, Any]]) -> dict[str, str]:
    """Build mapping from Ollama names to OpenRouter IDs."""
    mapping = {}
    for model in models:
        ollama_name = openrouter_id_to_ollama_name(model["id"])
        mapping[ollama_name] = model["id"]
        logger.debug(
            "Added model mapping", ollama_name=ollama_name, openrouter_id=model["id"]
        )

    logger.info("Built Ollama to OpenRouter mapping", count=len(mapping))
    return mapping


def build_openrouter_to_ollama_map(models: list[dict[str, Any]]) -> dict[str, str]:
    """Build mapping from OpenRouter IDs to Ollama names."""
    mapping = {}
    for model in models:
        ollama_name = openrouter_id_to_ollama_name(model["id"])
        mapping[model["id"]] = ollama_name

    logger.info("Built OpenRouter to Ollama mapping", count=len(mapping))
    return mapping


def filter_models(
    models: list[dict[str, Any]], filter_set: set[str]
) -> list[dict[str, Any]]:
    """Filter models based on allowed set."""
    if not filter_set:
        logger.debug("No model filter applied")
        return models

    filtered = []
    for model in models:
        ollama_name = openrouter_id_to_ollama_name(model["id"])
        if ollama_name in filter_set:
            filtered.append(model)

    logger.info(
        "Applied model filter",
        original_count=len(models),
        filtered_count=len(filtered),
        filter_size=len(filter_set),
    )
    return filtered


def resolve_model_name(
    requested_name: str | None, ollama_map: dict[str, str]
) -> tuple[str | None, str | None]:
    """
    Resolve a potentially short/aliased model name to the full Ollama name and OpenRouter ID.

    Uses pattern matching for better readability and performance.

    Args:
        requested_name: The model name from the user request.
        ollama_map: The mapping from full Ollama names to OpenRouter IDs.

    Returns:
        A tuple containing (resolved_ollama_name, openrouter_id), or (None, None) if no match.
    """
    match requested_name:
        case None:
            return None, None
        case name if name in ollama_map:
            # Exact match
            return name, ollama_map[name]
        case name if ":" not in name:
            # Prefix match for names without version specifier
            for ollama_name, openrouter_id in ollama_map.items():
                if ollama_name.startswith(f"{name}:"):
                    logger.debug(
                        "Resolved model by prefix", requested=name, resolved=ollama_name
                    )
                    return ollama_name, openrouter_id
            return None, None
        case _:
            # No match found
            logger.debug("No model match found", requested_name=requested_name)
            return None, None
