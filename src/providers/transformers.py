"""Request/response transformers for standardizing API formats.

This module provides utilities for transforming requests and responses
between Ollama format and various provider formats, ensuring compatibility
while maintaining the flexibility to support provider-specific features.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import structlog

from .base import ProviderType

logger = structlog.get_logger(__name__)


class RequestTransformer:
    """Transforms Ollama requests to provider-specific formats."""

    @staticmethod
    def transform_chat_request(
        ollama_request: Dict[str, Any],
        provider_type: ProviderType,
    ) -> Dict[str, Any]:
        """Transform Ollama chat request to provider format."""
        if provider_type == ProviderType.OPENAI:
            return RequestTransformer._ollama_chat_to_openai(ollama_request)
        elif provider_type == ProviderType.ANTHROPIC:
            return RequestTransformer._ollama_chat_to_anthropic(ollama_request)
        elif provider_type == ProviderType.GOOGLE:
            return RequestTransformer._ollama_chat_to_google(ollama_request)
        elif provider_type == ProviderType.OPENROUTER:
            return RequestTransformer._ollama_chat_to_openrouter(ollama_request)
        else:
            # Default: minimal transformation
            return ollama_request.copy()

    @staticmethod
    def transform_generate_request(
        ollama_request: Dict[str, Any],
        provider_type: ProviderType,
    ) -> Dict[str, Any]:
        """Transform Ollama generate request to provider format."""
        if provider_type == ProviderType.OPENAI:
            return RequestTransformer._ollama_generate_to_openai(ollama_request)
        elif provider_type == ProviderType.ANTHROPIC:
            return RequestTransformer._ollama_generate_to_anthropic(ollama_request)
        elif provider_type == ProviderType.GOOGLE:
            return RequestTransformer._ollama_generate_to_google(ollama_request)
        elif provider_type == ProviderType.OPENROUTER:
            return RequestTransformer._ollama_generate_to_openrouter(ollama_request)
        else:
            # Default: minimal transformation
            return ollama_request.copy()

    @staticmethod
    def transform_embeddings_request(
        ollama_request: Dict[str, Any],
        provider_type: ProviderType,
    ) -> Dict[str, Any]:
        """Transform Ollama embeddings request to provider format."""
        if provider_type == ProviderType.OPENAI:
            return RequestTransformer._ollama_embeddings_to_openai(ollama_request)
        elif provider_type == ProviderType.GOOGLE:
            return RequestTransformer._ollama_embeddings_to_google(ollama_request)
        elif provider_type == ProviderType.OPENROUTER:
            return RequestTransformer._ollama_embeddings_to_openrouter(ollama_request)
        else:
            # Anthropic doesn't support embeddings
            return ollama_request.copy()

    @staticmethod
    def _ollama_chat_to_openai(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama chat to OpenAI format."""
        openai_request = {
            "model": ollama_request.get("model", "gpt-3.5-turbo"),
            "messages": ollama_request.get("messages", []),
        }

        # Map options
        options = ollama_request.get("options", {})
        for ollama_key, openai_key in [
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("top_p", "top_p"),
            ("frequency_penalty", "frequency_penalty"),
            ("presence_penalty", "presence_penalty"),
            ("stop", "stop"),
        ]:
            if options and ollama_key in options:
                openai_request[openai_key] = options[ollama_key]

        # Handle streaming
        if ollama_request.get("stream", False):
            openai_request["stream"] = True

        # Handle response format
        if ollama_request.get("format") == "json":
            openai_request["response_format"] = {"type": "json_object"}

        return openai_request

    @staticmethod
    def _ollama_chat_to_anthropic(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama chat to Anthropic format."""
        messages = ollama_request.get("messages", [])

        # Separate system messages
        system_messages = []
        user_assistant_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg.get("content", ""))
            elif msg.get("role") in ["user", "assistant"]:
                user_assistant_messages.append(msg)

        anthropic_request = {
            "model": ollama_request.get("model", "claude-3-sonnet-20240229"),
            "messages": user_assistant_messages,
            "max_tokens": 4096,  # Required for Anthropic
        }

        # Add system message if present
        if system_messages:
            anthropic_request["system"] = "\n\n".join(system_messages)

        # Map options
        options = ollama_request.get("options", {})
        for ollama_key, anthropic_key in [
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
        ]:
            if ollama_key in options:
                anthropic_request[anthropic_key] = options[ollama_key]

        # Handle stop sequences
        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                anthropic_request["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                anthropic_request["stop_sequences"] = stop

        # Handle streaming
        if ollama_request.get("stream", False):
            anthropic_request["stream"] = True

        return anthropic_request

    @staticmethod
    def _ollama_chat_to_google(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama chat to Google format."""
        messages = ollama_request.get("messages", [])
        contents = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Google doesn't have system role, add as user message
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {content}"}]
                })
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",  # Google uses "model" instead of "assistant"
                    "parts": [{"text": content}]
                })

        google_request: Dict[str, Any] = {
            "contents": contents,
        }

        # Add generation config
        generation_config: Dict[str, Any] = {}
        options = ollama_request.get("options", {})
        for ollama_key, google_key in [
            ("temperature", "temperature"),
            ("max_tokens", "maxOutputTokens"),
            ("top_p", "topP"),
            ("top_k", "topK"),
        ]:
            if ollama_key in options:
                generation_config[google_key] = options[ollama_key]

        # Handle stop sequences
        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                generation_config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                generation_config["stopSequences"] = stop

        if generation_config:
            google_request["generationConfig"] = generation_config

        return google_request

    @staticmethod
    def _ollama_chat_to_openrouter(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama chat to OpenRouter format (similar to OpenAI)."""
        return RequestTransformer._ollama_chat_to_openai(ollama_request)

    @staticmethod
    def _ollama_generate_to_openai(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama generate to OpenAI completions format."""
        openai_request = {
            "model": ollama_request.get("model", "gpt-3.5-turbo-instruct"),
            "prompt": ollama_request.get("prompt", ""),
        }

        # Map options
        options = ollama_request.get("options", {})
        for ollama_key, openai_key in [
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("top_p", "top_p"),
            ("frequency_penalty", "frequency_penalty"),
            ("presence_penalty", "presence_penalty"),
            ("stop", "stop"),
        ]:
            if ollama_key in options:
                openai_request[openai_key] = options[ollama_key]

        if ollama_request.get("stream", False):
            openai_request["stream"] = True

        return openai_request

    @staticmethod
    def _ollama_generate_to_anthropic(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama generate to Anthropic format."""
        prompt = ollama_request.get("prompt", "")
        system = ollama_request.get("system")

        messages = [{"role": "user", "content": prompt}]

        anthropic_request = {
            "model": ollama_request.get("model", "claude-3-sonnet-20240229"),
            "messages": messages,
            "max_tokens": 4096,
        }

        if system:
            anthropic_request["system"] = system

        # Map options
        options = ollama_request.get("options", {})
        for ollama_key, anthropic_key in [
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
        ]:
            if ollama_key in options:
                anthropic_request[anthropic_key] = options[ollama_key]

        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                anthropic_request["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                anthropic_request["stop_sequences"] = stop

        if ollama_request.get("stream", False):
            anthropic_request["stream"] = True

        return anthropic_request

    @staticmethod
    def _ollama_generate_to_google(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama generate to Google format."""
        prompt = ollama_request.get("prompt", "")
        system = ollama_request.get("system")

        contents = []
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {system}"}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        google_request: Dict[str, Any] = {
            "contents": contents,
        }

        # Add generation config
        generation_config: Dict[str, Any] = {}
        options = ollama_request.get("options", {})
        for ollama_key, google_key in [
            ("temperature", "temperature"),
            ("max_tokens", "maxOutputTokens"),
            ("top_p", "topP"),
            ("top_k", "topK"),
        ]:
            if ollama_key in options:
                generation_config[google_key] = options[ollama_key]

        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                generation_config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                generation_config["stopSequences"] = stop

        if generation_config:
            google_request["generationConfig"] = generation_config

        return google_request

    @staticmethod
    def _ollama_generate_to_openrouter(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama generate to OpenRouter format."""
        # Convert to chat format for OpenRouter
        messages = []
        if ollama_request.get("system"):
            messages.append({"role": "system", "content": ollama_request["system"]})
        messages.append({"role": "user", "content": ollama_request.get("prompt", "")})

        openrouter_request = {
            "model": ollama_request.get("model"),
            "messages": messages,
        }

        # Map options
        options = ollama_request.get("options", {})
        if options:
            openrouter_request.update(options)

        if ollama_request.get("stream", False):
            openrouter_request["stream"] = True

        if ollama_request.get("format") == "json":
            openrouter_request["response_format"] = {"type": "json_object"}

        return openrouter_request

    @staticmethod
    def _ollama_embeddings_to_openai(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama embeddings to OpenAI format."""
        return {
            "model": ollama_request.get("model", "text-embedding-ada-002"),
            "input": ollama_request.get("input") or ollama_request.get("prompt", ""),
        }

    @staticmethod
    def _ollama_embeddings_to_google(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama embeddings to Google format."""
        input_text = ollama_request.get("input") or ollama_request.get("prompt", "")
        return {
            "content": {
                "parts": [{"text": input_text}]
            }
        }

    @staticmethod
    def _ollama_embeddings_to_openrouter(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama embeddings to OpenRouter format."""
        return RequestTransformer._ollama_embeddings_to_openai(ollama_request)


class ResponseTransformer:
    """Transforms provider responses to Ollama format."""

    @staticmethod
    def transform_chat_response(
        provider_response: Dict[str, Any],
        provider_type: ProviderType,
    ) -> Dict[str, Any]:
        """Transform provider chat response to Ollama format."""
        if provider_type == ProviderType.OPENAI:
            return ResponseTransformer._openai_chat_to_ollama(provider_response)
        elif provider_type == ProviderType.ANTHROPIC:
            return ResponseTransformer._anthropic_to_ollama(provider_response)
        elif provider_type == ProviderType.GOOGLE:
            return ResponseTransformer._google_to_ollama(provider_response)
        elif provider_type == ProviderType.OPENROUTER:
            return ResponseTransformer._openrouter_to_ollama(provider_response)
        else:
            # Default: minimal transformation
            return provider_response.copy()

    @staticmethod
    def transform_generate_response(
        provider_response: Dict[str, Any],
        provider_type: ProviderType,
    ) -> Dict[str, Any]:
        """Transform provider generate response to Ollama format."""
        if provider_type == ProviderType.OPENAI:
            return ResponseTransformer._openai_completion_to_ollama(provider_response)
        elif provider_type == ProviderType.ANTHROPIC:
            return ResponseTransformer._anthropic_to_ollama(provider_response)
        elif provider_type == ProviderType.GOOGLE:
            return ResponseTransformer._google_to_ollama(provider_response)
        elif provider_type == ProviderType.OPENROUTER:
            return ResponseTransformer._openrouter_to_ollama(provider_response)
        else:
            # Default: minimal transformation
            return provider_response.copy()

    @staticmethod
    def transform_embeddings_response(
        provider_response: Dict[str, Any],
        provider_type: ProviderType,
    ) -> Dict[str, Any]:
        """Transform provider embeddings response to Ollama format."""
        if provider_type == ProviderType.OPENAI:
            return ResponseTransformer._openai_embeddings_to_ollama(provider_response)
        elif provider_type == ProviderType.GOOGLE:
            return ResponseTransformer._google_embeddings_to_ollama(provider_response)
        elif provider_type == ProviderType.OPENROUTER:
            return ResponseTransformer._openrouter_embeddings_to_ollama(provider_response)
        else:
            # Default: minimal transformation
            return provider_response.copy()

    @staticmethod
    def _openai_chat_to_ollama(openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI chat response to Ollama format."""
        if "choices" not in openai_response or not openai_response["choices"]:
            return {"error": "No choices in OpenAI response"}

        choice = openai_response["choices"][0]
        message = choice.get("message", {})

        ollama_response = {
            "model": openai_response.get("model", "unknown"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            },
            "done": choice.get("finish_reason") is not None,
        }

        # Add usage information
        if "usage" in openai_response:
            usage = openai_response["usage"]
            ollama_response.update({
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_duration": 0,
                "eval_duration": 0,
            })

        return ollama_response

    @staticmethod
    def _openai_completion_to_ollama(openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI completion response to Ollama format."""
        if "choices" not in openai_response or not openai_response["choices"]:
            return {"error": "No choices in OpenAI response"}

        choice = openai_response["choices"][0]

        ollama_response = {
            "model": openai_response.get("model", "unknown"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": choice.get("text", ""),
            "done": choice.get("finish_reason") is not None,
        }

        # Add usage information
        if "usage" in openai_response:
            usage = openai_response["usage"]
            ollama_response.update({
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_duration": 0,
                "eval_duration": 0,
            })

        return ollama_response

    @staticmethod
    def _openai_embeddings_to_ollama(openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI embeddings response to Ollama format."""
        if "data" not in openai_response or not openai_response["data"]:
            return {"error": "No data in OpenAI embeddings response"}

        embeddings = [item["embedding"] for item in openai_response["data"]]

        if len(embeddings) == 1:
            return {"embedding": embeddings[0]}
        else:
            return {"embeddings": embeddings}

    @staticmethod
    def _anthropic_to_ollama(anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Anthropic response to Ollama format."""
        if "content" not in anthropic_response or not anthropic_response["content"]:
            return {"error": "No content in Anthropic response"}

        content_blocks = anthropic_response["content"]
        if isinstance(content_blocks, list) and content_blocks:
            content = content_blocks[0].get("text", "")
        else:
            content = str(content_blocks)

        ollama_response = {
            "model": anthropic_response.get("model", "unknown"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": content,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "done": anthropic_response.get("stop_reason") is not None,
        }

        # Add usage information
        if "usage" in anthropic_response:
            usage = anthropic_response["usage"]
            ollama_response.update({
                "prompt_eval_count": usage.get("input_tokens", 0),
                "eval_count": usage.get("output_tokens", 0),
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_duration": 0,
                "eval_duration": 0,
            })

        return ollama_response

    @staticmethod
    def _google_to_ollama(google_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Google response to Ollama format."""
        if "candidates" not in google_response or not google_response["candidates"]:
            return {"error": "No candidates in Google response"}

        candidate = google_response["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text = ""
        if parts:
            text = parts[0].get("text", "")

        ollama_response = {
            "model": "gemini",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": text,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "done": candidate.get("finishReason") is not None,
        }

        # Add usage information
        if "usageMetadata" in google_response:
            usage = google_response["usageMetadata"]
            ollama_response.update({
                "prompt_eval_count": usage.get("promptTokenCount", 0),
                "eval_count": usage.get("candidatesTokenCount", 0),
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_duration": 0,
                "eval_duration": 0,
            })

        return ollama_response

    @staticmethod
    def _google_embeddings_to_ollama(google_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Google embeddings response to Ollama format."""
        if "embedding" not in google_response:
            return {"error": "No embedding in Google response"}

        embedding = google_response["embedding"].get("values", [])
        return {"embedding": embedding}

    @staticmethod
    def _openrouter_to_ollama(openrouter_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenRouter response to Ollama format (similar to OpenAI)."""
        return ResponseTransformer._openai_chat_to_ollama(openrouter_response)

    @staticmethod
    def _openrouter_embeddings_to_ollama(openrouter_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenRouter embeddings response to Ollama format."""
        return ResponseTransformer._openai_embeddings_to_ollama(openrouter_response)