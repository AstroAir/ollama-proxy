"""API endpoint implementations with modern patterns, dependency injection, and enhanced error handling."""

from __future__ import annotations

import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from . import models, openrouter, utils
from .config import AppState
from .exceptions import (
    ErrorContext,
    ErrorType,
    ModelForbiddenError,
    ModelNotFoundError,
    NetworkError,
    ProxyError,
    ValidationError,
)
from .monitoring import MetricType, get_metrics_collector, record_metric, track_request
from .openrouter import OpenRouterClient, OpenRouterError

logger = structlog.get_logger(__name__)

router = APIRouter()


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency to retrieve the application state from the request context.

    This dependency function provides access to the global application state
    that contains configuration, model mappings, and other shared resources.

    Args:
        request: The FastAPI request object containing the application state.

    Returns:
        AppState: The application state containing configuration and model mappings.

    Example:
        >>> @router.get("/example")
        >>> async def example_endpoint(app_state: AppState = Depends(get_app_state)):
        >>>     return {"model_count": len(app_state.all_models)}
    """
    return request.app.state.app_state


def get_openrouter_client(
    app_state: Annotated[AppState, Depends(get_app_state)],
) -> OpenRouterClient:
    """FastAPI dependency to retrieve the OpenRouter client from application state.

    This dependency provides access to the configured OpenRouter client instance
    for making API calls to the OpenRouter service.

    Args:
        app_state: The application state containing the OpenRouter client.

    Returns:
        OpenRouterClient: Configured client for OpenRouter API interactions.

    Example:
        >>> @router.post("/example")
        >>> async def example_endpoint(
        >>>     client: OpenRouterClient = Depends(get_openrouter_client)
        >>> ):
        >>>     response = await client.chat_completion(payload)
        >>>     return response
    """
    return app_state.openrouter_client


@asynccontextmanager
async def error_context(request_id: str | None = None):
    """Async context manager for enhanced error handling and logging.

    This context manager provides structured error handling for API endpoints,
    automatically converting unexpected exceptions to ProxyError instances
    while preserving ProxyError exceptions as-is.

    Args:
        request_id: Optional request identifier for error tracking.

    Yields:
        ErrorContext: Context object containing request metadata.

    Raises:
        ProxyError: For both expected proxy errors and converted unexpected errors.

    Example:
        >>> async with error_context("req_123") as ctx:
        >>>     # API endpoint logic here
        >>>     result = await some_operation()
        >>>     return result
    """
    context = ErrorContext(request_id=request_id)
    try:
        yield context
    except ProxyError:
        # Re-raise proxy errors as-is to maintain error hierarchy
        raise
    except Exception as e:
        # Convert unexpected errors to proxy errors with proper logging
        logger.error("Unexpected error", error=str(e),
                     error_type=type(e).__name__)
        raise ProxyError(
            message=f"Internal server error: {str(e)}",
            error_type=ErrorType.INTERNAL_ERROR,
            status_code=500,
        ) from e


@router.head("/")
async def head_root():
    # Explicitly handle HEAD / to mimic Ollama's 200 OK response
    # Return correct headers but no body
    return Response(
        status_code=200, media_type="text/plain", headers={"Content-Length": "17"}
    )


@router.get("/")
async def root():
    """Root endpoint that mimics Ollama's behavior.

    Returns a simple text response indicating the server is running,
    maintaining compatibility with Ollama clients that check this endpoint.

    Returns:
        PlainTextResponse: Simple text response "Ollama is running".
    """
    # Return plain text like standard Ollama server
    # This also implicitly handles HEAD / requests via FastAPI
    return PlainTextResponse("Ollama is running")


@router.get("/api/version")
def api_version():
    """Get the API version information.

    Returns version information in Ollama-compatible format, indicating
    this is the OpenRouter proxy version.

    Returns:
        dict: Version information with "version" key.

    Example:
        >>> GET /api/version
        >>> {"version": "0.1.0-openrouter"}
    """
    return {"version": "0.1.0-openrouter"}


@router.get("/api/tags")
async def api_tags(app_state: AppState = Depends(get_app_state)):
    """Get available models in Ollama format with enhanced error handling."""
    async with error_context() as ctx:
        # Convert frozenset to set for compatibility
        filter_models_set = (
            set(app_state.model_filter.models)
            if app_state.model_filter.models
            else set()
        )
        filtered = utils.filter_models(app_state.all_models, filter_models_set)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        resp_models = []

        for model in filtered:
            # Enhanced model details with pattern matching
            model_info = _create_ollama_model_details(model)
            resp_models.append(model_info)

        logger.info(
            "Served model tags",
            count=len(resp_models),
            total_available=len(app_state.all_models),
            filtered_count=len(filtered),
            request_id=ctx.request_id,
        )
        return models.OllamaTagsResponse(models=resp_models)


def _create_ollama_model_details(model: dict[str, Any]) -> models.OllamaTagModel:
    """Create Ollama model details from OpenRouter model data using pattern matching."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Extract model information using pattern matching
    match model:
        case {"id": str(model_id), "name": str(model_name)}:
            ollama_name = utils.openrouter_id_to_ollama_name(model_id)
        case {"id": str(model_id)}:
            ollama_name = utils.openrouter_id_to_ollama_name(model_id)
            model_name = model_id
        case _:
            raise ValidationError("Invalid model data structure")

    # Enhanced details based on available model information
    details = models.OllamaTagDetails(
        format="gguf",  # Standard format for compatibility
        family=_extract_model_family(model),
        families=[_extract_model_family(model)],
        parameter_size=_extract_parameter_size(model),
        quantization_level="Unknown",  # Not available from OpenRouter
    )

    return models.OllamaTagModel(
        name=ollama_name,
        modified_at=now,
        size=_extract_model_size(model),
        digest=f"sha256:{hash(model.get('id', ''))}"[
            :16],  # Generate consistent digest
        details=details,
    )


def _extract_model_family(model: dict[str, Any]) -> str:
    """Extract model family using pattern matching."""
    match model.get("id", ""):
        case id_str if "gpt" in id_str.lower():
            return "gpt"
        case id_str if "claude" in id_str.lower():
            return "claude"
        case id_str if "llama" in id_str.lower():
            return "llama"
        case id_str if "gemini" in id_str.lower():
            return "gemini"
        case _:
            return "openrouter"


def _extract_parameter_size(model: dict[str, Any]) -> str:
    """Extract parameter size from model information."""
    # Try to extract from model name or description
    model_str = f"{model.get('name', '')} {model.get('description', '')}".lower(
    )

    match model_str:
        case s if "7b" in s:
            return "7B"
        case s if "13b" in s:
            return "13B"
        case s if "30b" in s:
            return "30B"
        case s if "70b" in s:
            return "70B"
        case s if "175b" in s:
            return "175B"
        case _:
            return "Unknown"


def _extract_model_size(model: dict[str, Any]) -> int:
    """Extract approximate model size in bytes."""
    # Rough estimation based on parameter count
    param_size = _extract_parameter_size(model)
    match param_size:
        case "7B":
            return 7_000_000_000
        case "13B":
            return 13_000_000_000
        case "30B":
            return 30_000_000_000
        case "70B":
            return 70_000_000_000
        case "175B":
            return 175_000_000_000
        case _:
            return 0


def _resolve_and_validate_model(
    model_name: str, app_state: AppState
) -> tuple[str, str]:
    """Resolve and validate model name with enhanced error handling."""
    # First, resolve the potentially short/aliased model name to full names
    # This handles cases like "gpt-4" -> "gpt-4:latest" -> "openai/gpt-4"
    resolved_ollama_name, openrouter_id = utils.resolve_model_name(
        model_name, app_state.ollama_to_openrouter_map
    )

    # Ensure the model exists in our mapping
    if not resolved_ollama_name or not openrouter_id:
        raise ModelNotFoundError(model_name)

    # Apply model filtering if configured
    # This allows administrators to restrict which models are available
    if not app_state.model_filter.is_allowed(resolved_ollama_name):
        raise ModelForbiddenError(resolved_ollama_name)

    return resolved_ollama_name, openrouter_id


@router.post("/api/chat")
async def api_chat(
    request: Request,
    app_state: AppState = Depends(get_app_state),
    openrouter_client: OpenRouterClient = Depends(get_openrouter_client),
):
    """Handle chat completion requests with enhanced error handling and modern patterns."""
    async with error_context() as ctx:
        # Parse and validate request
        body = await request.json()
        req = models.OllamaChatRequest(**body)

        # Add request metadata
        req.metadata = models.RequestMetadata(
            request_id=ctx.request_id,
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None,
        )

        # Validate conversation flow
        if not req.validate_conversation_flow():
            raise ValidationError("Invalid conversation flow detected")

        # Resolve model name with enhanced error handling
        resolved_ollama_name, openrouter_id = _resolve_and_validate_model(
            req.model, app_state
        )

        logger.info(
            "Processing chat request",
            model=req.model,
            resolved_model=resolved_ollama_name,
            openrouter_id=openrouter_id,
            stream=req.stream,
            message_count=req.message_count,
            total_content_length=req.total_content_length,
            request_id=ctx.request_id,
        )

        # Build enhanced OpenRouter payload
        payload = _build_chat_payload(req, openrouter_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if req.stream:
            return await _handle_streaming_chat(
                openrouter_client, payload, resolved_ollama_name, now, ctx.request_id
            )
        else:
            return await _handle_non_streaming_chat(
                openrouter_client, payload, resolved_ollama_name, now, ctx.request_id
            )


def _build_chat_payload(
    req: models.OllamaChatRequest, openrouter_id: str
) -> dict[str, Any]:
    """Build OpenRouter payload from Ollama chat request."""
    # Start with core required fields for OpenRouter API
    payload = {
        "model": openrouter_id,  # Use resolved OpenRouter model ID
        "messages": [msg.model_dump() for msg in req.messages],  # Convert Pydantic models to dicts
        "stream": req.stream,
    }

    # Merge any additional options from the Ollama request
    # This allows passing through parameters like temperature, max_tokens, etc.
    if req.options:
        payload.update(req.options)

    # Handle JSON response format request
    # Ollama uses "json" string, OpenRouter expects structured format
    if req.format == models.ResponseFormat.JSON:
        payload["response_format"] = {"type": "json_object"}

    return payload


async def _handle_streaming_chat(
    client: OpenRouterClient,
    payload: dict[str, Any],
    resolved_model: str,
    timestamp: str,
    request_id: str,
) -> StreamingResponse:
    """Handle streaming chat completion."""

    async def streamer():
        try:
            # Buffer for accumulating partial chunks that may be split across network packets
            buffer = ""
            stream_result = await client.chat_completion(payload, stream=True)

            # Type guard to ensure we have an AsyncIterator (not OpenRouterResponse)
            # This is necessary because the return type is a union
            if not hasattr(stream_result, '__aiter__'):
                raise OpenRouterError(
                    "Expected streaming response but got non-streaming response")

            stream_iterator = stream_result  # type: ignore[assignment]

            # Process each chunk from the OpenRouter streaming response
            async for raw_chunk in stream_iterator:
                if raw_chunk:
                    try:
                        # Decode bytes to string, handling potential encoding issues
                        decoded_chunk = raw_chunk.decode("utf-8")
                        buffer += decoded_chunk
                    except UnicodeDecodeError:
                        # Skip malformed chunks rather than failing the entire stream
                        continue

                    # Process complete lines from the buffer
                    # OpenRouter sends Server-Sent Events (SSE) format with \n delimiters
                    while "\n" in buffer:
                        line_end = buffer.find("\n")
                        raw_line = buffer[:line_end]
                        decoded_line = raw_line.strip()
                        # Keep remaining data in buffer for next iteration
                        buffer = buffer[line_end + 1:]

                        if decoded_line.startswith("data:"):
                            data_content = decoded_line[len("data:"):].strip()
                            if data_content == "[DONE]":
                                continue

                            try:
                                chunk = json.loads(data_content)
                                choice = chunk.get("choices", [{}])[0]
                                delta = choice.get("delta", {})
                                finish_reason = choice.get("finish_reason")

                                if (
                                    delta
                                    and "content" in delta
                                    and delta["content"] is not None
                                ):
                                    content_to_yield = delta["content"]
                                    yield (
                                        json.dumps(
                                            {
                                                "model": resolved_model,
                                                "created_at": timestamp,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": content_to_yield,
                                                },
                                                "done": False,
                                            }
                                        )
                                        + "\n"
                                    )

                                if finish_reason:
                                    yield (
                                        json.dumps(
                                            {
                                                "model": resolved_model,
                                                "created_at": timestamp,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": "",
                                                },
                                                "done": True,
                                            }
                                        )
                                        + "\n"
                                    )
                                    break
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Could not decode JSON from data line",
                                    data_content=data_content,
                                    request_id=request_id,
                                )
                                continue
                            except Exception as e:
                                logger.error(
                                    "Exception processing chunk",
                                    error=str(e),
                                    line=decoded_line,
                                    request_id=request_id,
                                )
                                continue

        except Exception as e:
            logger.error("Streaming error", error=str(e),
                         request_id=request_id)
            yield json.dumps({"error": f"Streaming Error: {str(e)}"}) + "\n"

    return StreamingResponse(streamer(), media_type="application/x-ndjson")


async def _handle_non_streaming_chat(
    client: OpenRouterClient,
    payload: dict[str, Any],
    resolved_model: str,
    timestamp: str,
    request_id: str,
) -> models.OllamaChatResponse:
    """Handle non-streaming chat completion."""
    response_result = await client.chat_completion(payload, stream=False)

    # Type guard to ensure we have an OpenRouterResponse (not AsyncIterator)
    if hasattr(response_result, "__aiter__"):
        raise OpenRouterError(
            "Unexpected streaming response for non-streaming request")

    # Now we know it's OpenRouterResponse
    from .openrouter import OpenRouterResponse

    response: OpenRouterResponse = response_result  # type: ignore[assignment]

    # Enhanced response validation
    if not response.is_success:
        raise OpenRouterError(f"OpenRouter API error: {response.status_code}")

    content = response.get_content()
    if content is None:
        raise OpenRouterError("No content in OpenRouter response")

    logger.info(
        "Chat completion successful",
        model=resolved_model,
        request_id=request_id,
        response_length=len(content),
    )

    return models.OllamaChatResponse(
        model=resolved_model,
        created_at=timestamp,
        message=models.OllamaChatMessage(
            role=models.MessageRole.ASSISTANT, content=content, images=None
        ),
        done=True,
    )


@router.get("/health")
async def health_check():
    """Enhanced health check endpoint with monitoring integration."""
    collector = get_metrics_collector()
    health_status = collector.get_health_status()

    # Record health check metric
    record_metric(
        "health_check_total", 1, {
            "status": health_status["status"]}, MetricType.COUNTER
    )

    return JSONResponse(
        content=health_status,
        status_code=200 if health_status["status"] in [
            "healthy", "degraded"] else 503,
    )


@router.get("/metrics")
async def metrics_endpoint():
    """Metrics endpoint for monitoring systems."""
    collector = get_metrics_collector()

    # Get recent metrics (last 5 minutes)
    recent_metrics = collector.get_metrics(max_age_seconds=300)
    endpoint_stats = collector.get_endpoint_stats()

    return JSONResponse(
        content={
            "metrics": [m.to_dict() for m in recent_metrics],
            "statistics": endpoint_stats,
            "timestamp": time.time(),
        }
    )


@router.post("/api/generate")
async def api_generate(request: Request):
    # Access config, maps, and filter set from app state
    api_key = request.app.state.config["api_key"]
    ollama_map = request.app.state.ollama_to_openrouter_map
    filter_set = request.app.state.filter_set

    try:
        body = await request.json()
        req = models.OllamaGenerateRequest(**body)

        # Resolve model name using the new utility function
        resolved_ollama_name, openrouter_id = utils.resolve_model_name(
            req.model, ollama_map
        )

        if not resolved_ollama_name or not openrouter_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{req.model}' not found.",
            )

        # Check against filter set if it exists
        if filter_set and resolved_ollama_name not in filter_set:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model '{resolved_ollama_name}' is not allowed by the filter.",
            )

        # Build OpenRouter payload (map prompt to messages)
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        payload: dict[str, Any] = {
            "model": openrouter_id,
            "messages": messages,
            "stream": req.stream,
        }
        if req.options:
            payload.update(req.options)
        if req.format == "json":
            payload["response_format"] = {"type": "json_object"}

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if req.stream:

            async def streamer():
                full_response_content = ""
                try:
                    buffer = ""
                    stream_iterator = await openrouter.chat_completion(
                        api_key, payload, stream=True
                    )
                    try:
                        async for raw_chunk in stream_iterator:
                            if raw_chunk:
                                try:
                                    decoded_chunk = raw_chunk.decode("utf-8")
                                    buffer += decoded_chunk
                                except UnicodeDecodeError:
                                    continue

                                while "\n" in buffer:
                                    line_end = buffer.find("\n")
                                    raw_line = buffer[:line_end]
                                    decoded_line = raw_line.strip()
                                    buffer = buffer[line_end + 1:]

                                    if decoded_line.startswith("data:"):
                                        data_content = decoded_line[
                                            len("data:"):
                                        ].strip()
                                        if data_content == "[DONE]":
                                            continue
                                        try:
                                            chunk = json.loads(data_content)
                                            choice = chunk.get(
                                                "choices", [{}])[0]
                                            delta = choice.get("delta", {})
                                            finish_reason = choice.get(
                                                "finish_reason")

                                            if (
                                                delta
                                                and "content" in delta
                                                and delta["content"] is not None
                                            ):
                                                content_to_yield = delta["content"]
                                                full_response_content += (
                                                    content_to_yield
                                                )
                                                yield (
                                                    json.dumps(
                                                        {
                                                            # Use resolved name in response
                                                            "model": resolved_ollama_name,
                                                            "created_at": now,
                                                            "response": content_to_yield,
                                                            "done": False,
                                                        }
                                                    )
                                                    + "\n"
                                                )

                                            if finish_reason:
                                                # Send final done message with empty response and stats
                                                yield (
                                                    json.dumps(
                                                        {
                                                            # Use resolved name in response
                                                            "model": resolved_ollama_name,
                                                            "created_at": now,
                                                            "response": "",
                                                            "done": True,
                                                            # Add synthesized stats here
                                                            "context": req.context
                                                            or [],
                                                            "total_duration": 0,  # Placeholder
                                                            "load_duration": 0,  # Placeholder
                                                            "prompt_eval_count": 0,  # Placeholder
                                                            "prompt_eval_duration": 0,  # Placeholder
                                                            "eval_count": 0,  # Placeholder
                                                            "eval_duration": 0,  # Placeholder
                                                        }
                                                    )
                                                    + "\n"
                                                )
                                                break
                                        except json.JSONDecodeError:
                                            print(
                                                f"Warning: Could not decode JSON from data line: {data_content!r}"
                                            )
                                            continue
                                        except Exception as e:
                                            print(
                                                f"PROXY: Exception processing chunk: {type(e)} - {e}, Line: {decoded_line!r}"
                                            )
                                            continue
                                    else:
                                        pass  # Ignore non-data lines
                    except Exception:
                        raise

                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    try:
                        error_details = exc.response.json()
                        error_message = error_details.get("error", {}).get(
                            "message", str(error_details)
                        )
                    except Exception:
                        error_details = exc.response.text
                        error_message = error_details
                    yield (
                        json.dumps({"error": error_message,
                                   "status_code": status_code})
                        + "\n"
                    )
                except Exception as e:
                    error_message = f"Streaming Error: {str(e)}"
                    yield json.dumps({"error": error_message}) + "\n"

            return StreamingResponse(streamer(), media_type="application/x-ndjson")

    except Exception as exc:
        logger.error("Error in generate endpoint", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/show")
async def api_show(request: Request):
    # Access maps from app state
    ollama_map = request.app.state.ollama_to_openrouter_map
    openrouter_map = request.app.state.openrouter_to_ollama_map

    try:
        body = await request.json()
        req = models.OllamaShowRequest(**body)

        # Prioritize using req.name if provided, otherwise fallback to req.model
        name_to_resolve = req.name if req.name is not None else req.model

        # Use the resolve function which now handles None input
        resolved_ollama_name, _ = utils.resolve_model_name(
            name_to_resolve, ollama_map)

        if not resolved_ollama_name:
            # Provide a clearer error message indicating which name was attempted
            attempted_name = (
                name_to_resolve if name_to_resolve is not None else "(Not provided)"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{attempted_name}' not found.",
            )

        # Construct the stubbed response
        # Note: We don't fetch specific details from OpenRouter for this endpoint
        details = models.OllamaShowDetails()
        # Attempt to extract family/parameter size from ID/name (basic)
        if resolved_ollama_name and "/" in resolved_ollama_name:
            details.family = resolved_ollama_name.split("/")[0]
        if resolved_ollama_name and ":" in resolved_ollama_name:
            # Very basic parsing, may not always be accurate
            parts = resolved_ollama_name.split(":")[-1]
            if "b" in parts.lower() or "m" in parts.lower():  # e.g., 7b, 8x7b, 1.5m
                details.parameter_size = parts.upper()

        # Fill in details fields with empty string/list if not available
        details.parent_model = ""
        details.format = details.format or ""
        details.family = details.family or ""
        details.families = details.families or []
        details.parameter_size = details.parameter_size or ""
        details.quantization_level = details.quantization_level or ""

        # Synthesize model_info as empty dict (Ollama returns a populated dict, but we can't fetch this info)
        model_info: dict[str, Any] = {}

        # Synthesize tensors as empty list (Ollama returns a large list, but we can't fetch this info)
        tensors: list[Any] = []

        # Synthesize license, modelfile, parameters, template as empty strings
        response_obj = models.OllamaShowResponse(
            license="",
            modelfile="",
            parameters="",
            template="",
            details=details,
            model_info=model_info,
            tensors=tensors,
        )

        return response_obj

    except httpx.HTTPStatusError as exc:
        logging.getLogger("ollama-proxy").error(
            "[ERROR] /api/show HTTPStatusError: %s\n%s",
            str(exc),
            traceback.format_exc(),
        )
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.name}' not found upstream at OpenRouter.",
            )
        raise HTTPException(
            status_code=exc.response.status_code, detail=str(exc))
    except HTTPException as http_exc:  # Re-raise existing HTTPExceptions
        logging.getLogger("ollama-proxy").error(
            "[ERROR] /api/show HTTPException: %s\n%s",
            str(http_exc),
            traceback.format_exc(),
        )
        raise http_exc
    except Exception as exc:
        logging.getLogger("ollama-proxy").error(
            "[ERROR] /api/show Exception: %s\n%s", str(
                exc), traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/ps")
async def api_ps():
    # This endpoint provides dummy info about running models
    # No need to access config or state for now
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "models": [],
        "created_at": now,
    }


# --- Embeddings Endpoints ---


async def _handle_embeddings(
    request: Request, ollama_model_name: str, input_data: str | list[str]
) -> dict:
    """Helper to handle shared logic for embedding endpoints."""
    # Access config and maps from app state
    api_key = request.app.state.config["api_key"]
    ollama_map = request.app.state.ollama_to_openrouter_map
    filter_set = request.app.state.filter_set

    # Resolve model name
    resolved_ollama_name, openrouter_id = utils.resolve_model_name(
        ollama_model_name, ollama_map
    )

    if not resolved_ollama_name or not openrouter_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{ollama_model_name}' not found.",
        )

    # Check against filter set if it exists
    if filter_set and resolved_ollama_name not in filter_set:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Model '{resolved_ollama_name}' is not allowed by the filter.",
        )

    # Check if OpenRouter actually supports embeddings for this model ID
    # (We might need a separate check or rely on OpenRouter API error)
    # For now, assume any resolved model might work if the API supports it.

    payload = {"input": input_data, "model": openrouter_id}
    try:
        embedding_response = await openrouter.fetch_embeddings(api_key, payload)
        # Transform OpenRouter response to Ollama format
        # Assuming OpenRouter returns a list of embedding objects with an 'embedding' field
        embeddings = [item["embedding"]
                      for item in embedding_response.get("data", [])]
        # Ollama expects a single list for single input, list of lists for multiple
        if isinstance(input_data, str):
            return {"embedding": embeddings[0] if embeddings else []}
        else:
            return {"embeddings": embeddings}

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/embed")
async def api_embed(request: Request):
    # Uses _handle_embeddings which now takes request object
    try:
        body = await request.json()
        req = models.OllamaEmbedRequest(**body)
        result = await _handle_embeddings(request, req.model, req.input)
        return result
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/embeddings")  # Older endpoint
async def api_embeddings(request: Request):
    # Uses _handle_embeddings which now takes request object
    try:
        body = await request.json()
        req = models.OllamaEmbeddingsRequest(**body)
        # Handle both single prompt (str) and multiple prompts (list)
        result = await _handle_embeddings(request, req.model, req.prompt)
        return result
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Unsupported Endpoints ---


def raise_not_supported():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="This Ollama API endpoint is not supported by the OpenRouter proxy.",
    )


@router.post("/api/create")
async def api_create(request: Request):
    raise_not_supported()


@router.post("/api/copy")
async def api_copy(request: Request):
    raise_not_supported()


@router.delete("/api/delete")
async def api_delete(request: Request):
    raise_not_supported()


@router.post("/api/pull")
async def api_pull(request: Request):
    raise_not_supported()


@router.post("/api/push")
async def api_push(request: Request):
    raise_not_supported()


@router.post("/api/blobs/{digest}")
async def api_blobs_post(digest: str, request: Request):
    raise_not_supported()


@router.head("/api/blobs/{digest}")
async def api_blobs_head(digest: str, request: Request):
    raise_not_supported()
