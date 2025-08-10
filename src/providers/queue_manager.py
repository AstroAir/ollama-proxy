"""Request queue manager for handling high-load scenarios.

This module provides request queuing, prioritization, and load balancing
capabilities for the multi-provider system.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

import structlog

from .base import ProviderCapability, ProviderType

logger = structlog.get_logger(__name__)


class RequestPriority(IntEnum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class QueuedRequest:
    """Represents a queued request."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    capability: ProviderCapability = ProviderCapability.CHAT_COMPLETION
    model: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    timeout: float = 300.0
    callback: Optional[Callable] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return time.time() - self.created_at > self.timeout


class RequestQueueManager:
    """Manages request queuing and prioritization."""
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        max_concurrent_requests: int = 100,
        enable_prioritization: bool = True,
    ):
        self.max_queue_size = max_queue_size
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_prioritization = enable_prioritization
        
        # Request queues by priority
        self._queues: Dict[RequestPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in RequestPriority
        }
        
        # Active requests tracking
        self._active_requests: Dict[str, QueuedRequest] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Metrics
        self._total_queued = 0
        self._total_processed = 0
        self._total_expired = 0
        self._total_rejected = 0
        
        # Background task for processing expired requests
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the queue manager."""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        
        logger.info(
            "Started request queue manager",
            max_queue_size=self.max_queue_size,
            max_concurrent=self.max_concurrent_requests,
            prioritization=self.enable_prioritization,
        )

    async def stop(self) -> None:
        """Stop the queue manager."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped request queue manager")

    async def enqueue_request(
        self,
        capability: ProviderCapability,
        model: Optional[str] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> str:
        """Enqueue a request for processing.
        
        Args:
            capability: Required capability
            model: Model name (optional)
            priority: Request priority
            timeout: Request timeout in seconds
            **kwargs: Additional request parameters
            
        Returns:
            Request ID
            
        Raises:
            asyncio.QueueFull: If queue is full
        """
        # Check if we have space in the queue
        total_queued = sum(queue.qsize() for queue in self._queues.values())
        if total_queued >= self.max_queue_size:
            self._total_rejected += 1
            raise asyncio.QueueFull("Request queue is full")
        
        # Create queued request
        request = QueuedRequest(
            capability=capability,
            model=model,
            priority=priority,
            timeout=timeout,
            kwargs=kwargs,
        )
        
        # Add to appropriate priority queue
        queue = self._queues[priority]
        await queue.put(request)
        
        self._total_queued += 1
        
        logger.debug(
            "Enqueued request",
            request_id=request.id,
            capability=capability.value,
            model=model,
            priority=priority.name,
            queue_size=queue.qsize(),
        )
        
        return request.id

    async def dequeue_request(self) -> Optional[QueuedRequest]:
        """Dequeue the next request based on priority.
        
        Returns:
            Next request to process or None if no requests available
        """
        if not self.enable_prioritization:
            # Simple FIFO from normal priority queue
            queue = self._queues[RequestPriority.NORMAL]
            if not queue.empty():
                return await queue.get()
            return None
        
        # Priority-based dequeuing (highest priority first)
        for priority in sorted(RequestPriority, reverse=True):
            queue = self._queues[priority]
            if not queue.empty():
                request = await queue.get()
                
                # Check if request has expired
                if request.is_expired():
                    self._total_expired += 1
                    logger.warning(
                        "Request expired in queue",
                        request_id=request.id,
                        age=time.time() - request.created_at,
                    )
                    continue
                
                return request
        
        return None

    async def process_request(
        self,
        request: QueuedRequest,
        processor: Callable,
    ) -> Any:
        """Process a request with concurrency control.
        
        Args:
            request: Request to process
            processor: Function to process the request
            
        Returns:
            Processing result
        """
        async with self._semaphore:
            self._active_requests[request.id] = request
            
            try:
                logger.debug(
                    "Processing request",
                    request_id=request.id,
                    capability=request.capability.value,
                    model=request.model,
                )
                
                result = await processor(request)
                self._total_processed += 1
                
                logger.debug(
                    "Request processed successfully",
                    request_id=request.id,
                    processing_time=time.time() - request.created_at,
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    "Request processing failed",
                    request_id=request.id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            finally:
                self._active_requests.pop(request.id, None)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_sizes = {
            priority.name: queue.qsize() 
            for priority, queue in self._queues.items()
        }
        
        return {
            "total_queued": self._total_queued,
            "total_processed": self._total_processed,
            "total_expired": self._total_expired,
            "total_rejected": self._total_rejected,
            "active_requests": len(self._active_requests),
            "queue_sizes": queue_sizes,
            "total_in_queue": sum(queue_sizes.values()),
            "max_queue_size": self.max_queue_size,
            "max_concurrent": self.max_concurrent_requests,
        }

    async def _cleanup_expired_requests(self) -> None:
        """Background task to clean up expired requests."""
        while self._running:
            try:
                # Clean up expired active requests
                expired_ids = [
                    req_id for req_id, request in self._active_requests.items()
                    if request.is_expired()
                ]
                
                for req_id in expired_ids:
                    expired_request = self._active_requests.pop(req_id, None)
                    if expired_request:
                        self._total_expired += 1
                        logger.warning(
                            "Active request expired",
                            request_id=req_id,
                            age=time.time() - expired_request.created_at,
                        )
                
                # Sleep before next cleanup
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in cleanup task",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await asyncio.sleep(5)  # Brief pause on error


# Global queue manager instance
_queue_manager: Optional[RequestQueueManager] = None


def get_queue_manager() -> RequestQueueManager:
    """Get the global queue manager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = RequestQueueManager()
    return _queue_manager


async def initialize_queue_manager(
    max_queue_size: int = 1000,
    max_concurrent_requests: int = 100,
    enable_prioritization: bool = True,
) -> None:
    """Initialize the global queue manager."""
    global _queue_manager
    _queue_manager = RequestQueueManager(
        max_queue_size=max_queue_size,
        max_concurrent_requests=max_concurrent_requests,
        enable_prioritization=enable_prioritization,
    )
    await _queue_manager.start()


async def shutdown_queue_manager() -> None:
    """Shutdown the global queue manager."""
    global _queue_manager
    if _queue_manager:
        await _queue_manager.stop()
        _queue_manager = None
