"""Metrics collection and monitoring utilities."""

import time
from typing import Dict, Any, Optional
from functools import wraps
from shared.logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    """Collects and logs performance metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {}

    def record_timing(
        self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record timing for an operation.

        Args:
            operation: Operation name (e.g., "blockchain_write", "ipfs_upload")
            duration: Duration in seconds
            metadata: Additional metadata to log
        """
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_duration": 0.0,
                "min_duration": float("inf"),
                "max_duration": 0.0,
            }

        metric = self.metrics[operation]
        metric["count"] += 1
        metric["total_duration"] += duration
        metric["min_duration"] = min(metric["min_duration"], duration)
        metric["max_duration"] = max(metric["max_duration"], duration)

        log_data = {
            "operation": operation,
            "duration_seconds": duration,
            "count": metric["count"],
            "avg_duration": metric["total_duration"] / metric["count"],
        }
        if metadata:
            log_data.update(metadata)

        logger.info("Performance metric", extra=log_data)

    def record_counter(
        self, name: str, value: int = 1, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a counter metric.

        Args:
            name: Counter name (e.g., "client_updates_received", "rollbacks_triggered")
            value: Counter increment (default: 1)
            metadata: Additional metadata to log
        """
        if name not in self.metrics:
            self.metrics[name] = 0

        self.metrics[name] += value

        log_data = {"counter": name, "value": self.metrics[name], "increment": value}
        if metadata:
            log_data.update(metadata)

        logger.info("Counter metric", extra=log_data)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def timed_operation(operation_name: str):
    """
    Decorator to time an operation and record metrics.

    Args:
        operation_name: Name of the operation for metrics

    Example:
        @timed_operation("blockchain_write")
        def write_to_blockchain(...):
            ...
    """

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                get_metrics_collector().record_timing(
                    operation_name,
                    duration,
                    metadata={"function": func.__name__, "status": "success"},
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                get_metrics_collector().record_timing(
                    operation_name,
                    duration,
                    metadata={
                        "function": func.__name__,
                        "status": "error",
                        "error": str(e),
                    },
                )
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                get_metrics_collector().record_timing(
                    operation_name,
                    duration,
                    metadata={"function": func.__name__, "status": "success"},
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                get_metrics_collector().record_timing(
                    operation_name,
                    duration,
                    metadata={
                        "function": func.__name__,
                        "status": "error",
                        "error": str(e),
                    },
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
