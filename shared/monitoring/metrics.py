"""Metrics collection and monitoring utilities."""

import asyncio
import time
from typing import Dict, Any, Optional, List
from functools import wraps
from shared.logger import setup_logger
from shared.monitoring.system_metrics import SystemMetricsCollector

logger = setup_logger(__name__)


class MetricsCollector:
    """Collects and logs performance metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {}
        self.system_metrics = SystemMetricsCollector()
        self.operation_timings: Dict[str, List[float]] = {}  # Store individual timings
        self.operation_metadata: Dict[str, List[Dict[str, Any]]] = (
            {}
        )  # Store metadata for each operation
        self.scenario_info: Dict[str, Any] = {}  # Store scenario configuration
        # Track system metrics samples per iteration: iteration -> list of sample indices
        self.iteration_system_samples: Dict[int, List[int]] = {}
        # Track which sample index corresponds to which iteration
        self.sample_to_iteration: Dict[int, int] = {}

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
            self.operation_timings[operation] = []
            self.operation_metadata[operation] = []

        metric = self.metrics[operation]
        metric["count"] += 1
        metric["total_duration"] += duration
        metric["min_duration"] = min(metric["min_duration"], duration)
        metric["max_duration"] = max(metric["max_duration"], duration)

        # Store individual timing and metadata
        self.operation_timings[operation].append(duration)
        if metadata:
            self.operation_metadata[operation].append(metadata.copy())
        else:
            self.operation_metadata[operation].append({})

        # Track which timing sample index corresponds to which iteration
        # This helps us match timing samples to system samples later
        if metadata and "iteration" in metadata:
            iteration = metadata["iteration"]
            if isinstance(iteration, (int, float)):
                iteration = int(iteration)
                # Store the iteration so the next collect_system_sample() call can associate with it
                # We'll use a simple approach: track the last iteration seen
                # When collect_system_sample() is called, it will check for pending iteration
                if not hasattr(self, "_pending_iteration"):
                    self._pending_iteration = None
                self._pending_iteration = iteration
                # Also track which timing sample index this is for each operation
                timing_idx = len(self.operation_timings[operation]) - 1
                if not hasattr(self, "_timing_to_iteration"):
                    self._timing_to_iteration = (
                        {}
                    )  # (operation, timing_idx) -> iteration
                self._timing_to_iteration[(operation, timing_idx)] = iteration

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

    def set_scenario_info(self, scenario_info: Dict[str, Any]) -> None:
        """
        Set scenario configuration information.

        Args:
            scenario_info: Dictionary with scenario details (e.g., blockchain_enabled, ipfs_enabled, dataset_name, etc.)
        """
        self.scenario_info = scenario_info.copy()
        logger.info(f"Scenario info set: {scenario_info}")

    def collect_system_sample(self) -> Dict[str, Any]:
        """
        Collect a sample of system metrics.

        Returns:
            Dictionary with current system metrics
        """
        sample: Dict[str, Any] = self.system_metrics.collect_sample()
        if "error" in sample:
            logger.warning(f"Failed to collect system metrics: {sample.get('error')}")
        else:
            # Associate this sample with the pending iteration if available
            if (
                hasattr(self, "_pending_iteration")
                and self._pending_iteration is not None
            ):
                iteration = self._pending_iteration
                sample_idx = self.system_metrics.sample_count - 1
                if iteration not in self.iteration_system_samples:
                    self.iteration_system_samples[iteration] = []
                self.iteration_system_samples[iteration].append(sample_idx)
                self.sample_to_iteration[sample_idx] = iteration
                # Clear pending iteration after associating
                self._pending_iteration = None
        return sample

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics including system metrics summary.

        Returns:
            Dictionary of metrics with system metrics summary
        """
        metrics = self.metrics.copy()

        # Add system metrics summary
        metrics["system_metrics"] = self.system_metrics.get_summary()

        # Add scenario info
        metrics["scenario_info"] = self.scenario_info.copy()

        # Add detailed timing data
        metrics["detailed_timings"] = {}
        for operation, timings in self.operation_timings.items():
            metrics["detailed_timings"][operation] = {
                "timings": timings,
                "metadata": self.operation_metadata.get(operation, []),
            }

        return metrics

    def get_metrics_for_export(self) -> Dict[str, Any]:
        """
        Get metrics in a format suitable for CSV export.

        Returns:
            Dictionary with flattened metrics ready for CSV export
        """
        metrics = self.get_metrics()

        # Calculate per-iteration system metrics summaries
        # For each iteration, calculate summary from samples collected during that iteration
        iteration_system_metrics = {}
        # Sort iterations to process them in order
        sorted_iterations = sorted(self.iteration_system_samples.keys())
        prev_iteration_end_idx = 0

        for iteration in sorted_iterations:
            sample_indices = self.iteration_system_samples[iteration]
            if sample_indices:
                # Get the range of samples for this iteration
                start_idx = min(sample_indices)
                end_idx = max(sample_indices) + 1  # +1 because end_idx is exclusive

                # Calculate summary for this iteration's samples
                # For single-sample iterations, calculate from previous iteration's end
                # For multi-sample iterations, calculate difference within the iteration
                if len(sample_indices) == 1:
                    # For single sample, calculate from previous iteration's end (or start if first)
                    calc_start_idx = (
                        prev_iteration_end_idx
                        if prev_iteration_end_idx < start_idx
                        else start_idx
                    )
                    iteration_summary = self.system_metrics.get_summary(
                        start_idx=calc_start_idx, end_idx=end_idx
                    )
                else:
                    # For multiple samples, calculate difference within the iteration
                    iteration_summary = self.system_metrics.get_summary(
                        start_idx=start_idx, end_idx=end_idx
                    )

                iteration_system_metrics[iteration] = iteration_summary
                # Update previous iteration's end index for next iteration
                prev_iteration_end_idx = end_idx

        # Flatten metrics for CSV export
        export_data = {
            "scenario_info": metrics.get("scenario_info", {}),
            "system_metrics": metrics.get("system_metrics", {}),
            "operation_metrics": {},
            "detailed_timings": metrics.get("detailed_timings", {}),
            "iteration_system_samples": self.iteration_system_samples.copy(),
            "iteration_system_metrics": iteration_system_metrics,
        }

        # Add operation summaries
        for operation, metric_data in metrics.items():
            if operation not in ["system_metrics", "scenario_info", "detailed_timings"]:
                export_data["operation_metrics"][operation] = metric_data

        return export_data

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.operation_timings.clear()
        self.operation_metadata.clear()
        self.scenario_info.clear()
        self.iteration_system_samples.clear()
        self.sample_to_iteration.clear()
        if hasattr(self, "_pending_iteration"):
            self._pending_iteration = None
        if hasattr(self, "_timing_to_iteration"):
            self._timing_to_iteration.clear()
        self.system_metrics.reset()


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

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
