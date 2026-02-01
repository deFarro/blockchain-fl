"""System metrics collection utilities."""

import time
from typing import Dict, Any, Optional, cast
from shared.logger import setup_logger
import psutil


logger = setup_logger(__name__)


class SystemMetricsCollector:
    """Collects system-level metrics (CPU, memory, network, disk)."""

    def __init__(self):
        """Initialize system metrics collector."""
        self.start_time = time.time()
        self.timestamps: list[float] = []
        self.cpu_samples: list[float] = []
        self.memory_samples: list[float] = []
        self.network_samples: list[Dict[str, int]] = []
        self.disk_samples: list[Dict[str, Any]] = []
        self.sample_count = 0

    def collect_sample(self) -> Dict[str, Any]:
        """
        Collect a single sample of system metrics.

        Returns:
            Dictionary with current system metrics
        """
        try:
            # CPU metrics
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Network metrics
            network_io = psutil.net_io_counters()

            # Disk metrics
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage("/")

            # Get CPU times for absolute measurement (cumulative CPU time)
            cpu_times = psutil.cpu_times()
            cpu_time_total: float = float(
                sum(cpu_times[:4])
            )  # user, nice, system, idle
            timestamp: float = time.time()

            sample = {
                "timestamp": timestamp,
                "cpu": {
                    "time_total_seconds": cpu_time_total,
                    "count": cpu_count,
                },
                "memory": {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "swap_total_bytes": swap.total,
                    "swap_used_bytes": swap.used,
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent if network_io else 0,
                    "bytes_recv": network_io.bytes_recv if network_io else 0,
                    "packets_sent": network_io.packets_sent if network_io else 0,
                    "packets_recv": network_io.packets_recv if network_io else 0,
                },
                "disk": {
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0,
                    "read_count": disk_io.read_count if disk_io else 0,
                    "write_count": disk_io.write_count if disk_io else 0,
                    "total_bytes": disk_usage.total,
                    "used_bytes": disk_usage.used,
                    "free_bytes": disk_usage.free,
                },
            }

            # Store samples for aggregation
            self.timestamps.append(timestamp)
            # Store absolute CPU time (cumulative, so we track the difference)
            self.cpu_samples.append(cpu_time_total)
            # Store absolute memory usage in bytes instead of percentage
            self.memory_samples.append(memory.used)
            network_copy: Dict[str, int] = cast(Dict[str, int], sample["network"])
            disk_copy: Dict[str, Any] = cast(Dict[str, Any], sample["disk"])
            self.network_samples.append(network_copy)
            self.disk_samples.append(disk_copy)
            self.sample_count += 1

            network_data: Dict[str, int] = cast(Dict[str, int], sample["network"])
            bytes_sent = network_data.get("bytes_sent", 0)
            logger.debug(
                f"Collected system metrics sample {self.sample_count}: "
                f"CPU time={cpu_time_total:.2f}s, "
                f"Memory={memory.used / (1024**2):.2f}MB, "
                f"Network sent={bytes_sent} bytes"
            )

            return sample

        except Exception as e:
            logger.warning(f"Error collecting system metrics: {str(e)}")
            return {
                "timestamp": time.time(),
                "error": str(e),
            }

    def get_summary(
        self, start_idx: Optional[int] = None, end_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics from collected samples.

        Args:
            start_idx: Optional start index for sample range (default: 0)
            end_idx: Optional end index for sample range (default: last sample)

        Returns:
            Dictionary with aggregated metrics
        """
        # Determine sample range
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.cpu_samples)

        # Get samples in range
        cpu_range = self.cpu_samples[start_idx:end_idx]
        memory_range = self.memory_samples[start_idx:end_idx]
        network_range = self.network_samples[start_idx:end_idx]
        disk_range = self.disk_samples[start_idx:end_idx]
        sample_count = len(cpu_range)

        if sample_count == 0:
            # Even if no samples collected, return structure with zeros
            # This ensures CSV export includes all columns
            return {
                "sample_count": 0,
                "duration_seconds": 0.0,
                "cpu": {
                    "total_time_seconds": 0,
                },
                "memory": {
                    "avg_used_bytes": 0,
                    "max_used_bytes": 0,
                    "min_used_bytes": 0,
                },
                "network": {
                    "total_bytes_sent": 0,
                    "total_bytes_recv": 0,
                },
                "disk": {
                    "total_bytes_read": 0,
                    "total_bytes_written": 0,
                },
            }

        # Duration of this sample range (wall-clock between first and last sample)
        duration_seconds = 0.0
        if start_idx < len(self.timestamps) and end_idx <= len(self.timestamps):
            ts_range = self.timestamps[start_idx:end_idx]
            if len(ts_range) >= 2:
                duration_seconds = float(ts_range[-1] - ts_range[0])
            elif len(ts_range) == 1:
                duration_seconds = 0.0

        # Calculate statistics for absolute resource usage
        # CPU: Calculate total CPU time used (difference between first and last sample in range)
        cpu_time_used = 0.0
        if len(cpu_range) >= 2:
            cpu_time_used = float(cpu_range[-1] - cpu_range[0])

        # Memory: Calculate average, min, max in bytes
        memory_avg = sum(memory_range) / len(memory_range) if memory_range else 0
        memory_max = max(memory_range) if memory_range else 0
        memory_min = min(memory_range) if memory_range else 0

        # Network totals (difference between first and last in range)
        network_total_sent = 0
        network_total_recv = 0
        if len(network_range) >= 2:
            network_total_sent = (
                network_range[-1]["bytes_sent"] - network_range[0]["bytes_sent"]
            )
            network_total_recv = (
                network_range[-1]["bytes_recv"] - network_range[0]["bytes_recv"]
            )

        # Disk totals (difference between first and last in range)
        disk_total_read = 0
        disk_total_write = 0
        if len(disk_range) >= 2:
            disk_total_read = disk_range[-1]["read_bytes"] - disk_range[0]["read_bytes"]
            disk_total_write = (
                disk_range[-1]["write_bytes"] - disk_range[0]["write_bytes"]
            )

        return {
            "sample_count": sample_count,
            "duration_seconds": duration_seconds,
            "cpu": {
                "total_time_seconds": cpu_time_used,
            },
            "memory": {
                "avg_used_bytes": memory_avg,
                "max_used_bytes": memory_max,
                "min_used_bytes": memory_min,
            },
            "network": {
                "total_bytes_sent": network_total_sent,
                "total_bytes_recv": network_total_recv,
            },
            "disk": {
                "total_bytes_read": disk_total_read,
                "total_bytes_written": disk_total_write,
            },
        }

    def reset(self) -> None:
        """Reset all collected samples."""
        self.start_time = time.time()
        self.timestamps.clear()
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.network_samples.clear()
        self.disk_samples.clear()
        self.sample_count = 0
