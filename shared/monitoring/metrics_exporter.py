"""Metrics export utilities for CSV generation."""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from shared.logger import setup_logger

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

logger = setup_logger(__name__)

# Module-level variables to track active CSV file across instances
_active_csv_path: Optional[Path] = None
_active_fieldnames: Optional[List[str]] = None


class MetricsExporter:
    """Exports metrics to CSV format."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize metrics exporter.

        Args:
            output_dir: Directory to save CSV files (defaults to ./metrics_output)
        """
        if output_dir is None:
            output_dir = Path("./metrics_output")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_active_csv_path(cls) -> Optional[Path]:
        """Return the path of the active CSV file for incremental export, or None."""
        global _active_csv_path
        return _active_csv_path

    def export_to_csv(
        self, metrics_data: Dict[str, Any], filename: Optional[str] = None
    ) -> Path:
        """
        Export metrics to CSV file.

        Args:
            metrics_data: Dictionary with metrics data (from get_metrics_for_export)
            filename: Optional filename (defaults to timestamp-based name)

        Returns:
            Path to created CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"

        csv_path = self.output_dir / filename

        self._delete_previous_csv_files(csv_path)

        # Flatten metrics data for CSV
        rows = self._flatten_metrics(metrics_data)

        if not rows:
            logger.warning("No metrics data to export")
            return csv_path

        # Get all unique keys from all rows
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        # Add units to fieldnames
        fieldnames_with_units = [
            self._add_unit_to_header(field) for field in fieldnames
        ]

        # Write CSV file
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header with units
            writer.writerow(dict(zip(fieldnames, fieldnames_with_units)))
            writer.writerows(rows)

        logger.info(f"Exported metrics to {csv_path} ({len(rows)} rows)")
        return csv_path

    def initialize_csv_file(
        self, scenario_info: Dict[str, Any], filename: Optional[str] = None
    ) -> Path:
        """
        Initialize a CSV file with headers for incremental writing.

        Args:
            scenario_info: Scenario configuration information
            filename: Optional filename (defaults to timestamp-based name)

        Returns:
            Path to created CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Add scenario info to filename
            bc_status = "bc_on" if scenario_info.get("blockchain_enabled") else "bc_off"
            ipfs_status = "ipfs_on" if scenario_info.get("ipfs_enabled") else "ipfs_off"
            filename = f"metrics_{timestamp}_{bc_status}_{ipfs_status}.csv"

        csv_path = self.output_dir / filename
        self._delete_previous_csv_files(csv_path)

        # Create a dummy row to determine all possible fieldnames.
        # Include all known operations so the header has every timing_* column
        # (otherwise initialize+append would drop columns for operations not in the dummy).
        # Exclude blockchain_enabled from export (used only for filename).
        scenario_info_for_export = {
            k: v for k, v in scenario_info.items() if k != "blockchain_enabled"
        }
        dummy_metrics = {
            "scenario_info": scenario_info_for_export,
            "system_metrics": {
                "sample_count": 0,
                "duration_seconds": 0.0,
                "cpu": {"total_time_seconds": 0.0},
                "memory": {
                    "avg_used_bytes": 0,
                    "max_used_bytes": 0,
                    "min_used_bytes": 0,
                },
                "network": {"total_bytes_sent": 0, "total_bytes_recv": 0},
                "disk": {"total_bytes_read": 0, "total_bytes_written": 0},
            },
            "operation_metrics": {},
            "detailed_timings": {
                "blockchain_register": {
                    "timings": [0.0],
                    "metadata": [
                        {
                            "iteration": 0,
                            "model_version_id": "",
                            "transaction_id": "",
                        }
                    ],
                },
                "fedavg_aggregation": {
                    "timings": [0.0],
                    "metadata": [
                        {
                            "iteration": 0,
                            "num_clients": 0,
                            "total_samples": 0,
                            "excluded_clients": 0,
                        }
                    ],
                },
                "ipfs_upload": {
                    "timings": [0.0],
                    "metadata": [{"model_version_id": "", "cid": "", "size_bytes": 0}],
                },
                "model_validation": {
                    "timings": [0.0],
                    "metadata": [
                        {"model_version_id": "", "accuracy": 0.0, "loss": 0.0}
                    ],
                },
                "blockchain_get_provenance": {
                    "timings": [0.0],
                    "metadata": [{"iteration": 0, "model_version_id": ""}],
                },
                "blockchain_get_most_recent_rollback": {
                    "timings": [0.0],
                    "metadata": [{"iteration": 0, "found": False}],
                },
                "blockchain_list_models": {"timings": [0.0], "metadata": [{"iteration": 0, "total": 0}]},
                "blockchain_record_validation": {
                    "timings": [0.0],
                    "metadata": [
                        {
                            "iteration": 0,
                            "model_version_id": "",
                            "transaction_id": "",
                        }
                    ],
                },
                "blockchain_rollback": {
                    "timings": [0.0],
                    "metadata": [
                        {
                            "iteration": 0,
                            "target_version_id": "",
                            "transaction_id": "",
                        }
                    ],
                },
                "ipfs_download": {
                    "timings": [0.0],
                    "metadata": [
                        {
                            "iteration": 0,
                            "cid": "",
                            "model_version_id": "",
                            "size_bytes": 0,
                        }
                    ],
                },
            },
            "iteration_system_samples": {},
            "iteration_system_metrics": {},
            "blockchain_service_system_metrics": {
                "timestamp": "",
                "memory": {
                    "alloc_bytes": 0,
                    "total_alloc_bytes": 0,
                    "sys_bytes": 0,
                    "num_gc": 0,
                },
            },
        }
        dummy_rows = self._flatten_metrics(dummy_metrics)

        # Get all unique keys from dummy rows
        all_keys: set[str] = set()
        for row in dummy_rows:
            all_keys.update(row.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        # Add units to fieldnames
        fieldnames_with_units = [
            self._add_unit_to_header(field) for field in fieldnames
        ]

        # Write CSV header
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header with units
            writer.writerow(dict(zip(fieldnames, fieldnames_with_units)))

        # Store in module-level variables for persistence across instances
        global _active_csv_path, _active_fieldnames
        _active_csv_path = csv_path
        _active_fieldnames = fieldnames

        logger.info(f"Initialized CSV file {csv_path} with {len(fieldnames)} columns")
        return csv_path

    def append_iteration_metrics(self, iteration: int, metrics_collector: Any) -> None:
        """
        Append metrics for a single iteration to the active CSV file.

        Args:
            iteration: Iteration number
            metrics_collector: MetricsCollector instance
        """
        global _active_csv_path, _active_fieldnames

        if _active_csv_path is None:
            logger.warning(
                "No active CSV file. Call initialize_csv_file() first. "
                "Falling back to full export."
            )
            # Fallback to full export
            metrics_data = metrics_collector.get_metrics_for_export()
            self.export_to_csv(metrics_data)
            return

        if not _active_csv_path.exists():
            logger.warning(
                f"Active CSV file {_active_csv_path} does not exist. "
                "Reinitializing..."
            )
            # Reinitialize if file was deleted
            scenario_info = metrics_collector.scenario_info
            self.initialize_csv_file(scenario_info)

        # Get metrics for this specific iteration
        # Calculate per-iteration system metrics on-the-fly for this specific iteration
        metrics_data = metrics_collector.get_metrics()

        # Find all timing samples for this iteration to determine which system samples to use
        iteration_timing_indices: dict[str, list[int]] = (
            {}
        )  # operation -> list of timing indices for this iteration
        for operation, metadata_list in metrics_collector.operation_metadata.items():
            for idx, metadata in enumerate(metadata_list):
                if metadata.get("iteration") == iteration:
                    if operation not in iteration_timing_indices:
                        iteration_timing_indices[operation] = []
                    iteration_timing_indices[operation].append(idx)

        # Find system samples associated with this iteration
        iteration_sample_indices = metrics_collector.iteration_system_samples.get(
            iteration, []
        )

        logger.debug(
            f"Iteration {iteration}: Found {len(iteration_sample_indices)} system samples: {iteration_sample_indices}, "
            f"Total samples collected: {metrics_collector.system_metrics.sample_count}, "
            f"All iteration samples: {dict(metrics_collector.iteration_system_samples)}"
        )

        # Use only the first contiguous block of samples for this iteration.
        # After a rollback, the same iteration number can be reused and many more
        # system samples get attributed to it, which would make the summary span
        # the whole run (huge cpu/memory). We only want the first occurrence.
        iteration_sample_indices = self._first_contiguous_sample_block(
            sorted(iteration_sample_indices)
        )

        # Calculate per-iteration system metrics summary
        iteration_system_summary = None
        if iteration_sample_indices:
            start_idx = min(iteration_sample_indices)
            end_idx = max(iteration_sample_indices) + 1

            # Calculate summary for this iteration's samples
            if len(iteration_sample_indices) == 1:
                # For single sample, calculate from previous iteration's end (or start if first)
                prev_iterations = [
                    it
                    for it in sorted(metrics_collector.iteration_system_samples.keys())
                    if it < iteration
                ]
                if prev_iterations:
                    prev_end_idx = (
                        max(
                            metrics_collector.iteration_system_samples[
                                prev_iterations[-1]
                            ]
                        )
                        + 1
                    )
                    calc_start_idx = (
                        prev_end_idx if prev_end_idx < start_idx else start_idx
                    )
                else:
                    calc_start_idx = start_idx

                iteration_system_summary = metrics_collector.system_metrics.get_summary(
                    start_idx=calc_start_idx, end_idx=end_idx
                )
            else:
                # For multiple samples, calculate difference within the iteration
                iteration_system_summary = metrics_collector.system_metrics.get_summary(
                    start_idx=start_idx, end_idx=end_idx
                )
        else:
            # No samples for this iteration, use empty summary
            logger.warning(f"No system samples found for iteration {iteration}")
            iteration_system_summary = {
                "sample_count": 0,
                "duration_seconds": 0.0,
                "cpu": {"total_time_seconds": 0.0},
                "memory": {
                    "avg_used_bytes": 0,
                    "max_used_bytes": 0,
                    "min_used_bytes": 0,
                },
                "network": {"total_bytes_sent": 0, "total_bytes_recv": 0},
                "disk": {"total_bytes_read": 0, "total_bytes_written": 0},
            }

        # Build row(s) for this iteration explicitly, always using per-iteration system
        # summary (never global), so each iteration gets its own system metrics.
        detailed_timings = metrics_data.get("detailed_timings", {})
        scenario_info = metrics_data.get("scenario_info", {})

        # Fetch per-iteration blockchain-service process metrics (if enabled)
        bc_sys_for_iteration: Dict[str, Any] = {}
        if scenario_info.get("blockchain_enabled") and httpx is not None:
            try:
                from shared.config import settings
                if getattr(settings, "blockchain_service_url", None):
                    bc_url = settings.blockchain_service_url.rstrip("/")
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.get(f"{bc_url}/api/v1/system-metrics")
                        resp.raise_for_status()
                        bc_sys_for_iteration = resp.json()
                        metrics_collector.record_blockchain_service_metrics(
                            iteration, bc_sys_for_iteration
                        )
            except Exception as e:
                logger.debug(
                    "Could not fetch per-iteration blockchain-service metrics: %s", e
                )

        # One row per FL iteration: use a single canonical timing index per iteration.
        # Prefer the index where model_validation (or fedavg, blockchain_register) has this
        # iteration — i.e. the main completion of the iteration, not rollback-related ops.
        canonical_idx = self._canonical_timing_index_for_iteration(
            detailed_timings, iteration
        )
        if canonical_idx is None:
            logger.debug(f"No timing index found for iteration {iteration}")
            return

        # Build exactly one row for this iteration (exclude blockchain_enabled from export)
        base_row: Dict[str, Any] = {}
        for key, value in scenario_info.items():
            if key == "blockchain_enabled":
                continue
            base_row[f"scenario_{key}"] = self._format_value(value)

        system_summary_for_row = iteration_system_summary
        if system_summary_for_row is None:
            system_summary_for_row = {
                "sample_count": 0,
                "duration_seconds": 0.0,
                "cpu": {"total_time_seconds": 0.0},
                "memory": {
                    "avg_used_bytes": 0,
                    "max_used_bytes": 0,
                    "min_used_bytes": 0,
                },
                "network": {"total_bytes_sent": 0, "total_bytes_recv": 0},
                "disk": {"total_bytes_read": 0, "total_bytes_written": 0},
            }

        i = canonical_idx
        row = dict(base_row)
        row["timing_sample_index"] = self._format_value(i)
        for key, value in system_summary_for_row.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    row[f"system_{key}_{sub_key}"] = self._format_value(sub_value)
            else:
                row[f"system_{key}"] = self._format_value(value)
        for operation, timing_data in detailed_timings.items():
            timings = timing_data.get("timings", [])
            metadata_list = timing_data.get("metadata", [])
            if i < len(timings):
                row[f"timing_{operation}_duration"] = self._format_value(timings[i])
            if i < len(metadata_list):
                for meta_key, meta_value in metadata_list[i].items():
                    row[f"timing_{operation}_{meta_key}"] = self._format_value(
                        meta_value
                    )
        # Per-iteration total time per operation (e.g. total get_provenance time this iteration)
        for op, total_sec in self._operation_total_durations_for_iteration(
            detailed_timings, iteration
        ).items():
            row[f"timing_{op}_total_duration"] = self._format_value(total_sec)
        # Per-iteration blockchain-service process metrics
        for key, value in self._flatten_dict_with_prefix(
            bc_sys_for_iteration, "blockchain_service_"
        ).items():
            row[key] = self._format_value(value)
        iteration_rows = [row]

        # Append rows to CSV
        assert _active_fieldnames is not None, "Cannot append: no active CSV session"
        fieldnames: List[str] = _active_fieldnames
        with open(_active_csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for row in iteration_rows:
                complete_row = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(complete_row)

        logger.info(
            f"Appended 1 row for iteration {iteration} to {_active_csv_path}"
        )

    @staticmethod
    def _operation_total_durations_for_iteration(
        detailed_timings: Dict[str, Any], iteration: int
    ) -> Dict[str, float]:
        """
        For each operation, sum duration over all timing indices that belong to
        this iteration (any op has metadata.iteration == iteration at that index).
        So "total time spent on get_provenance this iteration" includes all
        get_provenance calls at those indices, even if get_provenance has no
        iteration in metadata.
        """
        # Set of timing indices where at least one operation has this iteration
        indices_for_iteration: set[int] = set()
        for timing_data in detailed_timings.values():
            for idx, metadata in enumerate(timing_data.get("metadata", [])):
                if metadata.get("iteration") == iteration:
                    indices_for_iteration.add(idx)

        totals: Dict[str, float] = {}
        for operation, timing_data in detailed_timings.items():
            timings = timing_data.get("timings", [])
            total = 0.0
            for i in indices_for_iteration:
                if i < len(timings):
                    total += timings[i]
            totals[operation] = total
        return totals

    @staticmethod
    def _canonical_timing_index_for_iteration(
        detailed_timings: Dict[str, Any], iteration: int
    ) -> Optional[int]:
        """
        Return a single timing index that represents this FL iteration, so we write
        one row per iteration. Use position-based index first (iteration N -> index N-1)
        so we get 0, 1, 2, ... and avoid duplicates if metadata.iteration is updated
        in place; then fall back to metadata-based search for rollback/out-of-order.
        """
        # Prefer index = iteration - 1 for model_validation (1-based iteration -> 0-based index)
        # so we get a unique index per iteration even if metadata is overwritten later
        mv = detailed_timings.get("model_validation")
        if mv:
            timings = mv.get("timings", [])
            idx = int(iteration) - 1
            if 0 <= idx < len(timings):
                return idx
        # Fallback: first index where an op has this iteration in metadata
        preferred_ops = (
            "model_validation",
            "fedavg_aggregation",
            "blockchain_register",
        )
        for op in preferred_ops:
            timing_data = detailed_timings.get(op)
            if not timing_data:
                continue
            metadata_list = timing_data.get("metadata", [])
            for idx, metadata in enumerate(metadata_list):
                if metadata.get("iteration") == iteration:
                    return idx
        for operation, timing_data in detailed_timings.items():
            metadata_list = timing_data.get("metadata", [])
            for idx, metadata in enumerate(metadata_list):
                if metadata.get("iteration") == iteration:
                    return idx
        return None

    @staticmethod
    def _first_contiguous_sample_block(sorted_indices: List[int]) -> List[int]:
        """
        Return the first contiguous block of sample indices.
        After rollback, an iteration can have many non-contiguous indices (e.g. [2,3,50..120]).
        Using the full range would produce a global-like summary; we use only the first run.
        """
        if not sorted_indices:
            return []
        block = [sorted_indices[0]]
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == block[-1] + 1:
                block.append(sorted_indices[i])
            else:
                break
        return block

    def _delete_previous_csv_files(self, current_csv_path: Path) -> None:
        """
        Delete previous CSV metrics files in the output directory.

        Args:
            current_csv_path: Path to the CSV file that will be created (to avoid deleting it)
        """
        try:
            # Find all CSV files matching metrics_*.csv pattern
            for csv_file in self.output_dir.glob("metrics_*.csv"):
                # Don't delete the file we're about to create
                if csv_file != current_csv_path:
                    csv_file.unlink()
                    logger.info(f"Deleted previous metrics file: {csv_file}")
        except Exception as e:
            logger.warning(f"Failed to delete previous CSV files: {str(e)}")

    def _flatten_metrics(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten nested metrics data into rows suitable for CSV.

        Args:
            metrics_data: Nested metrics dictionary

        Returns:
            List of flattened dictionaries (one per row)
        """
        rows = []

        # Extract scenario info
        scenario_info = metrics_data.get("scenario_info", {})

        # Extract system metrics summary (global summary, used as fallback)
        system_metrics = metrics_data.get("system_metrics", {})

        # Extract iteration system samples mapping
        iteration_system_samples = metrics_data.get("iteration_system_samples", {})

        # Extract detailed timings (no op_* aggregates; each row has per-iteration timing_* only)
        detailed_timings = metrics_data.get("detailed_timings", {})

        # Create base row with scenario info only (no op_*)
        base_row = {}

        # Add scenario info (exclude blockchain_enabled from export)
        for key, value in scenario_info.items():
            if key == "blockchain_enabled":
                continue
            base_row[f"scenario_{key}"] = self._format_value(value)

        # Per-iteration blockchain-service metrics (fallback: run-level snapshot)
        iteration_bc_metrics = metrics_data.get(
            "iteration_blockchain_service_metrics", {}
        )
        run_level_bc_sys = metrics_data.get("blockchain_service_system_metrics", {})

        # Create one row per FL iteration (same as incremental append)
        if detailed_timings:
            iteration_system_metrics = metrics_data.get(
                "iteration_system_metrics", {}
            )
            # Collect iterations that have a canonical timing index
            iterations_seen: set[int] = set()
            for op in ("model_validation", "fedavg_aggregation", "blockchain_register"):
                timing_data = detailed_timings.get(op)
                if not timing_data:
                    continue
                for metadata in timing_data.get("metadata", []):
                    iter_val = metadata.get("iteration")
                    if isinstance(iter_val, (int, float)):
                        iterations_seen.add(int(iter_val))
            for op, timing_data in detailed_timings.items():
                for metadata in timing_data.get("metadata", []):
                    iter_val = metadata.get("iteration")
                    if isinstance(iter_val, (int, float)):
                        iterations_seen.add(int(iter_val))

            for iteration in sorted(iterations_seen):
                i = self._canonical_timing_index_for_iteration(
                    detailed_timings, iteration
                )
                if i is None:
                    continue
                row = base_row.copy()
                row["timing_sample_index"] = self._format_value(i)

                # Per-iteration blockchain-service metrics (fallback to run-level)
                iteration_key = int(iteration)
                bc_sys = (
                    iteration_bc_metrics.get(iteration_key)
                    or iteration_bc_metrics.get(str(iteration_key))
                    or run_level_bc_sys
                )
                for key, value in self._flatten_dict_with_prefix(
                    bc_sys, "blockchain_service_"
                ).items():
                    row[key] = self._format_value(value)

                per_iter_summary = iteration_system_metrics.get(
                    iteration_key
                ) or iteration_system_metrics.get(str(iteration_key))
                if per_iter_summary is not None and isinstance(per_iter_summary, dict):
                    system_summary = per_iter_summary
                else:
                    system_summary = system_metrics

                for key, value in system_summary.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            row[f"system_{key}_{sub_key}"] = self._format_value(
                                sub_value
                            )
                    else:
                        row[f"system_{key}"] = self._format_value(value)

                for operation, timing_data in detailed_timings.items():
                    timings = timing_data.get("timings", [])
                    metadata_list = timing_data.get("metadata", [])

                    if i < len(timings):
                        row[f"timing_{operation}_duration"] = self._format_value(
                            timings[i]
                        )
                    if i < len(metadata_list):
                        for meta_key, meta_value in metadata_list[i].items():
                            row[f"timing_{operation}_{meta_key}"] = (
                                self._format_value(meta_value)
                            )

                for op, total_sec in self._operation_total_durations_for_iteration(
                    detailed_timings, iteration
                ).items():
                    row[f"timing_{op}_total_duration"] = self._format_value(
                        total_sec
                    )

                rows.append(row)
        else:
            # No detailed timings, just add summary row with global system metrics
            row = base_row.copy()
            for key, value in self._flatten_dict_with_prefix(
                run_level_bc_sys, "blockchain_service_"
            ).items():
                row[key] = self._format_value(value)
            system_summary = system_metrics
            for key, value in system_summary.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"system_{key}_{sub_key}"] = self._format_value(sub_value)
                else:
                    row[f"system_{key}"] = self._format_value(value)
            rows.append(row)

        return rows

    def _flatten_dict_with_prefix(
        self, d: Dict[str, Any], prefix: str
    ) -> Dict[str, Any]:
        """Flatten nested dict to top-level keys with prefix (e.g. memory.alloc_bytes -> prefix_memory_alloc_bytes)."""
        out: Dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, dict) and not isinstance(value, list):
                for sub_key, sub_value in self._flatten_dict_with_prefix(
                    value, f"{prefix}{key}_"
                ).items():
                    out[sub_key] = sub_value
            else:
                out[f"{prefix}{key}"] = value
        return out

    def _format_value(self, value: Any) -> str:
        """
        Format a value for CSV export.

        Args:
            value: Value to format

        Returns:
            Formatted string
        """
        if value is None:
            return ""
        elif isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, float):
            return f"{value:.6f}"
        else:
            return str(value)

    def _add_unit_to_header(self, field_name: str) -> str:
        """
        Add units to CSV header names.

        Args:
            field_name: Original field name

        Returns:
            Field name with unit in brackets
        """
        # Define unit mappings
        unit_map = {
            # Duration/time metrics
            "duration": "seconds",
            "total_duration": "seconds",
            "min_duration": "seconds",
            "max_duration": "seconds",
            "avg_duration": "seconds",
            "duration_seconds": "seconds",
            "training_duration_seconds": "seconds",
            # CPU metrics (absolute resource usage)
            "cpu_time_total_seconds": "seconds",
            "cpu_total_time_seconds": "seconds",
            "cpu_count": "cores",
            # Memory metrics (absolute resource usage in bytes)
            "memory_total_bytes": "bytes",
            "memory_available_bytes": "bytes",
            "memory_used_bytes": "bytes",
            "memory_avg_used_bytes": "bytes",
            "memory_max_used_bytes": "bytes",
            "memory_min_used_bytes": "bytes",
            "swap_total_bytes": "bytes",
            "swap_used_bytes": "bytes",
            # Network metrics
            "network_total_bytes_sent": "bytes",
            "network_total_bytes_recv": "bytes",
            "bytes_sent": "bytes",
            "bytes_recv": "bytes",
            "packets_sent": "packets",
            "packets_recv": "packets",
            # Disk metrics
            "disk_total_bytes_read": "bytes",
            "disk_total_bytes_written": "bytes",
            "read_bytes": "bytes",
            "write_bytes": "bytes",
            "read_count": "operations",
            "write_count": "operations",
            "total_bytes": "bytes",
            "used_bytes": "bytes",
            "free_bytes": "bytes",
            # Size metrics
            "size_bytes": "bytes",
            "alloc_bytes": "bytes",
            "total_alloc_bytes": "bytes",
            "sys_bytes": "bytes",
            # Count metrics
            "count": "operations",
            "sample_count": "samples",
            "num_clients": "clients",
            "total_iterations": "iterations",
            "rollback_count": "rollbacks",
            # Accuracy/metrics
            "accuracy": "%",
            "final_accuracy": "%",
            "target_accuracy": "%",
            "loss": "value",
            "final_loss": "value",
            # Configuration
            "max_iterations": "iterations",
            "max_rollbacks": "rollbacks",
        }

        # Check if any unit keyword matches
        for keyword, unit in unit_map.items():
            if keyword in field_name.lower():
                return f"{field_name} ({unit})"

        # No unit found, return original
        return field_name
