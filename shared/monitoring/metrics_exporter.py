"""Metrics export utilities for CSV generation."""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from shared.logger import setup_logger

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
        dummy_metrics = {
            "scenario_info": scenario_info,
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
            },
            "iteration_system_samples": {},
            "iteration_system_metrics": {},
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

        # Find timing indices that belong to this iteration (any operation with iteration in metadata)
        timing_indices_for_iteration: List[int] = []
        for operation, timing_data in detailed_timings.items():
            metadata_list = timing_data.get("metadata", [])
            for idx, metadata in enumerate(metadata_list):
                if metadata.get("iteration") == iteration:
                    if idx not in timing_indices_for_iteration:
                        timing_indices_for_iteration.append(idx)
        timing_indices_for_iteration.sort()

        if not timing_indices_for_iteration:
            logger.debug(f"No timing indices found for iteration {iteration}")
            return

        # Build base row (scenario only; no op_*; each row has per-iteration timing_* only)
        base_row: Dict[str, Any] = {}
        for key, value in scenario_info.items():
            base_row[f"scenario_{key}"] = self._format_value(value)

        iteration_rows = []
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
        for i in timing_indices_for_iteration:
            row = dict(base_row)
            row["timing_sample_index"] = self._format_value(i)
            # Always use per-iteration system summary for this row
            for key, value in system_summary_for_row.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"system_{key}_{sub_key}"] = self._format_value(sub_value)
                else:
                    row[f"system_{key}"] = self._format_value(value)
            # Add timing data for this index
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
            iteration_rows.append(row)

        # Append rows to CSV
        assert _active_fieldnames is not None, "Cannot append: no active CSV session"
        fieldnames: List[str] = _active_fieldnames
        with open(_active_csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for row in iteration_rows:
                complete_row = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(complete_row)

        logger.info(
            f"Appended {len(iteration_rows)} row(s) for iteration {iteration} "
            f"to {_active_csv_path}"
        )

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

        # Add scenario info
        for key, value in scenario_info.items():
            base_row[f"scenario_{key}"] = self._format_value(value)

        # Create rows for detailed timings
        if detailed_timings:
            # Find maximum number of timings across all operations
            max_timings = 0
            for operation, timing_data in detailed_timings.items():
                timings = timing_data.get("timings", [])
                max_timings = max(max_timings, len(timings))

            # Create one row per timing sample
            for i in range(max_timings):
                row = base_row.copy()
                row["timing_sample_index"] = self._format_value(i)

                # Determine iteration for this row from metadata
                iteration = None
                for operation, timing_data in detailed_timings.items():
                    metadata_list = timing_data.get("metadata", [])
                    if i < len(metadata_list):
                        metadata = metadata_list[i]
                        if "iteration" in metadata:
                            iter_val = metadata["iteration"]
                            if isinstance(iter_val, (int, float)):
                                iteration = int(iter_val)
                                break

                # Add system metrics for this iteration (or global if no iteration found)
                # Use per-iteration system metrics when available (key lookup by int)
                iteration_system_metrics = metrics_data.get(
                    "iteration_system_metrics", {}
                )
                iteration_system_samples = metrics_data.get(
                    "iteration_system_samples", {}
                )
                iteration_key = int(iteration) if iteration is not None else None
                per_iter_summary = (
                    iteration_system_metrics.get(iteration_key)
                    if iteration_key is not None
                    else None
                )

                if per_iter_summary is not None and isinstance(per_iter_summary, dict):
                    # Use per-iteration summary when available (actual values for this iteration)
                    system_summary = per_iter_summary
                    logger.debug(
                        f"Using per-iteration system metrics for iteration {iteration}"
                    )
                elif (
                    iteration_key is not None
                    and iteration_key in iteration_system_samples
                ):
                    # Iteration has samples but no precomputed summary: use global so we show real values
                    logger.debug(
                        f"Per-iteration summary missing for iteration {iteration}, "
                        "using global system summary"
                    )
                    system_summary = system_metrics
                else:
                    # No iteration or no per-iteration data: use global system metrics summary
                    if iteration is not None:
                        logger.debug(
                            f"No system samples for iteration {iteration}, "
                            "using global summary"
                        )
                    system_summary = system_metrics

                # Add system metrics to row
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
                        row[f"timing_{operation}_duration"] = timings[i]

                        # Add metadata if available
                        if i < len(metadata_list):
                            metadata = metadata_list[i]
                            for meta_key, meta_value in metadata.items():
                                row[f"timing_{operation}_{meta_key}"] = (
                                    self._format_value(meta_value)
                                )

                rows.append(row)
        else:
            # No detailed timings, just add summary row with global system metrics
            row = base_row.copy()
            system_summary = system_metrics
            for key, value in system_summary.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"system_{key}_{sub_key}"] = self._format_value(sub_value)
                else:
                    row[f"system_{key}"] = self._format_value(value)
            rows.append(row)

        return rows

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
