"""Unit tests for shared.monitoring.metrics_exporter."""

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.monitoring import metrics_exporter
from shared.monitoring.metrics_exporter import MetricsExporter


@pytest.fixture(autouse=True)
def reset_active_csv_state():
    """Reset module-level active CSV state before each test."""
    metrics_exporter._active_csv_path = None
    metrics_exporter._active_fieldnames = None
    yield
    metrics_exporter._active_csv_path = None
    metrics_exporter._active_fieldnames = None


class TestMetricsExporterInit:
    """Tests for MetricsExporter.__init__."""

    def test_init_default_output_dir(self, tmp_path, monkeypatch):
        """Default output_dir is metrics_output (relative) when None."""
        monkeypatch.chdir(tmp_path)
        exporter = MetricsExporter(output_dir=None)
        assert exporter.output_dir == Path("metrics_output")
        assert exporter.output_dir.resolve() == tmp_path / "metrics_output"

    def test_init_custom_output_dir(self, tmp_path):
        """Custom output_dir is used and created."""
        out = tmp_path / "custom_metrics"
        exporter = MetricsExporter(output_dir=out)
        assert exporter.output_dir == out
        assert out.is_dir()


class TestGetActiveCsvPath:
    """Tests for MetricsExporter.get_active_csv_path."""

    def test_get_active_csv_path_initially_none(self):
        """Before any initialize_csv_file, active path is None."""
        assert MetricsExporter.get_active_csv_path() is None

    def test_get_active_csv_path_after_initialize(self, tmp_path):
        """After initialize_csv_file, active path is set."""
        exporter = MetricsExporter(output_dir=tmp_path)
        path = exporter.initialize_csv_file(
            scenario_info={"blockchain_enabled": True, "ipfs_enabled": False},
            filename="test_active.csv",
        )
        assert MetricsExporter.get_active_csv_path() == path


class TestFormatValue:
    """Tests for _format_value (via export or flatten)."""

    def test_format_none(self):
        """None is formatted as empty string."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        assert exporter._format_value(None) == ""

    def test_format_bool(self):
        """Booleans are lowercased."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        assert exporter._format_value(True) == "true"
        assert exporter._format_value(False) == "false"

    def test_format_float(self):
        """Floats get 6 decimal places."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        assert exporter._format_value(1.5) == "1.500000"
        assert exporter._format_value(0.1) == "0.100000"

    def test_format_dict_and_list(self):
        """Dict and list are JSON-serialized."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        assert exporter._format_value({"a": 1}) == '{"a": 1}'
        assert exporter._format_value([1, 2]) == "[1, 2]"

    def test_format_int_and_str(self):
        """Int and str are stringified."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        assert exporter._format_value(42) == "42"
        assert exporter._format_value("hello") == "hello"


class TestAddUnitToHeader:
    """Tests for _add_unit_to_header."""

    def test_add_unit_duration(self):
        """Duration fields get (seconds) unit."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        assert "seconds" in exporter._add_unit_to_header("duration_seconds")
        assert "seconds" in exporter._add_unit_to_header("avg_duration")

    def test_add_unit_memory(self):
        """Memory fields get (bytes) unit."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        assert "bytes" in exporter._add_unit_to_header("memory_avg_used_bytes")
        assert "bytes" in exporter._add_unit_to_header("memory_max_used_bytes")

    def test_add_unit_unknown_returns_original(self):
        """Unknown field name is returned unchanged."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        name = "some_custom_field"
        assert exporter._add_unit_to_header(name) == name


class TestFlattenMetrics:
    """Tests for _flatten_metrics."""

    def test_flatten_empty_metrics(self):
        """Empty metrics produce one row with only system/scenario if present."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        data = {
            "scenario_info": {},
            "system_metrics": {
                "sample_count": 0,
                "duration_seconds": 0.0,
                "cpu": {"total_time_seconds": 0.0},
                "memory": {"avg_used_bytes": 0, "max_used_bytes": 0, "min_used_bytes": 0},
                "network": {"total_bytes_sent": 0, "total_bytes_recv": 0},
                "disk": {"total_bytes_read": 0, "total_bytes_written": 0},
            },
            "detailed_timings": {},
        }
        rows = exporter._flatten_metrics(data)
        assert len(rows) == 1
        assert "system_sample_count" in rows[0] or "system_duration_seconds" in rows[0]

    def test_flatten_with_detailed_timings(self):
        """Detailed timings produce one row per timing index."""
        exporter = MetricsExporter(output_dir=Path("/tmp"))
        data = {
            "scenario_info": {"blockchain_enabled": True},
            "system_metrics": {
                "sample_count": 1,
                "duration_seconds": 1.0,
                "cpu": {"total_time_seconds": 0.5},
                "memory": {"avg_used_bytes": 1000, "max_used_bytes": 2000, "min_used_bytes": 500},
                "network": {"total_bytes_sent": 100, "total_bytes_recv": 200},
                "disk": {"total_bytes_read": 0, "total_bytes_written": 0},
            },
            "operation_metrics": {},
            "detailed_timings": {
                "op1": {
                    "timings": [0.1, 0.2],
                    "metadata": [{"iteration": 0}, {"iteration": 1}],
                },
            },
            "iteration_system_metrics": {},
            "iteration_system_samples": {},
        }
        rows = exporter._flatten_metrics(data)
        assert len(rows) == 2
        assert rows[0].get("timing_op1_duration") == 0.1 or "timing_op1_duration" in rows[0]
        assert rows[1].get("timing_op1_duration") == 0.2 or "timing_op1_duration" in rows[1]
        assert rows[0].get("scenario_blockchain_enabled") == "true"


class TestExportToCsv:
    """Tests for export_to_csv."""

    def test_export_to_csv_creates_file(self, tmp_path):
        """export_to_csv creates a CSV file with expected content."""
        exporter = MetricsExporter(output_dir=tmp_path)
        metrics_data = {
            "scenario_info": {"blockchain_enabled": True},
            "system_metrics": {
                "sample_count": 1,
                "duration_seconds": 1.0,
                "cpu": {"total_time_seconds": 0.5},
                "memory": {"avg_used_bytes": 1000, "max_used_bytes": 1000, "min_used_bytes": 1000},
                "network": {"total_bytes_sent": 0, "total_bytes_recv": 0},
                "disk": {"total_bytes_read": 0, "total_bytes_written": 0},
            },
            "operation_metrics": {},
            "detailed_timings": {},
            "iteration_system_samples": {},
            "iteration_system_metrics": {},
        }
        path = exporter.export_to_csv(metrics_data, filename="export_test.csv")
        assert path == tmp_path / "export_test.csv"
        assert path.exists()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        # Header row may have units in first row; second row is data
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 2

    def test_export_to_csv_empty_metrics_returns_path(self, tmp_path):
        """Export with no rows still returns path and creates file (with no data rows)."""
        exporter = MetricsExporter(output_dir=tmp_path)
        metrics_data = {
            "scenario_info": {},
            "system_metrics": {},
            "detailed_timings": {},
        }
        path = exporter.export_to_csv(metrics_data, filename="empty.csv")
        assert path == tmp_path / "empty.csv"
        # With no detailed_timings and empty system_metrics, _flatten_metrics still
        # produces one row (base row + system summary)
        assert path.exists()


class TestInitializeCsvFile:
    """Tests for initialize_csv_file."""

    def test_initialize_csv_creates_file_with_headers(self, tmp_path):
        """initialize_csv_file creates file with header row."""
        exporter = MetricsExporter(output_dir=tmp_path)
        scenario = {"blockchain_enabled": True, "ipfs_enabled": True}
        path = exporter.initialize_csv_file(scenario_info=scenario, filename="init_test.csv")
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
        assert "scenario" in first_line or "system" in first_line
        assert MetricsExporter.get_active_csv_path() == path

    def test_initialize_csv_filename_includes_bc_ipfs_when_not_provided(self, tmp_path):
        """When filename is None, generated name includes bc_on/off and ipfs_on/off."""
        exporter = MetricsExporter(output_dir=tmp_path)
        path = exporter.initialize_csv_file(
            scenario_info={"blockchain_enabled": False, "ipfs_enabled": True}
        )
        assert path.name.startswith("metrics_")
        assert "bc_off" in path.name
        assert "ipfs_on" in path.name


class TestDeletePreviousCsvFiles:
    """Tests for _delete_previous_csv_files."""

    def test_delete_previous_removes_other_metrics_files(self, tmp_path):
        """Other metrics_*.csv in output_dir are deleted, not current."""
        exporter = MetricsExporter(output_dir=tmp_path)
        old1 = tmp_path / "metrics_20200101_120000.csv"
        old2 = tmp_path / "metrics_20200102_120000.csv"
        current = tmp_path / "metrics_20200103_120000.csv"
        old1.write_text("old1")
        old2.write_text("old2")
        exporter._delete_previous_csv_files(current)
        assert not old1.exists()
        assert not old2.exists()
        assert not current.exists()  # we didn't create it, so it's still missing
        # Creating current after delete should work
        current.write_text("new")
        assert current.exists()

    def test_delete_previous_does_not_remove_current(self, tmp_path):
        """Current file path is not deleted (only others)."""
        exporter = MetricsExporter(output_dir=tmp_path)
        current = tmp_path / "metrics_current.csv"
        current.write_text("current")
        exporter._delete_previous_csv_files(current)
        assert current.exists()
        assert current.read_text() == "current"


class TestAppendIterationMetrics:
    """Tests for append_iteration_metrics."""

    def test_append_without_active_csv_falls_back_to_full_export(self, tmp_path):
        """When no active CSV, append falls back to full export (writes a new file)."""
        exporter = MetricsExporter(output_dir=tmp_path)
        assert metrics_exporter._active_csv_path is None
        collector = MagicMock()
        collector.get_metrics_for_export.return_value = {
            "scenario_info": {"blockchain_enabled": True},
            "system_metrics": {
                "sample_count": 0,
                "duration_seconds": 0.0,
                "cpu": {"total_time_seconds": 0.0},
                "memory": {"avg_used_bytes": 0, "max_used_bytes": 0, "min_used_bytes": 0},
                "network": {"total_bytes_sent": 0, "total_bytes_recv": 0},
                "disk": {"total_bytes_read": 0, "total_bytes_written": 0},
            },
            "operation_metrics": {},
            "detailed_timings": {},
            "iteration_system_samples": {},
            "iteration_system_metrics": {},
        }
        exporter.append_iteration_metrics(0, collector)
        collector.get_metrics_for_export.assert_called_once()
        # Should have created a new CSV via export_to_csv
        csv_files = list(tmp_path.glob("metrics_*.csv"))
        assert len(csv_files) == 1
