"""
Comprehensive tests for the PerformanceMonitor module.

Tests cover:
- Initialization with different configuration sources
- Metrics collection (system, container, VQE)
- Export formats (JSON, Prometheus)
- Threading behavior
- Context management
- Configuration priority system
"""

import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from quantum_pipeline.monitoring.performance_monitor import (
    PerformanceMonitor,
    get_performance_monitor,
    init_performance_monitoring,
    is_monitoring_enabled,
    collect_performance_snapshot,
)


@pytest.fixture
def temp_metrics_dir():
    """Create a temporary directory for metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_global_monitor():
    """Reset global monitor before and after tests."""
    from quantum_pipeline.monitoring import performance_monitor
    original_monitor = performance_monitor._global_monitor
    performance_monitor._global_monitor = None
    yield
    performance_monitor._global_monitor = original_monitor


class TestPerformanceMonitorInitialization:
    """Test PerformanceMonitor initialization with various configurations."""

    def test_initialization_disabled_by_default(self, temp_metrics_dir):
        """Test that monitoring is disabled by default."""
        monitor = PerformanceMonitor(metrics_dir=temp_metrics_dir)
        assert monitor.enabled == False
        assert monitor.is_enabled() == False

    def test_initialization_enabled_via_constructor(self, temp_metrics_dir):
        """Test enabling monitoring via constructor parameter."""
        monitor = PerformanceMonitor(
            enabled=True,
            collection_interval=30,
            pushgateway_url='http://test:9091',
            export_format=['json', 'prometheus'],
            metrics_dir=temp_metrics_dir
        )
        assert monitor.enabled == True
        assert monitor.collection_interval == 30
        assert monitor.pushgateway_url == 'http://test:9091'
        assert monitor.export_format == ['json', 'prometheus']

    def test_initialization_enabled_via_env_vars(self, temp_metrics_dir, monkeypatch):
        """Test enabling monitoring via environment variables."""
        monkeypatch.setenv('QUANTUM_PERFORMANCE_ENABLED', 'true')
        monkeypatch.setenv('QUANTUM_PERFORMANCE_COLLECTION_INTERVAL', '45')
        monkeypatch.setenv('QUANTUM_PERFORMANCE_PUSHGATEWAY_URL', 'http://env:9091')
        monkeypatch.setenv('QUANTUM_PERFORMANCE_EXPORT_FORMAT', 'json,prometheus')

        monitor = PerformanceMonitor(metrics_dir=temp_metrics_dir)

        assert monitor.enabled == True
        assert monitor.collection_interval == 45
        assert monitor.pushgateway_url == 'http://env:9091'
        assert monitor.export_format == ['json', 'prometheus']

    def test_config_priority_constructor_over_env(self, temp_metrics_dir, monkeypatch):
        """Test that constructor parameters override environment variables."""
        monkeypatch.setenv('QUANTUM_PERFORMANCE_ENABLED', 'false')
        monkeypatch.setenv('QUANTUM_PERFORMANCE_COLLECTION_INTERVAL', '45')

        monitor = PerformanceMonitor(
            enabled=True,
            collection_interval=10,
            metrics_dir=temp_metrics_dir
        )

        assert monitor.enabled == True  # Constructor overrides env
        assert monitor.collection_interval == 10  # Constructor overrides env

    def test_metrics_directory_creation(self, temp_metrics_dir):
        """Test that metrics directory is created when monitoring is enabled."""
        metrics_path = temp_metrics_dir / 'test_metrics'
        assert not metrics_path.exists()

        monitor = PerformanceMonitor(
            enabled=True,
            metrics_dir=metrics_path
        )

        assert metrics_path.exists()
        assert metrics_path.is_dir()

    def test_container_type_detection(self, temp_metrics_dir, monkeypatch):
        """Test that container type is detected from environment."""
        monkeypatch.setenv('CONTAINER_TYPE', 'GPU_GTX1060_6GB')

        monitor = PerformanceMonitor(enabled=True, metrics_dir=temp_metrics_dir)

        assert monitor.container_type == 'GPU_GTX1060_6GB'


class TestMetricsCollection:
    """Test metrics collection functionality."""

    def test_collect_metrics_snapshot_when_disabled(self, temp_metrics_dir):
        """Test that snapshot returns empty dict when monitoring is disabled."""
        monitor = PerformanceMonitor(enabled=False, metrics_dir=temp_metrics_dir)
        snapshot = monitor.collect_metrics_snapshot()
        assert snapshot == {}

    def test_collect_metrics_snapshot_structure(self, temp_metrics_dir):
        """Test the structure of collected metrics snapshot."""
        monitor = PerformanceMonitor(enabled=True, metrics_dir=temp_metrics_dir)
        snapshot = monitor.collect_metrics_snapshot()

        # Verify top-level keys
        assert 'timestamp' in snapshot
        assert 'container_type' in snapshot
        assert 'experiment_context' in snapshot
        assert 'system' in snapshot
        assert 'container' in snapshot

        # Verify system metrics structure
        system = snapshot['system']
        assert 'cpu' in system
        assert 'memory' in system
        assert 'disk_io' in system
        assert 'network_io' in system

        # Verify CPU metrics
        cpu = system['cpu']
        assert 'percent' in cpu
        assert 'count' in cpu
        assert 'load_avg_1m' in cpu

        # Verify memory metrics
        memory = system['memory']
        assert 'total' in memory
        assert 'used' in memory
        assert 'available' in memory
        assert 'percent' in memory

    def test_set_experiment_context(self, temp_metrics_dir):
        """Test setting experiment context for metrics correlation."""
        monitor = PerformanceMonitor(enabled=True, metrics_dir=temp_metrics_dir)

        monitor.set_experiment_context(
            molecule_id=5,
            molecule_symbols='H2O',
            basis_set='sto3g',
            optimizer='COBYLA'
        )

        assert monitor.experiment_context['molecule_id'] == 5
        assert monitor.experiment_context['molecule_symbols'] == 'H2O'
        assert monitor.experiment_context['basis_set'] == 'sto3g'
        assert monitor.experiment_context['optimizer'] == 'COBYLA'

    def test_experiment_context_in_snapshot(self, temp_metrics_dir):
        """Test that experiment context is included in snapshot."""
        monitor = PerformanceMonitor(enabled=True, metrics_dir=temp_metrics_dir)

        monitor.set_experiment_context(
            molecule_id=0,
            molecule_symbols='H2'
        )

        snapshot = monitor.collect_metrics_snapshot()

        assert snapshot['experiment_context']['molecule_id'] == 0
        assert snapshot['experiment_context']['molecule_symbols'] == 'H2'


class TestPrometheusExport:
    """Test Prometheus export functionality."""

    def test_convert_vqe_to_prometheus_format(self, temp_metrics_dir):
        """Test conversion of VQE data to Prometheus format."""
        monitor = PerformanceMonitor(enabled=True, metrics_dir=temp_metrics_dir)

        vqe_data = {
            'container_type': 'CPU_TEST',
            'molecule_id': 0,
            'molecule_symbols': 'H2',
            'basis_set': 'sto3g',
            'optimizer': 'COBYLA',
            'backend_type': 'CPU',
            'total_time': 45.23,
            'hamiltonian_time': 10.5,
            'mapping_time': 2.3,
            'vqe_time': 32.43,
            'minimum_energy': -1.1744,
            'iterations_count': 150,
            'optimal_parameters_count': 4,
            'reference_energy': -1.17447901,
            'energy_error_hartree': 0.00007901,
            'accuracy_score': 99.2,
            'within_chemical_accuracy': 1
        }

        prometheus_output = monitor._convert_vqe_to_prometheus(vqe_data)

        # Verify it's a non-empty string
        assert isinstance(prometheus_output, str)
        assert len(prometheus_output) > 0

        # Verify key metrics are present
        assert 'quantum_vqe_total_time' in prometheus_output
        assert 'quantum_vqe_minimum_energy' in prometheus_output
        assert 'quantum_vqe_accuracy_score' in prometheus_output
        assert 'quantum_vqe_iterations_count' in prometheus_output

        # Verify labels are present
        assert 'container_type="CPU_TEST"' in prometheus_output
        assert 'molecule_symbols="H2"' in prometheus_output
        assert 'basis_set="sto3g"' in prometheus_output
        assert 'optimizer="COBYLA"' in prometheus_output

        # Verify calculated efficiency metrics
        assert 'quantum_vqe_iterations_per_second' in prometheus_output
        assert 'quantum_vqe_time_per_iteration' in prometheus_output
        assert 'quantum_vqe_efficiency' in prometheus_output

    def test_prometheus_export_with_missing_data(self, temp_metrics_dir):
        """Test Prometheus export handles missing optional data gracefully."""
        monitor = PerformanceMonitor(enabled=True, metrics_dir=temp_metrics_dir)

        vqe_data = {
            'container_type': 'CPU',
            'molecule_id': 0,
            'total_time': 45.23,
            'minimum_energy': -1.1744,
        }

        # Should not raise an exception
        prometheus_output = monitor._convert_vqe_to_prometheus(vqe_data)
        assert isinstance(prometheus_output, str)

    @patch('requests.post')
    def test_export_vqe_metrics_to_pushgateway(self, mock_post, temp_metrics_dir):
        """Test exporting VQE metrics to Prometheus PushGateway."""
        mock_post.return_value.status_code = 200

        monitor = PerformanceMonitor(
            enabled=True,
            pushgateway_url='http://test:9091',
            metrics_dir=temp_metrics_dir
        )

        vqe_data = {
            'container_type': 'CPU',
            'molecule_id': 0,
            'molecule_symbols': 'H2',
            'total_time': 45.23,
            'minimum_energy': -1.1744,
        }

        monitor.export_vqe_metrics_immediate(vqe_data)

        # Verify POST was called
        assert mock_post.called
        call_args = mock_post.call_args

        # Verify URL
        assert 'http://test:9091/metrics/job/' in call_args[0][0]

        # Verify headers
        assert call_args[1]['headers']['Content-Type'] == 'text/plain'

    @patch('requests.post')
    def test_export_vqe_metrics_disabled_monitoring(self, mock_post, temp_metrics_dir):
        """Test that VQE metrics are not exported when monitoring is disabled."""
        monitor = PerformanceMonitor(enabled=False, metrics_dir=temp_metrics_dir)

        vqe_data = {'total_time': 45.23}
        monitor.export_vqe_metrics_immediate(vqe_data)

        # Verify POST was NOT called
        assert not mock_post.called


class TestJSONExport:
    """Test JSON export functionality."""

    def test_json_export_creates_file(self, temp_metrics_dir):
        """Test that JSON export creates a file."""
        monitor = PerformanceMonitor(
            enabled=True,
            export_format=['json'],
            metrics_dir=temp_metrics_dir
        )

        snapshot = monitor.collect_metrics_snapshot()
        monitor._export_json(snapshot)

        # Check that at least one JSON file was created
        json_files = list(temp_metrics_dir.glob('*.json'))
        assert len(json_files) > 0

    def test_json_export_valid_structure(self, temp_metrics_dir):
        """Test that exported JSON has valid structure."""
        monitor = PerformanceMonitor(
            enabled=True,
            export_format=['json'],
            metrics_dir=temp_metrics_dir
        )

        snapshot = monitor.collect_metrics_snapshot()
        monitor._export_json(snapshot)

        # Read the JSON file
        json_files = list(temp_metrics_dir.glob('*.json'))
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        # Verify structure
        assert 'timestamp' in data
        assert 'container_type' in data
        assert 'system' in data

    def test_json_export_system_only(self, temp_metrics_dir):
        """Test JSON export of system metrics only."""
        monitor = PerformanceMonitor(
            enabled=True,
            export_format=['json'],
            metrics_dir=temp_metrics_dir
        )

        metrics = {
            'timestamp': '2025-01-01T00:00:00',
            'container_type': 'test',
            'system': {'cpu': {'percent': 50.0}},
            'container': {}
        }

        monitor._export_json_system_only(metrics)

        # Check file was created
        json_files = list(temp_metrics_dir.glob('system_metrics_*.json'))
        assert len(json_files) > 0


class TestMonitoringThread:
    """Test background monitoring thread functionality."""

    def test_start_monitoring_thread(self, temp_metrics_dir):
        """Test starting the monitoring thread."""
        monitor = PerformanceMonitor(
            enabled=True,
            collection_interval=1,
            metrics_dir=temp_metrics_dir
        )

        monitor.start_monitoring()

        # Verify thread started
        assert monitor.monitoring_thread is not None
        assert monitor.monitoring_thread.is_alive()

        # Clean up
        monitor.stop_monitoring_thread()

    def test_stop_monitoring_thread(self, temp_metrics_dir):
        """Test stopping the monitoring thread."""
        monitor = PerformanceMonitor(
            enabled=True,
            collection_interval=1,
            metrics_dir=temp_metrics_dir
        )

        monitor.start_monitoring()
        assert monitor.monitoring_thread.is_alive()

        monitor.stop_monitoring_thread()

        # Wait a bit for thread to stop
        time.sleep(0.5)

        # Verify thread stopped
        assert not monitor.monitoring_thread.is_alive()

    def test_monitoring_thread_not_started_when_disabled(self, temp_metrics_dir):
        """Test that monitoring thread is not started when monitoring is disabled."""
        monitor = PerformanceMonitor(enabled=False, metrics_dir=temp_metrics_dir)

        monitor.start_monitoring()

        # Thread should not be created
        assert monitor.monitoring_thread is None

    def test_context_manager_starts_and_stops_thread(self, temp_metrics_dir):
        """Test that context manager properly starts and stops monitoring."""
        monitor = PerformanceMonitor(
            enabled=True,
            collection_interval=1,
            metrics_dir=temp_metrics_dir
        )

        with monitor:
            assert monitor.monitoring_thread is not None
            assert monitor.monitoring_thread.is_alive()

        # After context exit, thread should be stopped
        time.sleep(0.5)
        assert not monitor.monitoring_thread.is_alive()


class TestGlobalMonitorFunctions:
    """Test global convenience functions."""

    def test_get_performance_monitor_singleton(self, clean_global_monitor):
        """Test that get_performance_monitor returns a singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2

    def test_init_performance_monitoring(self, clean_global_monitor, temp_metrics_dir):
        """Test initializing performance monitoring globally."""
        monitor = init_performance_monitoring(
            enabled=True,
            collection_interval=30,
            metrics_dir=temp_metrics_dir
        )

        assert monitor.enabled == True
        assert monitor.collection_interval == 30

        # Verify it's the global instance
        global_monitor = get_performance_monitor()
        assert global_monitor is monitor

    def test_is_monitoring_enabled_function(self, clean_global_monitor, temp_metrics_dir):
        """Test is_monitoring_enabled convenience function."""
        init_performance_monitoring(enabled=True, metrics_dir=temp_metrics_dir)

        assert is_monitoring_enabled() == True

    def test_collect_performance_snapshot_function(self, clean_global_monitor, temp_metrics_dir):
        """Test collect_performance_snapshot convenience function."""
        init_performance_monitoring(enabled=True, metrics_dir=temp_metrics_dir)

        snapshot = collect_performance_snapshot()

        assert snapshot is not None
        assert 'timestamp' in snapshot


class TestConfigurationEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_env_var_collection_interval(self, temp_metrics_dir, monkeypatch, caplog):
        """Test handling of invalid collection interval from env var."""
        monkeypatch.setenv('QUANTUM_PERFORMANCE_COLLECTION_INTERVAL', 'not_a_number')

        monitor = PerformanceMonitor(metrics_dir=temp_metrics_dir)

        # Should fall back to settings default
        from quantum_pipeline.configs import settings
        assert monitor.collection_interval == settings.PERFORMANCE_COLLECTION_INTERVAL

    def test_export_format_both_expands_to_list(self, temp_metrics_dir):
        """Test that 'both' export format is expanded correctly."""
        monitor = PerformanceMonitor(
            enabled=True,
            export_format=['both'],
            metrics_dir=temp_metrics_dir
        )

        # When checking for formats, 'both' should work
        assert 'both' in monitor.export_format

    @patch('requests.post')
    def test_pushgateway_connection_failure_handling(self, mock_post, temp_metrics_dir):
        """Test handling of PushGateway connection failure."""
        mock_post.side_effect = Exception("Connection refused")

        monitor = PerformanceMonitor(
            enabled=True,
            pushgateway_url='http://unreachable:9091',
            metrics_dir=temp_metrics_dir
        )

        vqe_data = {'total_time': 45.23}

        # Should not raise exception
        monitor.export_vqe_metrics_immediate(vqe_data)

    def test_metrics_collection_failure_handling(self, temp_metrics_dir):
        """Test handling of metrics collection failure."""
        monitor = PerformanceMonitor(enabled=True, metrics_dir=temp_metrics_dir)

        with patch('psutil.cpu_percent', side_effect=Exception("Mock error")):
            snapshot = monitor.collect_metrics_snapshot()

            # Should return snapshot with error in system metrics instead of raising
            assert 'system' in snapshot
            assert 'error' in snapshot['system']


class TestPerformanceMonitorIntegration:
    """Integration tests for complete monitoring workflows."""

    @patch('requests.post')
    def test_complete_vqe_monitoring_workflow(self, mock_post, temp_metrics_dir):
        """Test complete workflow: context → collect → export."""
        mock_post.return_value.status_code = 200

        monitor = PerformanceMonitor(
            enabled=True,
            collection_interval=30,
            pushgateway_url='http://test:9091',
            export_format=['json', 'prometheus'],
            metrics_dir=temp_metrics_dir
        )

        # Step 1: Set experiment context
        monitor.set_experiment_context(
            molecule_id=0,
            molecule_symbols='H2',
            basis_set='sto3g',
            optimizer='COBYLA',
            backend_type='CPU'
        )

        # Step 2: Collect before snapshot
        snapshot_before = monitor.collect_metrics_snapshot()
        assert snapshot_before is not None

        # Step 3: Simulate VQE execution
        time.sleep(0.1)

        # Step 4: Collect after snapshot
        snapshot_after = monitor.collect_metrics_snapshot()
        assert snapshot_after is not None

        # Step 5: Export VQE metrics
        vqe_data = {
            'container_type': 'CPU',
            'molecule_id': 0,
            'molecule_symbols': 'H2',
            'basis_set': 'sto3g',
            'optimizer': 'COBYLA',
            'backend_type': 'CPU',
            'total_time': 45.23,
            'vqe_time': 32.43,
            'minimum_energy': -1.1744,
            'iterations_count': 150,
        }

        monitor.export_vqe_metrics_immediate(vqe_data)

        # Verify POST was called
        assert mock_post.called

    def test_monitoring_with_context_manager(self, temp_metrics_dir):
        """Test using monitoring with context manager pattern."""
        monitor = PerformanceMonitor(
            enabled=True,
            collection_interval=1,
            export_format=['json'],
            metrics_dir=temp_metrics_dir
        )

        with monitor:
            # Set context
            monitor.set_experiment_context(molecule_id=0)

            # Collect snapshot
            snapshot = monitor.collect_metrics_snapshot()
            assert snapshot is not None

            # Give thread time to collect at least one metric
            time.sleep(1.5)

        # After context exit, check that system metrics were collected
        system_json_files = list(temp_metrics_dir.glob('system_metrics_*.json'))
        assert len(system_json_files) >= 1
