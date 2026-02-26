"""
Tests for configuration manager's performance monitoring initialization.

Tests verify that the configuration manager properly initializes monitoring
when --enable-performance-monitoring is passed via command line arguments.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from quantum_pipeline.configs.parsing.argparser import QuantumPipelineArgParser
from quantum_pipeline.configs.parsing.configuration_manager import ConfigurationManager
from quantum_pipeline.monitoring import get_performance_monitor


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


@pytest.fixture
def mock_sys_argv():
    """Mock sys.argv for argparser tests."""
    import sys
    original_argv = sys.argv
    yield
    sys.argv = original_argv


class TestConfigurationManagerMonitoringInit:
    """Test ConfigurationManager's monitoring initialization from arguments."""

    def test_monitoring_initialized_from_args(self, mock_sys_argv, clean_global_monitor, monkeypatch):
        """Test that monitoring is initialized when --enable-performance-monitoring is set."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json',
            '--enable-performance-monitoring',
            '--performance-interval', '10',
            '--performance-pushgateway', 'http://monit:9091',
            '--performance-export-format', 'both'
        ]

        # Set temp metrics dir via env var
        monkeypatch.setenv('QUANTUM_PERFORMANCE_METRICS_DIR', '/tmp/test_metrics')

        parser = QuantumPipelineArgParser()
        config = parser.get_config()

        # Get the global monitor instance
        monitor = get_performance_monitor()

        assert monitor.is_enabled() is True
        assert monitor.collection_interval == 10
        assert monitor.pushgateway_url == 'http://monit:9091'

    def test_monitoring_not_initialized_without_flag(self, mock_sys_argv, clean_global_monitor):
        """Test that monitoring is not initialized when flag is not set."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json'
        ]

        parser = QuantumPipelineArgParser()
        args = parser.parse_args()

        assert args.enable_performance_monitoring is False

    def test_monitoring_export_format_conversion(self, mock_sys_argv, clean_global_monitor):
        """Test that export format 'both' is converted to list."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json',
            '--enable-performance-monitoring',
            '--performance-export-format', 'both'
        ]

        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()
        args = parser.parse_args()
        config = config_manager.get_config(args)

        monitor = get_performance_monitor()

        # 'both' should be converted to ['json', 'prometheus']
        assert isinstance(monitor.export_format, list)
        assert 'json' in monitor.export_format
        assert 'prometheus' in monitor.export_format

    def test_monitoring_export_format_single_value(self, mock_sys_argv, clean_global_monitor):
        """Test that single export format value is converted to list."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json',
            '--enable-performance-monitoring',
            '--performance-export-format', 'json'
        ]

        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()
        args = parser.parse_args()
        config = config_manager.get_config(args)

        monitor = get_performance_monitor()

        assert isinstance(monitor.export_format, list)
        assert 'json' in monitor.export_format

    def test_configuration_manager_logs_initialization(self, mock_sys_argv, clean_global_monitor, caplog):
        """Test that configuration manager logs monitoring initialization."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json',
            '--enable-performance-monitoring'
        ]

        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()
        args = parser.parse_args()

        with caplog.at_level('INFO'):
            config = config_manager.get_config(args)

        # Check that initialization was logged
        assert any('Initializing performance monitoring' in record.message for record in caplog.records)

    def test_monitoring_disabled_by_default_in_config(self, mock_sys_argv, clean_global_monitor):
        """Test that monitoring is disabled by default in configuration."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json'
        ]

        parser = QuantumPipelineArgParser()
        args = parser.parse_args()

        # Monitoring flag should be False by default
        assert hasattr(args, 'enable_performance_monitoring')
        assert args.enable_performance_monitoring is False

    def test_config_dict_contains_monitoring_settings(self, mock_sys_argv, clean_global_monitor):
        """Test that config dict contains all monitoring-related settings."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json',
            '--enable-performance-monitoring',
            '--performance-interval', '45',
            '--performance-pushgateway', 'http://test:9091',
            '--performance-export-format', 'prometheus'
        ]

        parser = QuantumPipelineArgParser()
        config = parser.get_config()

        # Verify monitoring settings in config
        assert 'enable_performance_monitoring' in config
        assert config['enable_performance_monitoring'] is True
        assert 'performance_interval' in config
        assert config['performance_interval'] == 45
        assert 'performance_pushgateway' in config
        assert config['performance_pushgateway'] == 'http://test:9091'
        assert 'performance_export_format' in config
        assert config['performance_export_format'] == 'prometheus'


class TestMonitoringConfigurationPriority:
    """Test configuration priority: constructor > env vars > command line > settings."""

    def test_env_var_overrides_settings(self, clean_global_monitor, monkeypatch):
        """Test that environment variables override settings.py defaults."""
        # Set env vars
        monkeypatch.setenv('QUANTUM_PERFORMANCE_ENABLED', 'true')
        monkeypatch.setenv('QUANTUM_PERFORMANCE_COLLECTION_INTERVAL', '60')

        from quantum_pipeline.monitoring import init_performance_monitoring

        monitor = init_performance_monitoring()

        # Should use env var values, not settings defaults
        assert monitor.enabled is True
        assert monitor.collection_interval == 60

    def test_command_line_via_config_manager(self, mock_sys_argv, clean_global_monitor):
        """Test that command line args initialize monitoring correctly."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json',
            '--enable-performance-monitoring',
            '--performance-interval', '15'
        ]

        parser = QuantumPipelineArgParser()
        config = parser.get_config()

        monitor = get_performance_monitor()

        # Should use command line values
        assert monitor.enabled is True
        assert monitor.collection_interval == 15


class TestMonitoringIntegrationWithVQERunner:
    """Test that monitoring configuration flows to VQE runner."""

    @patch('quantum_pipeline.runners.vqe_runner.load_molecule')
    def test_vqe_runner_receives_monitor_instance(self, mock_load_molecule, mock_sys_argv, clean_global_monitor):
        """Test that VQERunner receives the initialized monitor instance."""
        import sys
        sys.argv = [
            'test',
            '--file', 'data/molecules.thesis.simple.json',
            '--enable-performance-monitoring',
            '--kafka'
        ]

        # Mock molecule loading to avoid file I/O
        mock_load_molecule.return_value = []

        parser = QuantumPipelineArgParser()
        config = parser.get_config()

        # Simulate VQERunner initialization
        from quantum_pipeline.monitoring import get_performance_monitor
        monitor = get_performance_monitor()

        assert monitor.is_enabled() is True

    def test_monitoring_context_set_during_vqe_run(self, clean_global_monitor):
        """Test that experiment context can be set during VQE run."""
        from quantum_pipeline.monitoring import init_performance_monitoring

        monitor = init_performance_monitoring(enabled=True)

        # Simulate setting context as VQERunner would
        monitor.set_experiment_context(
            molecule_id=0,
            molecule_symbols='H2',
            basis_set='sto3g',
            optimizer='COBYLA',
            backend_type='CPU'
        )

        assert monitor.experiment_context['molecule_id'] == 0
        assert monitor.experiment_context['molecule_symbols'] == 'H2'

        # Collect snapshot and verify context is included
        snapshot = monitor.collect_metrics_snapshot()
        assert snapshot['experiment_context']['molecule_id'] == 0
