"""
Performance monitoring module for quantum pipeline thesis analysis.

This module provides comprehensive system and application performance monitoring
that can be completely switched on/off via settings, command line args, or env vars.
"""

import json
import os
import psutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests

from quantum_pipeline.configs import settings
from quantum_pipeline.utils.logger import get_logger


class PerformanceMonitor:
    """
    Modular performance monitoring system for quantum pipeline.

    Features:
    - System resource monitoring (CPU, Memory, I/O)
    - GPU metrics collection (when available)
    - Container-specific metrics
    - Prometheus integration
    - JSON export for analysis
    - Thread-safe collection
    - Configurable intervals
    """

    def __init__(
        self,
        enabled: bool = None,
        collection_interval: int = None,
        pushgateway_url: str = None,
        export_format: List[str] = None,
        metrics_dir: Path = None,
    ):
        """
        Initialize performance monitor with configuration.

        Args:
            enabled: Override for monitoring enabled state
            collection_interval: Metrics collection interval in seconds
            pushgateway_url: Prometheus PushGateway URL
            export_format: List of export formats ['json', 'prometheus']
            metrics_dir: Directory to store metrics files
        """
        self.logger = get_logger('PerformanceMonitor')

        # Configuration priority: constructor > env vars > command line > settings.py
        self.enabled = self._get_config_value('enabled', enabled, bool)
        self.collection_interval = self._get_config_value(
            'collection_interval', collection_interval, int
        )
        self.pushgateway_url = self._get_config_value('pushgateway_url', pushgateway_url, str)
        self.export_format = self._get_config_value('export_format', export_format, list)
        self.metrics_dir = metrics_dir or settings.PERFORMANCE_METRICS_DIR

        # Runtime state
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.container_type = os.getenv('CONTAINER_TYPE', 'unknown')
        self.experiment_context = {}

        # Ensure metrics directory exists
        if self.enabled:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f'Performance monitoring initialized - Container: {self.container_type}'
            )
            self.logger.info(f'Metrics directory: {self.metrics_dir}')
            self.logger.info(f'Push gateway url: {self.pushgateway_url}')
            self.logger.info(f'Collection interval: {self.collection_interval}s')
        else:
            self.logger.debug('Performance monitoring disabled')

    def _get_config_value(self, key: str, override_value: Any, expected_type: type) -> Any:
        """Get configuration value with priority: override > env > settings."""

        # Priority 1: Override parameter
        if override_value is not None:
            return override_value

        # Priority 2: Environment variable
        env_key = f'QUANTUM_PERFORMANCE_{key.upper()}'
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                if expected_type == bool:
                    return env_value.lower() in ('true', '1', 'yes', 'on')
                elif expected_type == int:
                    return int(env_value)
                elif expected_type == list:
                    return env_value.split(',') if env_value else []
                else:
                    return env_value
            except (ValueError, AttributeError):
                self.logger.warning(f'Invalid environment variable {env_key}={env_value}')

        # Priority 3: Settings.py defaults
        settings_map = {
            'enabled': settings.PERFORMANCE_MONITORING_ENABLED,
            'collection_interval': settings.PERFORMANCE_COLLECTION_INTERVAL,
            'pushgateway_url': settings.PERFORMANCE_PUSHGATEWAY_URL,
            'export_format': settings.PERFORMANCE_EXPORT_FORMAT,
        }

        return settings_map.get(key)

    def is_enabled(self) -> bool:
        """Check if performance monitoring is enabled."""
        return self.enabled

    def set_experiment_context(self, **context):
        """Set experiment context for correlation with metrics."""
        if self.enabled:
            self.experiment_context.update(context)
            self.logger.debug(f'Updated experiment context: {context}')

    def start_monitoring(self):
        """Start background monitoring thread."""
        if not self.enabled:
            self.logger.debug('Monitoring not enabled - skipping start')
            return

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning('Monitoring already running')
            return

        self.logger.info('Starting performance monitoring thread')
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name='PerformanceMonitor', daemon=True
        )
        self.monitoring_thread.start()

    def stop_monitoring_thread(self):
        """Stop background monitoring thread."""
        if not self.enabled or not self.monitoring_thread:
            return

        self.logger.info('Stopping performance monitoring thread')
        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

    def collect_metrics_snapshot(self) -> Dict[str, Any]:
        """Collect a single snapshot of all metrics."""
        if not self.enabled:
            return {}

        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'container_type': self.container_type,
                'experiment_context': self.experiment_context.copy(),
                # System metrics
                'system': self._collect_system_metrics(),
                # Container metrics
                'container': self._collect_container_metrics(),
            }

            # GPU metrics (if available)
            gpu_metrics = self._collect_gpu_metrics()
            if gpu_metrics:
                metrics['gpu'] = gpu_metrics

            return metrics

        except Exception as e:
            self.logger.error(f'Failed to collect metrics snapshot: {e}')
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # I/O metrics
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg_1m': load_avg[0],
                    'load_avg_5m': load_avg[1],
                    'load_avg_15m': load_avg[2],
                },
                'memory': {
                    'total': memory.total,
                    'used': memory.used,
                    'available': memory.available,
                    'percent': memory.percent,
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'percent': swap.percent,
                },
                'disk_io': {
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0,
                    'read_count': disk_io.read_count if disk_io else 0,
                    'write_count': disk_io.write_count if disk_io else 0,
                },
                'network_io': {
                    'bytes_sent': net_io.bytes_sent if net_io else 0,
                    'bytes_recv': net_io.bytes_recv if net_io else 0,
                    'packets_sent': net_io.packets_sent if net_io else 0,
                    'packets_recv': net_io.packets_recv if net_io else 0,
                },
            }
        except Exception as e:
            self.logger.error(f'Failed to collect system metrics: {e}')
            return {'error': str(e)}

    def _collect_gpu_metrics(self) -> Optional[List[Dict[str, Any]]]:
        """Collect GPU metrics using nvidia-smi."""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,clocks.current.graphics,clocks.current.memory',
                    '--format=csv,nounits,noheader',
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return None

            gpu_data = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        try:
                            # Helper function to safely parse numeric values
                            def safe_float(value):
                                if value in ('N/A', '[N/A]', 'N/A]', '[N/A'):
                                    return None
                                try:
                                    return float(value)
                                except ValueError:
                                    return None

                            gpu_info = {
                                'index': int(parts[0]),
                                'name': parts[1],
                                'temperature': safe_float(parts[2]),
                                'utilization_gpu': safe_float(parts[3]),
                                'utilization_memory': safe_float(parts[4]),
                                'memory_total': safe_float(parts[5]),
                                'memory_used': safe_float(parts[6]),
                                'power_draw': safe_float(parts[7]),
                            }

                            # Optional fields
                            if len(parts) >= 9:
                                gpu_info['clock_graphics'] = safe_float(parts[8])
                            if len(parts) >= 10:
                                gpu_info['clock_memory'] = safe_float(parts[9])

                            gpu_data.append(gpu_info)
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Failed to parse GPU line '{line}': {e}")
                            continue

            return gpu_data if gpu_data else None

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # nvidia-smi not available or failed
            return None
        except Exception as e:
            self.logger.error(f'Unexpected error collecting GPU metrics: {e}')
            return None

    def _collect_container_metrics(self) -> Dict[str, Any]:
        """Collect Docker container-specific metrics."""
        try:
            container_name = os.getenv('HOSTNAME', 'unknown')

            # Try to get Docker stats
            result = subprocess.run(
                [
                    'docker',
                    'stats',
                    '--no-stream',
                    '--format',
                    'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}',
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {'container_name': container_name, 'docker_stats_available': False}

            # Parse Docker stats output
            for line in result.stdout.split('\n'):
                if container_name in line:
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        return {
                            'container_name': container_name,
                            'docker_stats_available': True,
                            'cpu_percent': parts[1],
                            'memory_usage': parts[2],
                            'net_io': parts[3],
                            'block_io': parts[4],
                        }

            return {'container_name': container_name, 'docker_stats_available': False}

        except Exception as e:
            return {'error': str(e), 'container_name': os.getenv('HOSTNAME', 'unknown')}

    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        self.logger.info(
            f'Performance monitoring loop started (interval: {self.collection_interval}s)'
        )

        while not self.stop_monitoring.wait(self.collection_interval):
            try:
                metrics = self.collect_metrics_snapshot()
                if not metrics or 'error' in metrics:
                    continue

                # Export metrics in requested formats
                if 'json' in self.export_format or 'both' in self.export_format:
                    self._export_json(metrics)

                if 'prometheus' in self.export_format or 'both' in self.export_format:
                    self._export_prometheus(metrics)

            except Exception as e:
                self.logger.error(f'Error in monitoring loop: {e}')

        self.logger.info('Performance monitoring loop stopped')

    def _export_json(self, metrics: Dict[str, Any]):
        """Export metrics to JSON file."""
        try:
            timestamp = int(time.time())
            filename = f'metrics_{self.container_type.lower()}_{timestamp}.json'
            filepath = self.metrics_dir / filename

            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            self.logger.error(f'Failed to export JSON metrics: {e}')

    def _export_prometheus(self, metrics: Dict[str, Any]):
        """Export metrics to Prometheus PushGateway."""
        try:
            prometheus_metrics = self._convert_to_prometheus_format(metrics)

            job_name = f'quantum-{self.container_type.lower()}'
            url = f'{self.pushgateway_url}/metrics/job/{job_name}'

            response = requests.post(
                url, data=prometheus_metrics, headers={'Content-Type': 'text/plain'}, timeout=10
            )

            if response.status_code != 202:
                self.logger.warning(f'PushGateway returned status {response.status_code}')

        except Exception as e:
            self.logger.error(f'Failed to export Prometheus metrics: {e}')

    def _convert_to_prometheus_format(self, metrics: Dict[str, Any]) -> str:
        """Convert metrics dict to Prometheus exposition format."""
        lines = []
        try:
            # System metrics
            system = metrics.get('system', {})

            # CPU metrics
            cpu = system.get('cpu', {})
            if cpu.get('percent') is not None:
                lines.append(
                    f'quantum_cpu_percent{{container_type="{self.container_type}"}} {cpu["percent"]}'
                )
            if cpu.get('load_avg_1m') is not None:
                lines.append(
                    f'quantum_cpu_load_1m{{container_type="{self.container_type}"}} {cpu["load_avg_1m"]}'
                )

            # Memory metrics
            memory = system.get('memory', {})
            if memory.get('percent') is not None:
                lines.append(
                    f'quantum_memory_percent{{container_type="{self.container_type}"}} {memory["percent"]}'
                )
            if memory.get('used') is not None:
                lines.append(
                    f'quantum_memory_used_bytes{{container_type="{self.container_type}"}} {memory["used"]}'
                )

            # GPU metrics
            gpu_data = metrics.get('gpu', [])
            if isinstance(gpu_data, list):
                for gpu in gpu_data:
                    gpu_id = gpu.get('index', 0)
                    gpu_name = gpu.get('name', 'unknown')
                    gpu_name_escaped = gpu_name.replace('\\', '\\\\').replace('"', '\\"')

                    for metric_name, value in gpu.items():
                        if isinstance(value, (int, float)) and value is not None:
                            prometheus_name = f'quantum_gpu_{metric_name}'
                            lines.append(
                                f'{prometheus_name}{{container_type="{self.container_type}",gpu="{gpu_id}",name="{gpu_name_escaped}"}} {value}'
                            )

            # Experiment context
            context = metrics.get('experiment_context', {})
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    lines.append(
                        f'quantum_experiment_{key}{{container_type="{self.container_type}"}} {value}'
                    )

            return '\n'.join(lines) + '\n'  # PushGateway requires trailing newline

        except Exception as e:
            self.logger.error(f'Failed to convert metrics to Prometheus format: {e}')
            return ''

    def __enter__(self):
        """Context manager entry."""
        if self.enabled:
            self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.enabled:
            self.stop_monitoring_thread()


# Global instance for easy access
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(**kwargs) -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(**kwargs)

    return _global_monitor


def init_performance_monitoring(**kwargs):
    """Initialize global performance monitoring."""
    global _global_monitor
    _global_monitor = PerformanceMonitor(**kwargs)
    return _global_monitor


# Convenience functions
def is_monitoring_enabled() -> bool:
    """Check if performance monitoring is globally enabled."""
    monitor = get_performance_monitor()
    return monitor.is_enabled()


def collect_performance_snapshot() -> Dict[str, Any]:
    """Collect a performance metrics snapshot."""
    monitor = get_performance_monitor()
    return monitor.collect_metrics_snapshot()


def set_experiment_context(**context):
    """Set experiment context for performance correlation."""
    monitor = get_performance_monitor()
    monitor.set_experiment_context(**context)
