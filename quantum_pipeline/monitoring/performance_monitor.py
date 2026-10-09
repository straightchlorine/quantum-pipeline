"""Performance monitoring for quantum pipeline thesis analysis.

Includes Prometheus exposition helpers as well as the PerformanceMonitor class.
Helpers don't need the monitor instance.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import requests

from quantum_pipeline.configs import settings
from quantum_pipeline.utils.logger import get_logger

PUSH_TIMEOUT_S = 10
DOCKER_TIMEOUT_S = 10
PROMETHEUS_HEADERS = {'Content-Type': 'text/plain'}

# VQE values published when present and numeric.
VQE_METRIC_NAMES = (
    'total_time',
    'hamiltonian_time',
    'mapping_time',
    'vqe_time',
    'minimum_energy',
    'iterations_count',
    'optimal_parameters_count',
    'reference_energy',
    'energy_error_hartree',
    'energy_error_millihartree',
    'hf_deviation_score',
)

VQE_LABEL_NAMES = (
    'container_type',
    'molecule_id',
    'molecule_symbols',
    'basis_set',
    'optimizer',
    'backend_type',
)

# (exposition name, snapshot section, key within that section)
SYSTEM_METRIC_SOURCES = (
    ('qp_sys_cpu_percent', 'cpu', 'percent'),
    ('qp_sys_cpu_load_1m', 'cpu', 'load_avg_1m'),
    ('qp_sys_memory_percent', 'memory', 'percent'),
    ('qp_sys_memory_used_bytes', 'memory', 'used'),
)


def _parse_bool(raw: str) -> bool:
    return raw.lower() in ('true', '1', 'yes', 'on')


def _parse_list(raw: str) -> list[str]:
    return raw.split(',') if raw else []


# attribute -> (env var, also the settings attribute)
_CONFIG: dict[str, tuple[str, Callable[[str], Any]]] = {
    'enabled': ('MONITORING_ENABLED', _parse_bool),
    'collection_interval': ('MONITORING_INTERVAL', int),
    'pushgateway_url': ('PUSHGATEWAY_URL', str),
    'export_format': ('MONITORING_EXPORT_FORMAT', _parse_list),
}

# Prometheus exposition format


def escape_label(value: Any) -> str:
    """Escape a value for use inside a Prometheus label: backslash, quote, newline."""
    return str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def format_labels(pairs: dict[str, Any]) -> str:
    """Render an ordered label set as `key="value",key="value"`."""
    return ','.join(f'{key}="{escape_label(value)}"' for key, value in pairs.items())


def format_sample(name: str, labels: str, value: Any) -> str:
    """Render one exposition line."""
    return f'{name}{{{labels}}} {value}'


def as_number(value: Any) -> float | None:
    """Return value as a float, or None when it is not a real number (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def as_divisor(value: Any) -> float | None:
    """Return value as a float only when it is a number usable as a divisor."""
    number = as_number(value)
    return number if number is not None and number > 0 else None


def derive_vqe_ratios(vqe_data: dict[str, Any]) -> dict[str, float]:
    """Efficiency ratios computed from the raw VQE timings.

    A ratio is emitted only when every input it needs is numeric and its divisor is
    positive; otherwise it is omitted rather than reported as a misleading zero.
    """
    vqe_time = as_divisor(vqe_data.get('vqe_time'))
    total_time = as_divisor(vqe_data.get('total_time'))
    iterations = as_divisor(vqe_data.get('iterations_count'))
    hamiltonian_time = as_number(vqe_data.get('hamiltonian_time'))
    mapping_time = as_number(vqe_data.get('mapping_time'))

    ratios: dict[str, float] = {}

    if vqe_time and iterations:
        ratios['iterations_per_second'] = iterations / vqe_time
        ratios['time_per_iteration'] = vqe_time / iterations

    if vqe_time and total_time:
        # setup cost relative to computation, and the share of wall time spent in VQE
        ratios['overhead_ratio'] = (total_time - vqe_time) / vqe_time
        ratios['efficiency'] = vqe_time / total_time

    if hamiltonian_time is not None and mapping_time is not None and total_time:
        ratios['setup_ratio'] = (hamiltonian_time + mapping_time) / total_time

    return ratios


def build_vqe_exposition(vqe_data: dict[str, Any], default_container_type: str) -> str:
    """Render VQE experiment data as Prometheus exposition text."""
    defaults = dict.fromkeys(VQE_LABEL_NAMES, 'unknown') | {
        'container_type': default_container_type
    }
    labels = format_labels({name: vqe_data.get(name, defaults[name]) for name in VQE_LABEL_NAMES})

    lines = [
        format_sample(f'qp_vqe_{name}', labels, vqe_data[name])
        for name in VQE_METRIC_NAMES
        if as_number(vqe_data.get(name)) is not None
    ]
    lines += [
        format_sample(f'qp_vqe_{name}', labels, value)
        for name, value in derive_vqe_ratios(vqe_data).items()
    ]

    return '\n'.join(lines) + '\n'


def build_system_exposition(
    system: dict[str, Any], container_type: str, uptime_seconds: float
) -> str:
    """Render a system metrics snapshot as Prometheus exposition text."""
    labels = format_labels({'container_type': container_type})

    lines = [
        format_sample(name, labels, system[section][key])
        for name, section, key in SYSTEM_METRIC_SOURCES
        if system.get(section, {}).get(key) is not None
    ]
    lines.append(format_sample('qp_sys_uptime_seconds', labels, uptime_seconds))

    return '\n'.join(lines) + '\n'


def counter_fields(counters: Any, *names: str) -> dict[str, int]:
    """Read psutil counter attributes, reporting zeros when unavailable on this platform."""
    return {name: getattr(counters, name, 0) if counters else 0 for name in names}


class PerformanceMonitor:
    """Background collection of system and container metrics, exported to JSONL and Prometheus."""

    def __init__(
        self,
        enabled: bool | None = None,
        collection_interval: int | None = None,
        pushgateway_url: str | None = None,
        export_format: list[str] | None = None,
        metrics_dir: Path | None = None,
    ):
        """Initialize the monitor.

        Every setting resolves by priority: constructor argument > environment
        variable > `settings.py` default.

        Args:
            enabled: Override for monitoring enabled state
            collection_interval: Metrics collection interval in seconds
            pushgateway_url: Prometheus PushGateway URL
            export_format: List of export formats ['json', 'prometheus']
            metrics_dir: Directory to store metrics files
        """
        self.logger = get_logger('PerformanceMonitor')

        self.enabled = self._resolve_config('enabled', enabled)
        self.collection_interval = self._resolve_config('collection_interval', collection_interval)
        self.pushgateway_url = self._resolve_config('pushgateway_url', pushgateway_url)
        self.export_format = self._resolve_config('export_format', export_format)
        self.metrics_dir = metrics_dir or settings.MONITORING_METRICS_DIR

        self.monitoring_thread: threading.Thread | None = None
        self.container_type = os.getenv('CONTAINER_TYPE', 'unknown')
        self.experiment_context: dict[str, Any] = {}

        self._stop_event = threading.Event()
        self._context_lock = threading.Lock()
        self._start_time = time.time()  # reference point for qp_sys_uptime_seconds

        if not self.enabled:
            self.logger.debug('Performance monitoring disabled')
            return

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Performance monitoring initialized - Container: {self.container_type}')
        self.logger.info(f'Metrics directory: {self.metrics_dir}')
        self.logger.info(f'Push gateway url: {self.pushgateway_url}')
        self.logger.info(f'Collection interval: {self.collection_interval}s')

    def _resolve_config(self, key: str, override: Any) -> Any:
        """Resolve one setting: constructor override > environment > settings.py."""
        if override is not None:
            return override

        name, parse = _CONFIG[key]
        raw = os.getenv(name)
        if raw is not None:
            try:
                return parse(raw)
            except (ValueError, AttributeError):
                self.logger.warning(f'Invalid environment variable {name}={raw}')

        return getattr(settings, name)

    def is_enabled(self) -> bool:
        """Check if performance monitoring is enabled."""
        return bool(self.enabled)

    def set_experiment_context(self, **context):
        """Merge key/value pairs into the context attached to every snapshot."""
        if not self.enabled:
            return

        with self._context_lock:
            self.experiment_context.update(context)
        self.logger.debug(f'Updated experiment context: {context}')

    # lifecycle

    def start_monitoring(self):
        """Start the background collection thread."""
        if not self.enabled:
            self.logger.debug('Monitoring not enabled - skipping start')
            return

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning('Monitoring already running')
            return

        self.logger.info('Starting performance monitoring thread')
        self._stop_event.clear()

        # non-daemon: a tick in flight finishes writing before the process exits
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name='PerformanceMonitor', daemon=False
        )
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Signal the background thread to stop and wait for it to finish."""
        if not self.enabled or not self.monitoring_thread:
            return

        self.logger.info('Stopping performance monitoring thread')
        self._stop_event.set()
        if not self.monitoring_thread.is_alive():
            return

        # allow the in-flight collection to finish before giving up
        timeout = self.collection_interval + 10
        self.logger.debug(f'Waiting up to {timeout}s for monitoring thread to stop')
        self.monitoring_thread.join(timeout=timeout)

        if self.monitoring_thread.is_alive():
            self.logger.warning(f'Monitoring thread did not stop within {timeout}s timeout')
        else:
            self.logger.info('Monitoring thread stopped successfully')

    def __enter__(self):
        if self.enabled:
            self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.stop_monitoring()

    def _monitoring_loop(self):
        """Collect and export a snapshot every interval until stopped."""
        self.logger.info(
            f'System monitoring loop started (interval: {self.collection_interval}s) '
            '- VQE metrics handled separately'
        )

        while not self._stop_event.is_set():
            try:
                metrics = self.collect_metrics_snapshot()
                if 'error' in metrics or 'error' in metrics.get('system', {}):
                    self.logger.debug('Skipping export: metrics collection reported an error')
                else:
                    self._export_system_metrics(metrics)
            except Exception as e:
                self.logger.error(f'Error in system monitoring loop: {e}')

            if self._stop_event.wait(self.collection_interval):
                break

        self.logger.info('System monitoring loop stopped')

    def _export_system_metrics(self, metrics: dict[str, Any]):
        """Fan a snapshot out to every configured export format."""
        formats = self.export_format or []

        if 'json' in formats or 'both' in formats:
            self.export_system_json(metrics)

        if 'prometheus' in formats or 'both' in formats:
            self.export_system_prometheus(metrics)

    # collection

    def collect_metrics_snapshot(self) -> dict[str, Any]:
        """Collect a single snapshot of all metrics."""
        if not self.enabled:
            return {}

        try:
            with self._context_lock:
                context = self.experiment_context.copy()

            return {
                'timestamp': datetime.now().isoformat(),
                'container_type': self.container_type,
                'experiment_context': context,
                'system': self._collect_system_metrics(),
                'container': self._collect_container_metrics(),
            }
        except Exception as e:
            self.logger.error(f'Failed to collect metrics snapshot: {e}')
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _collect_system_metrics(self) -> dict[str, Any]:
        """Collect host CPU, memory and I/O metrics via psutil."""
        try:
            try:
                load_avg = os.getloadavg()
            except (AttributeError, OSError):
                load_avg = (0.0, 0.0, 0.0)  # not available on every platform

            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1.0),
                    'count': psutil.cpu_count(),
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
                'disk_io': counter_fields(
                    psutil.disk_io_counters(),
                    'read_bytes',
                    'write_bytes',
                    'read_count',
                    'write_count',
                ),
                'network_io': counter_fields(
                    psutil.net_io_counters(),
                    'bytes_sent',
                    'bytes_recv',
                    'packets_sent',
                    'packets_recv',
                ),
            }
        except Exception as e:
            self.logger.error(f'Failed to collect system metrics: {e}')
            return {'error': str(e)}

    def _collect_container_metrics(self) -> dict[str, Any]:
        """Collect this container's row from `docker stats`, if docker is reachable."""
        container_name = os.getenv('HOSTNAME', 'unknown')
        unavailable = {'container_name': container_name, 'docker_stats_available': False}

        try:
            docker_executable = shutil.which('docker')
            if docker_executable is None:
                return unavailable

            result = subprocess.run(  # noqa: S603
                [
                    docker_executable,
                    'stats',
                    '--no-stream',
                    '--format',
                    'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}',
                ],
                capture_output=True,
                shell=False,
                timeout=DOCKER_TIMEOUT_S,
                text=True,
            )
            if result.returncode != 0:
                return unavailable

            for line in result.stdout.split('\n'):
                parts = line.split('\t')
                if container_name in line and len(parts) >= 5:
                    return {
                        'container_name': container_name,
                        'docker_stats_available': True,
                        'cpu_percent': parts[1],
                        'memory_usage': parts[2],
                        'net_io': parts[3],
                        'block_io': parts[4],
                    }

            return unavailable
        except Exception as e:
            self.logger.warning(f'Failed to collect container metrics: {e}')
            return {'error': str(e), 'container_name': container_name}

    # export

    def export_system_json(self, metrics: dict[str, Any]):
        """Append a snapshot as one JSON line to the container's JSONL file."""
        filepath = self.metrics_dir / f'system_metrics_{self.container_type.lower()}.jsonl'
        try:
            with open(filepath, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            self.logger.error(f'Failed to export system JSON metrics: {e}')

    def export_system_prometheus(self, metrics: dict[str, Any]):
        """Push a system snapshot to the PushGateway."""
        if not self.pushgateway_url:
            return

        exposition = build_system_exposition(
            metrics.get('system', {}), self.container_type, time.time() - self._start_time
        )
        job_name = f'qp-sys-{self.container_type.lower()}'
        self._push(f'{self.pushgateway_url}/metrics/job/{job_name}', exposition, 'system metrics')

    def export_vqe_metrics_immediate(self, vqe_data: dict[str, Any]):
        """Push VQE metrics to the PushGateway now, outside the collection interval."""
        if not self.enabled or not self.pushgateway_url:
            return

        formats = self.export_format or []
        if 'prometheus' not in formats and 'both' not in formats:
            return

        exposition = build_vqe_exposition(vqe_data, self.container_type)
        if not exposition.strip():
            self.logger.warning('Empty metrics payload, skipping export')
            return

        self.logger.debug(f'VQE metrics payload: {exposition[:500]}...')

        molecule = vqe_data.get('molecule_symbols', 'unknown')
        optimizer = vqe_data.get('optimizer', 'unknown')
        url = (
            f'{self.pushgateway_url}/metrics/job/qp-vqe'
            f'/container_type/{self.container_type}'
            f'/molecule/{molecule}'
            f'/optimizer/{optimizer}'
        )
        molecule_id = vqe_data.get('molecule_id', 'unknown')
        self._push(url, exposition, f'VQE metrics for molecule {molecule_id}')

    def _push(self, url: str, exposition: str, description: str):
        """POST exposition text to the PushGateway. Never raises: monitoring is best-effort."""
        try:
            response = requests.post(
                url, data=exposition, headers=PROMETHEUS_HEADERS, timeout=PUSH_TIMEOUT_S
            )
            if response.status_code in (200, 202):
                self.logger.debug(
                    f'{description} exported successfully (status {response.status_code})'
                )
            else:
                self.logger.warning(
                    f'PushGateway returned status {response.status_code} for {description}. '
                    f'Response: {response.text}'
                )
        except Exception as e:
            self.logger.error(f'Failed to export {description} to Prometheus: {e}')


# --- global instance ---
_global_monitor: PerformanceMonitor | None = None


def get_performance_monitor(**kwargs) -> PerformanceMonitor:
    """Get the global monitor, creating it from kwargs on first call."""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(**kwargs)
    elif kwargs:
        _global_monitor.logger.warning(
            'Global performance monitor already exists - ignoring arguments: '
            f'{sorted(kwargs)}. Use init_performance_monitoring() to reconfigure.'
        )

    return _global_monitor


def init_performance_monitoring(**kwargs) -> PerformanceMonitor:
    """Replace the global monitor with a freshly configured one."""
    global _global_monitor
    _global_monitor = PerformanceMonitor(**kwargs)
    return _global_monitor


def is_monitoring_enabled() -> bool:
    """Check if performance monitoring is globally enabled."""
    return get_performance_monitor().is_enabled()


def collect_performance_snapshot() -> dict[str, Any]:
    """Collect a performance metrics snapshot from the global monitor."""
    return get_performance_monitor().collect_metrics_snapshot()


def set_experiment_context(**context):
    """Set experiment context on the global monitor."""
    get_performance_monitor().set_experiment_context(**context)
