from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from qiskit_ibm_runtime import IBMBackend

from quantum_pipeline.configs.defaults import DEFAULTS


@dataclass
class BackendConfig:
    """Dataclass for storing backend filter."""

    local: bool | None
    gpu: bool | None
    optimization_level: int | None
    min_num_qubits: int | None
    filters: Callable[[IBMBackend], bool] | None
    simulation_method: str | None
    gpu_opts: dict[str, Any] | None
    noise: str | None

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BackendConfig':
        """Create a BackendConfig instance from a dictionary."""
        return cls(
            local=data.get('local'),
            optimization_level=data.get('optimization_level'),
            min_num_qubits=data.get('min_num_qubits'),
            filters=None,
            gpu=data.get('gpu'),
            simulation_method=data.get('simulation_method'),
            gpu_opts=data.get('gpu_opts'),
            noise=data.get('noise'),
        )

    @classmethod
    def default_backend_config(cls) -> 'BackendConfig':
        """Return the default backend configuration."""
        return cls(
            local=DEFAULTS['backend']['local'],
            optimization_level=DEFAULTS['backend']['optimization_level'],
            min_num_qubits=DEFAULTS['backend']['min_qubits'],
            filters=DEFAULTS['backend']['filters'],
            gpu=DEFAULTS['backend']['gpu'],
            simulation_method=DEFAULTS['backend']['method'],
            gpu_opts=DEFAULTS['backend']['gpu_opts'],
            noise=DEFAULTS['backend']['noise_backend'],
        )
