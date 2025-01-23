from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict

from qiskit_ibm_runtime import IBMBackend


@dataclass
class BackendConfig:
    """Dataclass for storing backend filter."""

    local: bool | None
    optimization_level: int | None
    min_num_qubits: int | None
    filters: Callable[[IBMBackend], bool] | None

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackendConfig':
        """Create a BackendConfig instance from a dictionary."""
        return cls(
            local=data.get('local'),
            optimization_level=data.get('optimization_level'),
            min_num_qubits=data.get('min_num_qubits'),
            filters=None,
        )

    @classmethod
    def default_backend_config(cls) -> 'BackendConfig':
        """Return the default backend configuration."""
        return cls(
            local=True,
            optimization_level=3,
            min_num_qubits=None,
            filters=None,
        )
