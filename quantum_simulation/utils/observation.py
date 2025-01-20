from typing import Callable
import numpy as np
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime.qiskit_runtime_service import IBMBackend
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo


@dataclass
class BackendConfig:
    """Dataclass for storing backend filter."""

    local: bool | None
    min_num_qubits: int | None
    filters: Callable[[IBMBackend], bool] | None

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None and key != 'local'
        }

    def local_backend(self):
        """Check if the backend is local."""
        return self.local

    @classmethod
    def _get_local_backend(cls):
        return cls(local=True, min_num_qubits=None, filters=None)


@dataclass
class VQEInitialData:
    """Dataclass for storing VQE initial data."""

    backend: str
    num_qubits: int
    hamiltonian: np.ndarray
    num_parameters: int
    initial_parameters: np.ndarray
    optimizer: str
    basis_set: str
    ansatz: QuantumCircuit


@dataclass
class VQEProcess:
    """Dataclass for storing VQE process data."""

    iteration: int
    parameters: np.ndarray
    result: np.float64
    std: np.float64


@dataclass
class VQEResult:
    """Dataclass for storing VQE result data."""

    vqe_initial_data: VQEInitialData
    vqe_iterations: list[VQEProcess]
    minimum: np.float64
    iterations: int
    optimal_parameters: np.ndarray
    maxcv: np.float64


@dataclass
class VQEDecoratedResult:
    """Dataclass for storing VQE result data with additional information."""

    vqe_result: VQEResult
    molecule: MoleculeInfo
    hamiltonian_time: np.float64
    mapping_time: np.float64
    vqe_time: np.float64
    total_time: np.float64
    id: int


@dataclass
class ObservationData:
    """Dataclass for storing observation data."""

    time: float
    state: np.ndarray
    probability: float
