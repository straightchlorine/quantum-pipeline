from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo


@dataclass
class VQEInitialData:
    """Dataclass for storing VQE initial data."""

    backend: str
    num_qubits: int
    hamiltonian: np.ndarray
    num_parameters: int
    initial_parameters: np.ndarray
    optimizer: str
    ansatz: QuantumCircuit
    ansatz_reps: int


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

    initial_data: VQEInitialData
    iteration_list: list[VQEProcess]
    minimum: np.float64
    optimal_parameters: np.ndarray
    maxcv: np.float64


@dataclass
class VQEDecoratedResult:
    """Dataclass for storing VQE result data with additional information."""

    vqe_result: VQEResult
    molecule: MoleculeInfo
    basis_set: str
    hamiltonian_time: np.float64
    mapping_time: np.float64
    vqe_time: np.float64
    total_time: np.float64
    id: int
