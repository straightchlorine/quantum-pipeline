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
    noise_backend: str
    optimizer: str
    ansatz: QuantumCircuit
    ansatz_reps: int
    default_shots: int


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
    minimization_time: np.float64


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
    molecule_id: int

    def get_result_suffix(self) -> str:
        """Return the result suffix for Avro serialization."""
        return f'_it{self.vqe_result.iteration_list.__len__()}'

    def get_schema_suffix(self) -> str:
        """Return the schema name for Avro serialization."""

        mol_str = ''
        for symbol in self.molecule.symbols:
            mol_str += symbol

        return f'_mol{self.molecule_id}_{mol_str}_it{self.vqe_result.iteration_list.__len__()}_bs_{self.basis_set.replace('-','_')}_bk_{self.vqe_result.initial_data.backend.replace('-','_')}'
