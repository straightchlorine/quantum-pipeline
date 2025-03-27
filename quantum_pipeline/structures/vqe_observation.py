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
        """Generate a unique schema suffix for Avro serialization based on
        molecule, iteration, basis set, and backend details.

        Returns:
            str: A formatted schema suffix string
        """
        mol_str = ''.join(self.molecule.symbols)

        basis_set_formatted = self.basis_set.replace('-', '_')
        backend_formatted = self.vqe_result.initial_data.backend.replace('-', '_')

        return (
            f'_mol{self.molecule_id}'
            f'_{mol_str}'
            f'_it{len(self.vqe_result.iteration_list)}'
            f'_bs_{basis_set_formatted}'
            f'_bk_{backend_formatted}'
        )
