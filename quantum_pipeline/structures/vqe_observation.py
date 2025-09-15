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

    # Optional performance monitoring data
    performance_start: dict = None
    performance_end: dict = None

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

    def get_performance_delta(self) -> dict:
        """Calculate performance metrics delta between start and end snapshots."""
        if not (self.performance_start and self.performance_end):
            return {}

        try:
            delta = {}

            # System performance deltas
            start_sys = self.performance_start.get('system', {})
            end_sys = self.performance_end.get('system', {})

            # CPU metrics
            start_cpu = start_sys.get('cpu', {})
            end_cpu = end_sys.get('cpu', {})
            if start_cpu.get('percent') is not None and end_cpu.get('percent') is not None:
                delta['cpu_usage_delta'] = end_cpu['percent'] - start_cpu['percent']

            # Memory metrics
            start_mem = start_sys.get('memory', {})
            end_mem = end_sys.get('memory', {})
            if start_mem.get('used') is not None and end_mem.get('used') is not None:
                delta['memory_usage_delta'] = end_mem['used'] - start_mem['used']
                delta['memory_usage_delta_gb'] = delta['memory_usage_delta'] / (1024**3)

            # GPU metrics (if available)
            start_gpu = self.performance_start.get('gpu', [])
            end_gpu = self.performance_end.get('gpu', [])
            if start_gpu and end_gpu:
                gpu_deltas = []
                for i, (start_g, end_g) in enumerate(zip(start_gpu, end_gpu)):
                    if isinstance(start_g, dict) and isinstance(end_g, dict):
                        gpu_delta = {}
                        for metric in ['utilization_gpu', 'utilization_memory', 'power_draw']:
                            if start_g.get(metric) is not None and end_g.get(metric) is not None:
                                gpu_delta[f'{metric}_delta'] = end_g[metric] - start_g[metric]
                        if gpu_delta:
                            gpu_delta['gpu_index'] = i
                            gpu_deltas.append(gpu_delta)
                if gpu_deltas:
                    delta['gpu_deltas'] = gpu_deltas

            # Add VQE timing context
            delta['vqe_total_time'] = float(self.total_time)
            delta['container_type'] = self.performance_start.get('container_type', 'unknown')

            return delta

        except Exception as e:
            return {'error': str(e)}
