from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo


@dataclass
class VQEInitialData:
    backend: str
    num_qubits: int
    hamiltonian: np.ndarray
    num_parameters: int
    initial_parameters: np.ndarray
    noise_backend: str
    optimizer: str
    ansatz: QuantumCircuit
    ansatz_reps: int
    default_shots: int | None
    seed: int | None = None
    init_strategy: str = 'random'
    ansatz_name: str = 'EfficientSU2'
    exact_estimator: bool = False


@dataclass
class VQEProcess:
    iteration: int
    parameters: np.ndarray
    result: np.float64
    std: np.float64
    energy_delta: np.float64 | None = None
    parameter_delta_norm: np.float64 | None = None
    cumulative_min_energy: np.float64 | None = None


@dataclass
class VQEResult:
    initial_data: VQEInitialData
    iteration_list: list[VQEProcess]
    minimum: np.float64
    optimal_parameters: np.ndarray
    maxcv: np.float64 | None
    minimization_time: np.float64
    nuclear_repulsion_energy: np.float64 | None = None
    success: bool | None = None
    nfev: int | None = None
    nit: int | None = None

    @property
    def total_energy(self) -> np.float64:
        """Total energy = electronic (VQE minimum) + nuclear repulsion."""
        if self.nuclear_repulsion_energy is not None:
            return self.minimum + self.nuclear_repulsion_energy
        return self.minimum


@dataclass
class VQEDecoratedResult:
    vqe_result: VQEResult
    molecule: MoleculeInfo
    basis_set: str
    hamiltonian_time: np.float64
    mapping_time: np.float64
    vqe_time: np.float64
    total_time: np.float64
    molecule_id: int

    # performance monitoring data (optional)
    performance_start: dict | None = None
    performance_end: dict | None = None

    def get_performance_delta(self) -> dict:
        """Calculate performance metrics delta between start and end snapshots."""
        if not (self.performance_start and self.performance_end):
            return {}

        try:
            delta = {}

            # system performance
            start_sys = self.performance_start.get('system', {})
            end_sys = self.performance_end.get('system', {})

            # cpu metrics
            start_cpu = start_sys.get('cpu', {})
            end_cpu = end_sys.get('cpu', {})
            if start_cpu.get('percent') is not None and end_cpu.get('percent') is not None:
                delta['cpu_usage_delta'] = end_cpu['percent'] - start_cpu['percent']

            # memory metrics
            start_mem = start_sys.get('memory', {})
            end_mem = end_sys.get('memory', {})
            if start_mem.get('used') is not None and end_mem.get('used') is not None:
                delta['memory_usage_delta'] = end_mem['used'] - start_mem['used']
                delta['memory_usage_delta_gb'] = delta['memory_usage_delta'] / (1024**3)

            # gpu metrics
            start_gpu = self.performance_start.get('gpu', [])
            end_gpu = self.performance_end.get('gpu', [])
            if start_gpu and end_gpu:
                gpu_deltas = []
                for i, (start_g, end_g) in enumerate(zip(start_gpu, end_gpu, strict=False)):
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

            # vqe timing
            delta['vqe_total_time'] = float(self.total_time)
            delta['container_type'] = self.performance_start.get('container_type', 'unknown')

            return delta

        except Exception as e:
            return {'error': str(e)}
