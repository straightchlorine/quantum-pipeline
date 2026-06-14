"""Regression tests: the same seed must produce bit-identical VQE runs.

Two VQESolver instances with identical configuration and the same seed must
yield the same energy trajectory and the same final result.

Uses a real AerSimulator on a tiny 2-qubit Hamiltonian (no PySCF needed),
so the two runs complete in seconds.
"""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.vqe_solver import VQESolver


def _statevector_backend_config():
    """A real (non-mock) local statevector backend configuration."""
    return BackendConfig(
        local=True,
        gpu=False,
        optimization_level=2,
        min_num_qubits=None,
        filters=None,
        simulation_method='statevector',
        gpu_opts=None,
        noise=None,
    )


def _solve(seed):
    solver = VQESolver(
        qubit_op=SparsePauliOp.from_list([('ZZ', 1.0), ('XI', 0.5), ('IX', 0.5)]),
        backend_config=_statevector_backend_config(),
        max_iterations=15,
        optimizer='COBYLA',
        ansatz_reps=1,
        default_shots=512,
        seed=seed,
    )
    return solver.solve()


@pytest.mark.slow
@pytest.mark.parametrize('seed', [0, 7])  # 0 is the regression-prone value
def test_same_seed_runs_are_bit_identical(seed):
    first = _solve(seed)
    second = _solve(seed)

    assert float(first.minimum) == float(second.minimum)
    assert len(first.iteration_list) == len(second.iteration_list)
    for a, b in zip(first.iteration_list, second.iteration_list, strict=True):
        assert float(a.result) == float(b.result)
    np.testing.assert_array_equal(first.optimal_parameters, second.optimal_parameters)


@pytest.mark.slow
def test_different_seeds_differ():
    """Sanity check: seeding must not collapse distinct seeds onto one trajectory."""
    energies_a = [float(p.result) for p in _solve(0).iteration_list]
    energies_b = [float(p.result) for p in _solve(1).iteration_list]
    assert energies_a != energies_b
