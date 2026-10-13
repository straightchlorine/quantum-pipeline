"""Analytic-gradient statevector path: numerical equivalence with Aer and the eligibility gate."""

import importlib.util
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerBackend
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from scipy.optimize import minimize

from quantum_pipeline.circuits import HFData
from quantum_pipeline.configs.constants import (
    FAST_PATH_BACKEND,
    FAST_PATH_MAX_QUBITS,
    GRADIENT_OPTIMIZERS,
)
from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.mappers import JordanWignerMapper
from quantum_pipeline.solvers.optimizer_config import OptimizerConfigFactory
from quantum_pipeline.solvers.vqe_solver import InvalidSolverConfigError, VQESolver

# One SCF for the whole module. Nothing downstream mutates what it returns: the qubit operator
# and the HF data are read-only inputs, so module scope cannot leak state between tests.
H2_STO3G = MoleculeInfo(symbols=['H', 'H'], coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.7414)])

# Enough operator for the gate tests. They never build a circuit, and a real SCF would
# cost 0.2 s to decide a branch that does not depend on it.
TWO_QUBIT_OP = SparsePauliOp(['ZZ', 'XI'], coeffs=[1.0, 0.5])

# Every stochastic input in this file is pinned. This one draws the parameter points for the
# energy-equivalence test.
RNG = np.random.default_rng(0)

H2_FCI_ELECTRONIC = -1.8510242
H2_FCI_TOTAL = -1.1372702
H2_NUCLEAR_REPULSION = 0.7137539936876182


def _backend_config(**overrides):
    # BackendConfig has no field defaults, so all eight are given. local, gpu and noise are
    # the three the gate reads; the rest do not matter on the analytic path.
    fields = {
        'local': True,
        'gpu': False,
        'optimization_level': 1,
        'min_num_qubits': 4,
        'filters': None,
        'simulation_method': 'statevector',
        'gpu_opts': None,
        'noise': None,
    }
    fields.update(overrides)
    return BackendConfig(**fields)


@pytest.fixture(scope='module')
def h2_cell():
    problem = PySCFDriver.from_molecule(H2_STO3G, basis='sto3g').run()
    mapper = JordanWignerMapper()
    return SimpleNamespace(
        qubit_op=mapper.map(problem.second_q_ops()[0]),
        mapper=mapper,
        hf_data=HFData(
            num_particles=problem.num_particles,
            num_spatial_orbitals=problem.num_spatial_orbitals,
            reference_energy=problem.reference_energy,
            nuclear_repulsion_energy=problem.nuclear_repulsion_energy,
        ),
    )


@pytest.fixture
def backend_config():
    return _backend_config()


def _fast_solver(qubit_op, backend_config, **overrides):
    # Fast-path defaults, the inverse of the Aer-path test helper _make_solver (COBYLA, shots=64,
    # max_iterations=3). Duplicated on purpose: importing a private helper across test modules
    # would couple two files that are otherwise independent.
    defaults = {
        'qubit_op': qubit_op,
        'backend_config': backend_config,
        'max_iterations': 300,
        'optimizer': 'L-BFGS-B',
        'ansatz_reps': 3,
        'default_shots': None,
        'convergence_threshold': None,
        'optimization_level': 1,
        'seed': 4,
    }
    defaults.update(overrides)
    return VQESolver(**defaults)


@pytest.fixture(scope='module')
def h2_fast_result(h2_cell):
    # The only full optimisation in this file (1-2 s). Two tests read it, so it runs once.
    solver = _fast_solver(h2_cell.qubit_op, _backend_config(), hf_data=h2_cell.hf_data)
    return solver.via_statevector()


def test_fast_path_energy_matches_aer_exact(h2_cell, backend_config):
    """Statevector on the logical circuit == AerEstimatorV2(default_precision=0.0) on the ISA
    circuit, to 1e-10 Ha, at five parameter points drawn as
    2 * np.pi * RNG.random((5, ansatz.num_parameters)). Measured 1.5e-14 on LiH and worst case
    7.3e-14 at 14 qubits (Appendix A); the tolerance is 3 decades looser so a BLAS change cannot
    turn this red."""
    solver = _fast_solver(h2_cell.qubit_op, backend_config)
    ansatz = solver._build_ansatz(4)
    backend = solver.get_backend()
    isa, hisa = solver._optimize_circuits(ansatz, h2_cell.qubit_op, backend)
    est = AerEstimatorV2.from_backend(backend, options={'default_precision': 0.0})

    for x in 2 * np.pi * RNG.random((5, ansatz.num_parameters)):
        fast_energy, _ = solver._fast_path_objective(x, ansatz, h2_cell.qubit_op)
        aer_energy = float(solver.compute_energy(x, isa, hisa, est))
        assert abs(fast_energy - aer_energy) < 1e-10


def test_analytic_gradient_matches_finite_differences(h2_cell, backend_config):
    """Every component of the ReverseEstimatorGradient result agrees with a central difference at
    h=1e-5 to 2e-7 Ha/rad. This also pins the index order: the gradient is ordered like
    circuit.parameters, which is the order assign_parameters(sequence) binds, so a silent
    re-ordering would break here before it corrupted a trajectory. Measured 2.7e-11 at 4 qubits
    (Appendix A)."""
    solver = _fast_solver(h2_cell.qubit_op, backend_config)
    ansatz = solver._build_ansatz(4)
    x = 2 * np.pi * np.random.default_rng(1).random(ansatz.num_parameters)
    h = 1e-5

    _, gradient = solver._fast_path_objective(x, ansatz, h2_cell.qubit_op)

    for i in range(ansatz.num_parameters):
        step = np.zeros_like(x)
        step[i] = h
        plus, _ = solver._fast_path_objective(x + step, ansatz, h2_cell.qubit_op)
        minus, _ = solver._fast_path_objective(x - step, ansatz, h2_cell.qubit_op)
        assert abs(gradient[i] - (plus - minus) / (2 * h)) < 2e-7, i


def test_h2_sto3g_reaches_fci_total_energy(h2_fast_result):
    """H2/sto-3g at the shipped 0.7414 Angstrom geometry, EfficientSU2 reps=3, seed=4, L-BFGS-B,
    max_iterations=300: electronic -1.8510242 Ha and total -1.1372702 Ha to 1e-6, nfev <= 150. Asserts
    result.nuclear_repulsion_energy == pytest.approx(0.7137539936876182) before the total, so a missing
    hf_data fails on the offset rather than on the sum. Seed 4 is the cheapest seed that escapes the HF trap
    (Appendix A seed scan); the bound is an inequality because step counts are host-specific while the
    energies are not. FIXES.md:119's -1.137283 is the 0.74 Angstrom value; do not use it. The bound is 150,
    not a tight 60: Appendix A records a 1.75x cross-host swing on other seeds, and 150 is still half the
    301-call truncation and 6x under the 924 finite-difference calls, so the assertion keeps its meaning."""
    result = h2_fast_result
    assert float(result.minimum) == pytest.approx(H2_FCI_ELECTRONIC, abs=1e-6)
    assert result.nuclear_repulsion_energy == pytest.approx(H2_NUCLEAR_REPULSION)
    assert float(result.total_energy) == pytest.approx(H2_FCI_TOTAL, abs=1e-6)
    assert result.nfev is not None and result.nfev <= 150


def test_fast_path_records_one_row_per_objective_call(h2_cell, h2_fast_result):
    """len(result.iteration_list) == result.nfev, iterations are 1..nfev, every std is 0.0, and
    every result is electronic (all rows >= the exact electronic minimum - 1e-9)."""
    result = h2_fast_result
    rows = result.iteration_list
    # Lowest eigenvalue over every electron count: a lower bound no physical row can cross.
    exact_minimum = np.linalg.eigvalsh(h2_cell.qubit_op.to_matrix()).min()

    assert len(rows) == result.nfev
    assert [row.iteration for row in rows] == list(range(1, result.nfev + 1))
    assert all(float(row.std) == 0.0 for row in rows)
    assert all(float(row.result) >= exact_minimum - 1e-9 for row in rows)
    # A row with the nuclear offset added would sit near the total energy, not below -1.8 Ha.
    assert min(float(row.result) for row in rows) < H2_FCI_TOTAL - 0.5


def test_fast_path_labels_initial_data(h2_cell, h2_fast_result):
    """initial_data names the analytic path and holds the logical ansatz and Hamiltonian,
    not transpiled copies, and the run is marked exact."""
    data = h2_fast_result.initial_data
    assert data.backend == FAST_PATH_BACKEND
    assert data.exact_estimator is True
    assert data.ansatz.layout is None
    assert data.hamiltonian == h2_cell.qubit_op.to_list()


def test_fast_path_stops_at_max_iterations(h2_cell, backend_config):
    """max_iterations=5 stops the analytic path after exactly 5 objective calls. The partial
    result is kept: success is False, nit and nfev are unknown, and the minimum is the best row."""
    solver = _fast_solver(h2_cell.qubit_op, backend_config, max_iterations=5)
    result = solver.via_statevector()

    assert len(result.iteration_list) == 5
    assert result.success is False
    assert result.nit is None and result.nfev is None
    assert float(result.minimum) == min(float(row.result) for row in result.iteration_list)


@pytest.mark.parametrize(
    'overrides, expect_fast',
    [
        ({'default_shots': None, 'optimizer': 'L-BFGS-B'}, True),
        ({'default_shots': None, 'optimizer': 'SLSQP'}, True),
        ({'default_shots': None, 'optimizer': 'COBYLA'}, False),
        ({'default_shots': 1024, 'optimizer': 'COBYLA'}, False),
        # Sampled shots with a gradient optimizer: only the shots clause rejects this row.
        ({'default_shots': 1024, 'optimizer': 'L-BFGS-B'}, False),
    ],
)
def test_gate_selects_the_path(overrides, expect_fast, backend_config):
    """_fast_path_eligible() accepts exact, local, noiseless runs with a gradient optimizer and
    rejects the rest. Every row is a configuration a user can express on the CLI (D2)."""
    solver = _fast_solver(TWO_QUBIT_OP, backend_config, **overrides)
    assert solver._fast_path_eligible() is expect_fast


def _route(solver, backend):
    # Run solve() with every path and get_backend patched; return the three path mocks.
    with (
        patch.object(VQESolver, 'get_backend', return_value=backend),
        patch.object(VQESolver, 'via_statevector') as via_statevector,
        patch.object(VQESolver, 'via_aer') as via_aer,
        patch.object(VQESolver, 'via_ibmq') as via_ibmq,
    ):
        solver.solve()
    return via_statevector, via_aer, via_ibmq


def test_gate_rejects_noise_model(backend_config):
    """backend_config.noise='ibm_brisbane' with default_shots=None stays on the Aer path: the
    analytic statevector cannot represent a noise model, and --exact --noise --simulation-method
    density_matrix is a legitimate configuration (argparser.py:405-410)."""
    assert _fast_solver(TWO_QUBIT_OP, backend_config)._fast_path_eligible() is True
    solver = _fast_solver(TWO_QUBIT_OP, replace(backend_config, noise='ibm_brisbane'))
    assert solver._fast_path_eligible() is False

    via_statevector, via_aer, via_ibmq = _route(solver, MagicMock(spec=AerBackend))
    via_aer.assert_called_once()
    via_statevector.assert_not_called()
    via_ibmq.assert_not_called()


def test_gate_rejects_remote_backend(backend_config):
    """backend_config.local=False stays on the IBMQ path. get_backend is patched to return a
    non-Aer mock so via_ibmq is the asserted branch and no QiskitRuntimeService is created."""
    solver = _fast_solver(TWO_QUBIT_OP, replace(backend_config, local=False))
    assert solver._fast_path_eligible() is False

    via_statevector, via_aer, via_ibmq = _route(solver, MagicMock())
    via_ibmq.assert_called_once()
    via_statevector.assert_not_called()
    via_aer.assert_not_called()


def test_gate_rejects_unset_local(backend_config):
    """backend_config.local=None means the user did not ask for a local run, so the gate
    rejects it. Only local=True is eligible."""
    solver = _fast_solver(TWO_QUBIT_OP, replace(backend_config, local=None))
    assert solver._fast_path_eligible() is False


def test_gate_rejects_gpu_request(backend_config):
    """backend_config.gpu=True with default_shots=None and L-BFGS-B stays on the Aer path (D2)."""
    solver = _fast_solver(TWO_QUBIT_OP, replace(backend_config, gpu=True))
    assert solver._fast_path_eligible() is False

    via_statevector, via_aer, via_ibmq = _route(solver, MagicMock(spec=AerBackend))
    via_aer.assert_called_once()
    via_statevector.assert_not_called()
    via_ibmq.assert_not_called()


def test_gate_rejects_problems_above_qubit_cap(backend_config):
    """Above FAST_PATH_MAX_QUBITS the run stays on the Aer path. ReverseEstimatorGradient
    builds a sparse Hamiltonian per call, and H2 cc-pVDZ (20 qubits) was OOM-killed there."""
    at_cap = SparsePauliOp(['Z' * FAST_PATH_MAX_QUBITS])
    above_cap = SparsePauliOp(['Z' * (FAST_PATH_MAX_QUBITS + 1)])
    assert _fast_solver(at_cap, backend_config)._fast_path_eligible() is True
    solver = _fast_solver(above_cap, backend_config)
    assert solver._fast_path_eligible() is False

    via_statevector, via_aer, via_ibmq = _route(solver, MagicMock(spec=AerBackend))
    via_aer.assert_called_once()
    via_statevector.assert_not_called()
    via_ibmq.assert_not_called()


def test_solve_sends_eligible_run_to_statevector(backend_config):
    """An exact, local, noiseless L-BFGS-B run takes the analytic path and never asks for a
    backend, so no transpile or Aer work happens."""
    solver = _fast_solver(TWO_QUBIT_OP, backend_config)
    with (
        patch.object(VQESolver, 'get_backend') as get_backend,
        patch.object(VQESolver, 'via_statevector') as via_statevector,
        patch.object(VQESolver, 'via_aer') as via_aer,
        patch.object(VQESolver, 'via_ibmq') as via_ibmq,
    ):
        solver.solve()
    via_statevector.assert_called_once()
    via_aer.assert_not_called()
    via_ibmq.assert_not_called()
    get_backend.assert_not_called()


def test_solve_sends_gradient_free_run_to_aer(backend_config):
    """An exact local COBYLA run is not eligible, so it stays on the Aer path."""
    solver = _fast_solver(TWO_QUBIT_OP, backend_config, optimizer='COBYLA')
    via_statevector, via_aer, via_ibmq = _route(solver, MagicMock(spec=AerBackend))
    via_aer.assert_called_once()
    via_statevector.assert_not_called()
    via_ibmq.assert_not_called()


def test_sampled_shots_with_gradient_optimizer_raises(backend_config):
    """solve() raises InvalidSolverConfigError matching 'gradient-free optimizer' for every optimizer
    in GRADIENT_OPTIMIZERS when default_shots is not None, and does not raise for COBYLA. The
    exception is asserted to be a ValueError subclass, because the CLI contract is a ValueError.
    The COBYLA leg patches via_aer and get_backend; it asserts no raise and one via_aer call.
    Pins the defect measured today at seed 42, max_iterations=50: nit=0, nfev=33, success=True,
    minimum=-0.9156149 (Appendix A)."""
    assert issubclass(InvalidSolverConfigError, ValueError)

    # Newton-CG gets its own, more specific message (test_newton_cg_off_fast_path_raises).
    for name in sorted(GRADIENT_OPTIMIZERS - {'Newton-CG'}):
        solver = _fast_solver(TWO_QUBIT_OP, backend_config, optimizer=name, default_shots=1024)
        with (
            patch.object(VQESolver, 'get_backend') as get_backend,
            pytest.raises(InvalidSolverConfigError, match='gradient-free optimizer'),
        ):
            solver.solve()
        get_backend.assert_not_called()

    # The CLI defaults: the message must name the optimizer, the shots and both ways out.
    solver = _fast_solver(TWO_QUBIT_OP, backend_config, optimizer='L-BFGS-B', default_shots=1024)
    with (
        patch.object(VQESolver, 'get_backend') as get_backend,
        pytest.raises(InvalidSolverConfigError) as excinfo,
    ):
        solver.solve()
    message = str(excinfo.value)
    for part in ('L-BFGS-B', '1024', '--exact', 'COBYLA'):
        assert part in message
    get_backend.assert_not_called()

    # The guard lives in solve(), so the same flags with COBYLA construct and run on Aer.
    solver = _fast_solver(TWO_QUBIT_OP, backend_config, optimizer='COBYLA', default_shots=1024)
    via_statevector, via_aer, via_ibmq = _route(solver, MagicMock(spec=AerBackend))
    via_aer.assert_called_once()
    via_statevector.assert_not_called()
    via_ibmq.assert_not_called()


@pytest.mark.parametrize(
    ('config_overrides', 'solver_overrides'),
    [
        ({'noise': 'ibm_brisbane'}, {}),
        ({'local': False}, {}),
        ({'gpu': True}, {}),
        ({}, {'default_shots': 1024}),
    ],
    ids=['noise', 'remote', 'gpu', 'sampled'],
)
def test_newton_cg_off_fast_path_raises(config_overrides, solver_overrides, backend_config):
    """Newton-CG needs a gradient function, which only the analytic path passes to scipy.
    On any other run solve() raises before asking for a backend, naming the ways out. With
    sampled shots this Newton-CG message wins over the generic default_shots one."""
    solver = _fast_solver(
        TWO_QUBIT_OP,
        replace(backend_config, **config_overrides),
        optimizer='Newton-CG',
        **solver_overrides,
    )
    with (
        patch.object(VQESolver, 'get_backend') as get_backend,
        patch.object(VQESolver, 'via_aer') as via_aer,
        patch.object(VQESolver, 'via_ibmq') as via_ibmq,
        pytest.raises(InvalidSolverConfigError, match='Newton-CG needs an analytic gradient') as e,
    ):
        solver.solve()
    for alternative in ('--exact', 'L-BFGS-B', 'COBYLA'):
        assert alternative in str(e.value)
    get_backend.assert_not_called()
    via_aer.assert_not_called()
    via_ibmq.assert_not_called()


def test_newton_cg_on_fast_path_runs(backend_config):
    """An exact, local, noiseless Newton-CG run is eligible, so the guard lets it through."""
    solver = _fast_solver(TWO_QUBIT_OP, backend_config, optimizer='Newton-CG')
    via_statevector, via_aer, via_ibmq = _route(solver, MagicMock(spec=AerBackend))
    via_statevector.assert_called_once()
    via_aer.assert_not_called()
    via_ibmq.assert_not_called()


def test_gradient_optimizers_are_configurable():
    """GRADIENT_OPTIMIZERS is a subset of OptimizerConfigFactory.get_supported_optimizers(), so the
    gate can never select an optimizer get_optimizer_configuration would reject."""
    assert GRADIENT_OPTIMIZERS.issubset(OptimizerConfigFactory.get_supported_optimizers())


def test_no_configurable_gradient_optimizer_is_missing():
    """The converse of the test above, so the frozenset cannot silently fall out of sync with
    _configs. Every name in get_supported_optimizers() is probed with a real
    minimize(..., jac=True) on a quadratic.
    The names scipy accepts without warning must be exactly GRADIENT_OPTIMIZERS.
    Measured today: clean for L-BFGS-B, SLSQP, BFGS, CG, TNC, Newton-CG and trust-constr;
    warned for COBYLA, Nelder-Mead and Powell.
    Adding a gradient-capable optimizer to _configs and forgetting the frozenset fails here."""

    def fg(x):
        return float(x @ x), 2.0 * x

    accepted = set()
    for name in OptimizerConfigFactory.get_supported_optimizers():
        with warnings.catch_warnings(record=True) as caught:
            # pyproject filterwarnings starts with "error"; record=True does not reset the filter,
            # so without 'always' COBYLA's jac warning raises instead of being recorded.
            warnings.simplefilter('always')
            minimize(fg, np.array([1.0, 2.0]), jac=True, method=name)
        if not any('does not use gradient information' in str(w.message) for w in caught):
            accepted.add(name)

    assert accepted == set(GRADIENT_OPTIMIZERS)


def test_hf_point_gradient_vanishes(h2_cell, backend_config):
    """ExcitationPreserving(entanglement='full', initial_state=HartreeFock) at zero parameters:
    |grad| == 0.0 to 1e-12 on H2/sto-3g. The chapter-3 claim in miniature; the full 4/10/12/14
    table is the benchmark script's job."""
    solver = _fast_solver(
        h2_cell.qubit_op,
        backend_config,
        ansatz_type='ExcitationPreserving',
        hf_data=h2_cell.hf_data,
        mapper=h2_cell.mapper,
    )
    ansatz = solver._build_ansatz(4)

    gradient = solver._fast_path_objective(
        np.zeros(ansatz.num_parameters), ansatz, h2_cell.qubit_op
    )[1]

    assert ansatz.num_parameters > 0
    assert np.linalg.norm(gradient) == pytest.approx(0.0, abs=1e-12)


def test_benchmark_guard_rejects_nonzero_hf_gradient():
    path = Path(__file__).parents[2] / 'scripts' / 'benchmark_fast_path.py'
    spec = importlib.util.spec_from_file_location('benchmark_fast_path', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match='not zero'):
        module.check_hf_stationary('H2', np.array([1e-9]))
