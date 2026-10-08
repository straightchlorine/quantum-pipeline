"""VQE solver: classical optimization loop over a parameterised ansatz to minimize <H>."""

import numpy as np
from qiskit.circuit.library import EfficientSU2, ExcitationPreserving, RealAmplitudes
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.backends.aer_simulator import AerBackend
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_ibm_runtime import EstimatorV2, Session
from scipy.optimize import minimize

from quantum_pipeline.circuits import HFData, build_hf_initial_state
from quantum_pipeline.configs.constants import (
    EP_HF_INIT_JITTER,
    EP_INIT_JITTER,
    HF_FIDELITY_THRESHOLD,
    HF_PRE_OPT_ATTEMPTS,
    HF_PRE_OPT_MAXITER,
)
from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.mappers.mapper import Mapper
from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration
from quantum_pipeline.solvers.solver import Solver
from quantum_pipeline.structures.vqe_observation import (
    VQEInitialData,
    VQEProcess,
    VQEResult,
)
from quantum_pipeline.utils.timer import Timer


class MaxFunctionEvalsReachedError(Exception):
    """Raised when the hard function evaluation limit is exceeded."""

    def __init__(self, iteration: int, best_params: np.ndarray, best_energy: float):
        super().__init__(f'Hard function evaluation limit reached after {iteration} evaluations.')
        self.iteration = iteration
        self.best_params = best_params
        self.best_energy = best_energy


class VQESolver(Solver):
    def __init__(
        self,
        qubit_op,
        backend_config: BackendConfig,
        max_iterations: int | None = 50,
        optimizer: str = 'COBYLA',
        ansatz_reps: int = 3,
        ansatz_type: str = 'EfficientSU2',
        default_shots: int | None = 1024,
        convergence_threshold: float | None = None,
        optimization_level: int = 3,
        seed: int | None = None,
        init_strategy: str = 'random',
        hf_data: HFData | None = None,
        mapper: Mapper | None = None,
    ) -> None:
        super().__init__()
        self.qubit_op = qubit_op
        self.ansatz_reps = ansatz_reps
        self.ansatz_type = ansatz_type
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.seed = seed
        self.digits_iter = len(str(max_iterations))
        self.default_shots = default_shots
        self.backend_config = backend_config
        self.vqe_process: list[VQEProcess] = []
        self.current_iter = 1
        self.convergence_threshold = convergence_threshold
        self.optimization_level = optimization_level
        self.init_strategy = init_strategy
        self.hf_data = hf_data
        self.mapper = mapper
        self.effective_init_strategy = init_strategy

    def _optimize_circuits(self, ansatz, hamiltonian, backend):
        """Prepare ISA-compatible circuits and observables"""
        target = backend.target
        pm = generate_preset_pass_manager(
            target=target,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed,
        )
        ansatz_isa = pm.run(ansatz)

        hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
        return ansatz_isa, hamiltonian_isa

    def _build_ansatz(self, n_qubits):
        """Build the parameterised circuit whose parameters the optimizer will tune.

        Args:
            n_qubits: Qubit count of the mapped Hamiltonian (`hamiltonian.num_qubits`
                at the call site), one qubit per spin orbital under Jordan-Wigner.

        Returns:
            An untranspiled Qiskit circuit; `solve` transpiles it for the backend.

        Raises:
            ValueError: If `ansatz_type` is ExcitationPreserving and `hf_data` or
                `mapper` is missing. Without them the circuit has no Hartree-Fock
                initial state, starts in the zero-electron sector, and no number of
                `reps` can reach a state with the right electron count.

        `EfficientSU2` and `RealAmplitudes` are generic rotation-plus-CX circuits with
        no particle-number constraint. They are built bare, with no initial state.

        Prepending the HF circuit does not help: the fixed CX layers clear it out
        as soon as any rotation is non-zero. HF enters through the initial parameters,
        for both circuits: `_compute_hf_initial_parameters` runs a
        short search for parameters whose output has high fidelity with the HF state.

        `ExcitationPreserving` is built from RZ rotations and XX+YY two-qubit gates.
        An XX+YY gate only moves an excitation between its two qubits, so the number
        of set qubits never changes.
        That is why it needs the HF state prepended as `initial_state`: it fixes
        the electron count and places them in the lowest orbitals before any gate
        runs - the circuit can never leave that sector.
        `build_hf_initial_state` builds it with qiskit-nature's `HartreeFock`,
        which knows the blocked alpha-then-beta orbital ordering of the mapper.

        Its initial parameters are a seeded Gaussian jitter around zero, set in
        `_compute_initial_parameters`; the width depends on `init_strategy`. `'hf'` uses
        N(0, `EP_HF_INIT_JITTER`) = N(0, 0.01), `'random'` uses N(0, `EP_INIT_JITTER`) =
        N(0, 0.5). The start is not exactly zero - there the circuit outputs the
        HF state unchanged, and HF is a stationary point of the energy with respect to
        single excitations (Brillouin's theorem): the gradient there is zero and an
        optimizer started there has nothing to follow. That is why even the `hf` arm
        carries a small jitter.
        Measured on H2/sto3g: the `hf` arm starts within a few mHa of HF, but COBYLA then
        stays within ~8 mHa of it; the `random` arm starts about 0.5 Ha above HF and
        reaches the correlation energy in most seeds. `init_strategy` records the arm
        that ran.

        `entanglement='full'` is also needed. With gates only between neighbouring qubits,
        every XX+YY gate is a rotation between two adjacent spin orbitals.

        Under Jordan-Wigner the adjacent hopping term has no Z string, so the gate
        is a one-body orbital rotation, and a product of orbital rotations maps
        a Slater determinant to another Slater determinant.

        The best such state is HF itself, so the correlation energy stays out of
        reach at any `reps`. A gate between non-adjacent qubits is the same
        hopping term dressed with the parity of the qubits in between, which is
        a many-body operation; that is what lets the circuit build superpositions
        of determinants.

        An unknown `ansatz_type` falls back to `EfficientSU2` with a warning rather than
        failing. The result record stores `ansatz_name=self.ansatz_type` verbatim, so a
        typo produces rows labelled with a name that was never run.
        """
        if self.ansatz_type == 'ExcitationPreserving':
            if self.hf_data is None or self.mapper is None:
                raise ValueError(
                    'ExcitationPreserving requires hf_data and mapper. '
                    'Without an HF initial state the circuit starts in the 0-electron sector '
                    'and cannot reach the molecular ground state at any reps.'
                )
            hf_state = build_hf_initial_state(self.hf_data, self.mapper)
            return ExcitationPreserving(
                n_qubits, reps=self.ansatz_reps, entanglement='full', initial_state=hf_state
            )

        ansatze = {
            'EfficientSU2': lambda: EfficientSU2(n_qubits, reps=self.ansatz_reps),
            'RealAmplitudes': lambda: RealAmplitudes(n_qubits, reps=self.ansatz_reps),
        }
        if self.ansatz_type not in ansatze:
            self.logger.warning(
                'Unknown ansatz_type %r, falling back to EfficientSU2', self.ansatz_type
            )
        return ansatze.get(self.ansatz_type, ansatze['EfficientSU2'])()

    def _compute_hf_initial_parameters(self, ansatz):
        """Find parameters of a bare `EfficientSU2` or `RealAmplitudes` that prepare the HF state.

        Performs a short classical pre-optimization to find parameters where
        the ansatz output matches the HF state (maximizes state fidelity).
        This avoids the problem of prepending HF circuit + zero params, where
        the fixed CX gates in both circuits destroy the HF state.
        """
        if self.hf_data is None or self.mapper is None:
            raise ValueError('HF data and mapper must be provided for HF parameter computation')
        hf_circuit = build_hf_initial_state(self.hf_data, self.mapper)
        target_sv = Statevector(hf_circuit)

        def neg_fidelity(params):
            bound = ansatz.assign_parameters(params)
            sv = Statevector(bound)
            return -state_fidelity(sv, target_sv)

        best_fid = 0.0
        best_params = np.zeros(ansatz.num_parameters)
        n_attempts = HF_PRE_OPT_ATTEMPTS

        # Default to seed 0 for reproducible HF pre-optimization
        seed = self.seed if self.seed is not None else 0

        for i in range(n_attempts):
            rng = np.random.default_rng(seed + i)
            x0 = 2 * np.pi * rng.random(ansatz.num_parameters)
            res = minimize(
                neg_fidelity, x0, method='COBYLA', options={'maxiter': HF_PRE_OPT_MAXITER}
            )
            fid = -res.fun
            if fid > best_fid:
                best_fid = fid
                best_params = res.x
            if best_fid > HF_FIDELITY_THRESHOLD:
                break

        self.logger.info(
            f'HF parameter pre-optimization: fidelity={best_fid:.6f} after {i + 1} attempts'
        )
        return best_params

    def _compute_initial_parameters(self, ansatz):
        """Compute initial ansatz parameters based on the configured strategy.

        Sets `effective_init_strategy` to what actually produced the start point, and
        `_build_init_data` stores that value, not the requested `init_strategy`:

        - `'hf'` when the HF pre-optimization ran (`EfficientSU2`, `RealAmplitudes`),
        - `'hf'` or `'random'` for `ExcitationPreserving`, as requested: HF reference state
          plus a seeded N(0, `EP_HF_INIT_JITTER`) jitter for `'hf'` and N(0,
          `EP_INIT_JITTER`) for `'random'`,
        - `'random'` when `'hf'` was requested for the other ansatzes but `hf_data` or
          `mapper` is missing (a warning is logged).
        """
        param_num = ansatz.num_parameters

        # ExcitationPreserving prepends the HF circuit as initial_state (see _build_ansatz).
        if self.ansatz_type == 'ExcitationPreserving':
            sigma = EP_HF_INIT_JITTER if self.init_strategy == 'hf' else EP_INIT_JITTER
            self.effective_init_strategy = 'hf' if self.init_strategy == 'hf' else 'random'
            rng = np.random.default_rng(self.seed if self.seed is not None else 0)
            self.logger.info(
                'ExcitationPreserving: HF reference state plus N(0, %s) parameter jitter, '
                'recorded as %s',
                sigma,
                self.effective_init_strategy,
            )
            return sigma * rng.standard_normal(param_num)

        self.effective_init_strategy = 'random'
        if self.init_strategy == 'hf':
            if self.hf_data is not None and self.mapper is not None:
                self.logger.info('Computing HF-equivalent initial parameters...')
                params = self._compute_hf_initial_parameters(ansatz)
                self.effective_init_strategy = 'hf'
                return params
            self.logger.warning(
                'HF init strategy requested but no HF data/mapper available, '
                'falling back to random'
            )

        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            self.logger.info(f'Using seed {self.seed} for parameter initialization')
        else:
            rng = np.random.default_rng()
        return 2 * np.pi * rng.random(param_num)

    @property
    def _nuclear_repulsion(self) -> np.float64 | None:
        if self.hf_data is not None and self.hf_data.nuclear_repulsion_energy is not None:
            return np.float64(self.hf_data.nuclear_repulsion_energy)
        return None

    def _log_total_energy(self, electronic_energy: float) -> None:
        if self._nuclear_repulsion is not None:
            total = electronic_energy + self._nuclear_repulsion
            self.logger.info(
                f'Total energy (electronic + nuclear repulsion): {total:.8f} Ha '
                f'(nuclear repulsion: {self._nuclear_repulsion:.8f} Ha)'
            )

    def _best_step(self) -> VQEProcess:
        """Return the VQEProcess with the lowest evaluated energy.

        Keys on the raw per-iteration `result` (not `cumulative_min_energy`), so the
        returned step's energy and parameters are self-consistent.
        """
        return min(self.vqe_process, key=lambda p: p.result)

    def _make_truncated_result(self, best_energy, best_params, elapsed) -> VQEResult:
        """Build a VQEResult from the best observed state after early termination."""
        return VQEResult(
            initial_data=self.init_data,
            iteration_list=self.vqe_process,
            minimum=np.float64(best_energy),
            optimal_parameters=best_params,
            maxcv=None,
            minimization_time=np.float64(elapsed),
            nuclear_repulsion_energy=self._nuclear_repulsion,
            success=False,
        )

    def compute_energy(self, params, ansatz, hamiltonian, estimator):
        """scipy objective callback: evaluate <H>, record the step, enforce the hard eval limit."""

        if self.max_iterations is not None and self.current_iter > self.max_iterations:
            best = self._best_step()
            raise MaxFunctionEvalsReachedError(
                iteration=self.current_iter - 1,
                best_params=best.parameters,
                best_energy=float(best.result),
            )

        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy, std = result[0].data.evs[0], result[0].data.stds[0]

        if self.vqe_process:
            prev = self.vqe_process[-1]
            energy_delta = np.float64(energy - prev.result)
            parameter_delta_norm = np.float64(np.linalg.norm(params - prev.parameters))
            prev_min = prev.cumulative_min_energy
            cumulative_min_energy = np.float64(
                energy if prev_min is None else min(energy, prev_min)
            )
        else:
            energy_delta = None
            parameter_delta_norm = None
            cumulative_min_energy = np.float64(energy)

        step = VQEProcess(
            iteration=self.current_iter,
            parameters=np.array(params, copy=True),
            result=energy,
            std=std,
            energy_delta=energy_delta,
            parameter_delta_norm=parameter_delta_norm,
            cumulative_min_energy=cumulative_min_energy,
        )

        self.vqe_process.append(step)
        self.logger.debug(
            f'Iters. done: {self.current_iter:0{self.digits_iter}d} [Current cost: {energy}]'
        )
        self.current_iter += 1
        return energy

    def _prepare_circuit(self, backend):
        """Build and transpile the ansatz and Hamiltonian for the given backend.

        Returns:
            Tuple of (x0, ansatz_isa, hamiltonian_isa) where x0 contains the
            initial parameters and the _isa variants are the backend-compatible
            transpiled forms.
        """
        hamiltonian = self.qubit_op

        self.logger.info(f'Initializing the ansatz with {self.ansatz_reps} reps...')
        ansatz = self._build_ansatz(hamiltonian.num_qubits)
        self.logger.info('Ansatz initialized.')

        x0 = self._compute_initial_parameters(ansatz)
        self.logger.debug(f'Initial ansatz parameters:\n\n{x0}\n')

        self.logger.info('Optimizing ansatz and hamiltonian...')
        ansatz_isa, hamiltonian_isa = self._optimize_circuits(ansatz, hamiltonian, backend)
        self.logger.info('Ansatz and hamiltonian optimized.')

        return x0, ansatz_isa, hamiltonian_isa

    def _build_init_data(
        self, backend_name, ansatz_isa, hamiltonian_isa, x0, exact_estimator=False
    ):
        """Construct and store VQEInitialData on self.init_data."""
        self.init_data = VQEInitialData(
            backend=backend_name,
            num_qubits=hamiltonian_isa.num_qubits,
            hamiltonian=hamiltonian_isa.to_list(),
            num_parameters=ansatz_isa.num_parameters,
            initial_parameters=x0,
            optimizer=self.optimizer,
            ansatz=ansatz_isa,
            ansatz_reps=self.ansatz_reps,
            noise_backend=self.backend_config.noise if self.backend_config.noise else 'undef',
            default_shots=self.default_shots,
            seed=self.seed,
            init_strategy=self.effective_init_strategy,
            ansatz_name=self.ansatz_type,
            exact_estimator=exact_estimator,
        )

    def _run_optimization(self, ansatz_isa, hamiltonian_isa, estimator, x0) -> VQEResult:
        """Run scipy minimize; a hard-limit abort in compute_energy still yields a result.

        Raises:
            ValueError: If both `max_iterations` and `convergence_threshold` are set
                (mutually exclusive, enforced by `OptimizerConfig`).
        """
        optimization_params, minimize_tol = get_optimizer_configuration(
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
            num_parameters=len(x0),
        )

        self.logger.debug(f'Optimization params: {optimization_params}')
        if minimize_tol is not None:
            self.logger.debug(f'Minimize tolerance: {minimize_tol}')

        if self.convergence_threshold:
            self.logger.info(
                f'Starting VQE optimization with convergence threshold {self.convergence_threshold}'
            )
        elif self.max_iterations:
            self.logger.info(
                f'Starting VQE optimization with max iterations {self.max_iterations}'
            )
        else:
            self.logger.info('Starting VQE optimization with default settings')

        truncated = None
        res = None
        with Timer() as t:
            try:
                res = minimize(
                    self.compute_energy,
                    x0,
                    args=(ansatz_isa, hamiltonian_isa, estimator),
                    method=self.optimizer,
                    options=optimization_params,
                    tol=minimize_tol,
                )
            except MaxFunctionEvalsReachedError as e:
                truncated = e

        if truncated is not None:
            self.logger.info(
                f'Hard evaluation limit reached after {truncated.iteration} function evaluations '
                f'(limit: {self.max_iterations}). Best energy: {truncated.best_energy:.8f} Ha'
            )
            self._log_total_energy(truncated.best_energy)
            return self._make_truncated_result(
                truncated.best_energy, truncated.best_params, t.elapsed
            )

        # Past the truncated-return above, minimize() completed normally, so res is set.
        if res is None:
            raise RuntimeError('Optimization produced no result (minimize did not run).')

        actual_iterations = len(self.vqe_process)
        if self.vqe_process:
            best_step = self._best_step()
            best_energy = float(best_step.result)
            best_params = best_step.parameters
        else:
            best_energy = res.fun
            best_params = res.x
        if self.convergence_threshold:
            if res.success:
                self.logger.info(
                    f'VQE converged after {actual_iterations} iterations '
                    f'(threshold: {self.convergence_threshold}). '
                    f'Best energy: {best_energy:.8f} Ha'
                )
            else:
                self.logger.info(
                    f'VQE stopped after {actual_iterations} iterations - convergence not achieved. '
                    f'Best energy: {best_energy:.8f} Ha'
                )
        else:
            self.logger.info(
                f'VQE completed {actual_iterations} iterations. Best energy: {best_energy:.8f} Ha'
            )
        self._log_total_energy(best_energy)

        return VQEResult(
            initial_data=self.init_data,
            iteration_list=self.vqe_process,
            minimum=np.float64(best_energy),
            optimal_parameters=best_params,
            maxcv=getattr(res, 'maxcv', None),
            minimization_time=np.float64(t.elapsed),
            nuclear_repulsion_energy=self._nuclear_repulsion,
            success=bool(res.success),
            nfev=int(res.nfev) if hasattr(res, 'nfev') and res.nfev is not None else None,
            nit=int(res.nit) if hasattr(res, 'nit') and res.nit is not None else None,
        )

    def via_ibmq(self, backend) -> VQEResult:
        """Run the VQE simulation on IBM Quantum backend."""
        x0, ansatz_isa, hamiltonian_isa = self._prepare_circuit(backend)
        self._build_init_data(backend.name, ansatz_isa, hamiltonian_isa, x0)

        self.logger.info('Opening a session...')
        with Session(backend=backend) as session:
            estimator = EstimatorV2(mode=session)
            estimator.options.update(default_shots=self.default_shots)
            result = self._run_optimization(ansatz_isa, hamiltonian_isa, estimator, x0)

        self.logger.info('Session closed.')
        self.logger.info(
            f'Calculations on quantum hardware via IBMQ completed in {result.minimization_time:.6f} seconds.'
        )
        return result

    def via_aer(self, backend) -> VQEResult:
        """Run the VQE simulation via Aer simulator.

        When default_shots is None, uses qiskit_aer.primitives.EstimatorV2 with
        default_precision=0.0: exact expectation values, no shot noise. A noise model on
        the backend still applies. Otherwise uses qiskit_ibm_runtime.EstimatorV2 in local
        mode with the specified shot count.
        """
        x0, ansatz_isa, hamiltonian_isa = self._prepare_circuit(backend)

        exact = self.default_shots is None
        self._build_init_data(backend.name, ansatz_isa, hamiltonian_isa, x0, exact_estimator=exact)

        if exact:
            # from_backend, not AerEstimatorV2(): the bare constructor spins up its own
            # default AerSimulator and would drop the configured method, noise model and GPU.
            estimator = AerEstimatorV2.from_backend(backend, options={'default_precision': 0.0})
            self.logger.info(
                f'Using exact Aer estimator on {backend.name} '
                '(default_precision=0.0, no shot noise).'
            )
        else:
            estimator = EstimatorV2(mode=backend)
            estimator.options.default_shots = self.default_shots

        result = self._run_optimization(ansatz_isa, hamiltonian_isa, estimator, x0)

        self.logger.info(
            f'Simulation via Aer completed in {result.minimization_time:.6f} seconds '
            f'and {len(result.iteration_list)} iterations.'
        )
        return result

    def solve(self) -> VQEResult:
        """Run the VQE simulation and return the result."""
        self.current_iter = 1
        self.vqe_process = []

        backend = self.get_backend()

        return self.via_aer(backend) if isinstance(backend, AerBackend) else self.via_ibmq(backend)
