"""
vqe_solver.py

This module contains a function to solve a quantum operator using the
Variational Quantum Eigensolver (VQE). The VQE algorithm combines quantum and
classical optimization to find the minimum eigenvalue of a Hamiltonian.
"""

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.backends.aer_simulator import AerBackend
from qiskit_ibm_runtime import EstimatorV2, Session
from scipy.optimize import minimize

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration
from quantum_pipeline.solvers.solver import Solver
from quantum_pipeline.structures.vqe_observation import (
    VQEInitialData,
    VQEProcess,
    VQEResult,
)
from quantum_pipeline.utils.timer import Timer


class VQESolver(Solver):
    def __init__(
        self,
        qubit_op,
        backend_config: BackendConfig,
        max_iterations=50,
        optimizer='COBYLA',
        ansatz_reps=3,
        default_shots=1024,
        convergence_threshold=None,
        optimization_level=3,
    ):
        super().__init__()
        self.qubit_op = qubit_op
        self.ansatz_reps = ansatz_reps
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.digits_iter = len(str(max_iterations))
        self.default_shots = default_shots
        self.backend_config = backend_config
        self.vqe_process: list[VQEProcess] = []
        self.current_iter = 1
        self.convergence_threshold = convergence_threshold
        self.optimization_level = optimization_level

    def _optimize_circuits(self, ansatz, hamiltonian, backend):
        """Prepare ISA-compatible circuits and observables"""
        target = backend.target
        pm = generate_preset_pass_manager(
            target=target, optimization_level=self.optimization_level
        )
        ansatz_isa = pm.run(ansatz)

        hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
        return ansatz_isa, hamiltonian_isa

    def computeEnergy(self, params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator"""
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy, std = result[0].data.evs[0], result[0].data.stds[0]

        iter = VQEProcess(
            iteration=self.current_iter,
            parameters=params,
            result=energy,
            std=std,
        )

        self.vqe_process.append(iter)
        self.logger.debug(
            f'Iters. done: {self.current_iter:0{self.digits_iter}d} [Current cost: {energy}]'
        )
        self.current_iter += 1
        return energy

    def viaIBMQ(self, backend):
        """Run the VQE simulation on IBM Quantum backend."""
        hamiltonian = self.qubit_op

        self.logger.info(f'Initializing the ansatz with {self.ansatz_reps} reps...')
        ansatz = EfficientSU2(hamiltonian.num_qubits, reps=self.ansatz_reps)
        self.logger.info('Ansatz initialized.')

        param_num = ansatz.num_parameters
        x0 = 2 * np.pi * np.random.random(param_num)
        self.logger.debug(f'Initial ansatz parameters:\n\n{x0}\n')

        self.logger.info('Optimizing ansatz and hamiltonian...')
        ansatz_isa, hamiltonian_isa = self._optimize_circuits(ansatz, hamiltonian, backend)
        self.logger.info('Ansatz and hamiltonian optimized.')

        self.init_data = VQEInitialData(
            backend=backend.name,
            num_qubits=hamiltonian_isa.num_qubits,
            hamiltonian=hamiltonian_isa.to_list(),
            num_parameters=ansatz_isa.num_parameters,
            initial_parameters=x0,
            optimizer=self.optimizer,
            ansatz=ansatz_isa,
            ansatz_reps=self.ansatz_reps,
            noise_backend=self.backend_config.noise if self.backend_config.noise else 'undef',
            default_shots=self.default_shots,
        )
        self.logger.info('Opening a session...')

        with Session(backend=backend) as session:
            estimator = EstimatorV2(mode=session)
            estimator.options.default_shots = self.default_shots

            optimization_params, minimize_tol = get_optimizer_configuration(
                optimizer=self.optimizer,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                num_parameters=len(x0)
            )

            self.logger.debug(f'Optimization params: {optimization_params}')
            if minimize_tol is not None:
                self.logger.debug(f'Minimize tolerance: {minimize_tol}')

            if self.convergence_threshold and self.max_iterations:
                self.logger.info(
                    f'Starting VQE optimization with max iterations {self.max_iterations} taking priority over '
                    f'convergence threshold {self.convergence_threshold}'
                )
            elif self.convergence_threshold:
                self.logger.info(
                    f'Starting VQE optimization with convergence threshold {self.convergence_threshold}'
                )
            elif self.max_iterations:
                self.logger.info(
                    f'Starting VQE optimization with max iterations {self.max_iterations}'
                )
            else:
                self.logger.info(
                    'Starting VQE optimization with default settings'
                )

        with Timer() as t:
            res = minimize(
                self.computeEnergy,
                x0,
                args=(ansatz_isa, hamiltonian_isa, estimator),
                method=self.optimizer,
                options=optimization_params,
                tol=minimize_tol,
            )

        actual_iterations = len(self.vqe_process)
        if self.convergence_threshold:
            if res.success:
                self.logger.info(
                    f'VQE converged after {actual_iterations} iterations (threshold: {self.convergence_threshold})'
                )
            else:
                self.logger.info(
                    f'VQE stopped after {actual_iterations} iterations - convergence not achieved'
                )
        else:
            self.logger.info(f'VQE completed {actual_iterations} iterations')

        result = VQEResult(
            initial_data=self.init_data,
            iteration_list=self.vqe_process,
            minimum=res.fun,
            optimal_parameters=res.x,
            maxcv=getattr(res, 'maxcv', None),
            minimization_time=np.float64(t.elapsed),
        )

        self.logger.info('Session closed.')
        self.logger.info(
            f'Calculations on quantum hardware via IBMQ completed in {t.elapsed:.6f} seconds.'
        )
        return result

    def viaAer(self, backend):
        """Run the VQE simulation via Aer simulator."""
        hamiltonian = self.qubit_op

        self.logger.info('Initializing the ansatz...')
        ansatz = EfficientSU2(hamiltonian.num_qubits)
        self.logger.info('Ansatz initialized.')

        param_num = ansatz.num_parameters
        x0 = 2 * np.pi * np.random.random(param_num)
        self.logger.debug(f'Initial ansatz parameters:\n\n{x0}\n')

        self.logger.info('Optimizing ansatz and hamiltonian...')
        ansatz_isa, hamiltonian_isa = self._optimize_circuits(ansatz, hamiltonian, backend)
        self.logger.info('Ansatz and hamiltonian optimized.')

        self.init_data = VQEInitialData(
            backend=backend.name,
            num_qubits=hamiltonian_isa.num_qubits,
            hamiltonian=hamiltonian_isa.to_list(),
            num_parameters=ansatz_isa.num_parameters,
            initial_parameters=x0,
            optimizer=self.optimizer,
            ansatz=ansatz_isa,
            ansatz_reps=self.ansatz_reps,
            noise_backend=self.backend_config.noise if self.backend_config.noise else 'undef',
            default_shots=self.default_shots,
        )

        estimator = EstimatorV2(mode=backend)
        estimator.options.default_shots = self.default_shots

        optimization_params, minimize_tol = get_optimizer_configuration(
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
            num_parameters=len(x0)
        )

        self.logger.debug(f'Optimization params: {optimization_params}')
        if minimize_tol is not None:
            self.logger.debug(f'Minimize tolerance: {minimize_tol}')

        if self.convergence_threshold and self.max_iterations:
            self.logger.info(
                f'Starting VQE optimization with max iterations {self.max_iterations} taking priority over '
                f'convergence threshold {self.convergence_threshold}'
            )
        elif self.convergence_threshold:
            self.logger.info(
                f'Starting VQE optimization with convergence threshold {self.convergence_threshold}'
            )
        elif self.max_iterations:
            self.logger.info(
                f'Starting VQE optimization with max iterations {self.max_iterations}'
            )
        else:
            self.logger.info(
                'Starting VQE optimization with default settings'
            )
        with Timer() as t:
            res = minimize(
                self.computeEnergy,
                x0,
                args=(ansatz_isa, hamiltonian_isa, estimator),
                method=self.optimizer,
                options=optimization_params,
                tol=minimize_tol,
            )

        actual_iterations = len(self.vqe_process)
        if self.convergence_threshold:
            if res.success:
                self.logger.info(
                    f'VQE converged after {actual_iterations} iterations (threshold: {self.convergence_threshold})'
                )
            else:
                self.logger.info(
                    f'VQE stopped after {actual_iterations} iterations - convergence not achieved'
                )
        else:
            self.logger.info(f'VQE completed {actual_iterations} iterations')

        result = VQEResult(
            initial_data=self.init_data,
            iteration_list=self.vqe_process,
            minimum=res.fun,
            optimal_parameters=res.x,
            maxcv=getattr(res, 'maxcv', None),
            minimization_time=np.float64(t.elapsed),
        )

        self.logger.info(
            f'Simulation via Aer completed in {t.elapsed:.6f} seconds and {len(result.iteration_list)} iterations.'
        )
        return result

    def solve(self):
        """Run the VQE simulation and return the result."""
        self.current_iter = 1
        self.vqe_process = []

        backend = self.get_backend()

        result = None
        if isinstance(backend, AerBackend):
            result = self.viaAer(backend)
        else:
            result = self.viaIBMQ(backend)

        return result
