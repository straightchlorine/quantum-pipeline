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

from quantum_simulation.configs import settings
from quantum_simulation.solvers.solver import Solver
from quantum_simulation.utils.observation import (
    BackendConfig,
    VQEInitialData,
    VQEProcess,
    VQEResult,
)


class VQESolver(Solver):
    def __init__(
        self,
        qubit_op,
        backend_config: BackendConfig,
        max_iterations=1,
        default_shots=10000,
        ansatz_reps=None,
        optimizer=None,
    ):
        super().__init__()
        self.qubit_op = qubit_op

        if ansatz_reps:
            self.ansatz_reps = ansatz_reps
        elif settings.ANSATZ_REPS:
            self.ansatz_reps = settings.ANSATZ_REPS
        else:
            self.ansatz_reps = 3

        if optimizer:
            supported = list(settings.SUPPORTED_OPTIMIZERS.keys())
            if optimizer in [x.lower() for x in supported]:
                self.optimizer = optimizer
            else:
                support = self.supported_optimizers_prompt()
                raise ValueError(f'Unsupported optimizer: {optimizer}\nSupported:\n{support}')
        elif settings.OPTIMIZER:
            self.optimizer = settings.OPTIMIZER
        else:
            self.optimizer = 'COBYLA'

        self.max_iterations = max_iterations
        self.default_shots = default_shots
        self.backend_config = backend_config
        self.vqe_process: list[VQEProcess] = []
        self.current_iter = 1

    def _optimize_circuits(self, ansatz, hamiltonian, backend):
        """Prepare ISA-compatible circuits and observables"""
        target = backend.target
        pm = generate_preset_pass_manager(target=target, optimization_level=3)
        ansatz_isa = pm.run(ansatz)

        hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
        return ansatz_isa, hamiltonian_isa

    def computeEnergy(self, params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (EstimatorV2): Estimator primitive instance
            cost_history_dict: Dictionary for storing intermediate results

        Returns:
            float: Energy estimate
        """
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy, std = result[0].data.evs[0], result[0].data.stds[0]

        iter = VQEProcess(
            iteration=self.current_iter,
            parameters=params,
            result=energy,
            std=std,
        )

        self.current_iter += 1
        self.vqe_process.append(iter)
        self.logger.debug(f'Iters. done: {self.current_iter} [Current cost: {energy}]')
        return energy

    def viaIBMQ(self, backend):
        """Run the VQE simulation on IBM Quantum backend."""
        hamiltonian = self.qubit_op

        self.logger.info('Initializing the ansatz...')
        ansatz = EfficientSU2(hamiltonian.num_qubits)
        self.logger.info('Ansatz initialized.')

        param_num = ansatz.num_parameters
        x0 = 2 * np.pi * np.random.random(param_num)
        self.logger.info('Initial ansatz parameters:\n\n{}\n'.format(x0))

        self.logger.info('Optimizing ansatz and hamiltonian...')
        ansatz_isa, hamiltonian_isa = self._optimize_circuits(ansatz, hamiltonian, backend)
        self.logger.info('Ansatz and hamiltonian optimized.')

        self.init_data = VQEInitialData(
            backend=backend.name,
            num_qubits=hamiltonian_isa.num_qubits,
            hamiltonian=hamiltonian_isa.to_matrix(),
            num_parameters=ansatz_isa.num_parameters,
            initial_parameters=x0,
            optimizer=self.optimizer,
            basis_set=settings.BASIS_SET,
            ansatz=ansatz_isa,
        )
        self.logger.info('Opening a session...')

        with Session(backend=backend) as session:
            estimator = EstimatorV2(mode=session)
            estimator.options.default_shots = self.default_shots

            optimization_params = {
                'maxiter': self.max_iterations,
                'disp': False,
            }

            res = minimize(
                self.computeEnergy,
                x0,
                args=(ansatz_isa, hamiltonian_isa, estimator),
                method=self.optimizer,
                options=optimization_params,
            )

            result = VQEResult(
                vqe_initial_data=self.init_data,
                vqe_iterations=self.vqe_process,
                minimum=res.fun,
                iterations=res.nfev,
                optimal_parameters=res.x,
                maxcv=res.maxcv,
            )

        self.logger.info('Session closed.')
        return result

    def viaAer(self, backend):
        """Run the VQE simulation via Aer simulator."""
        hamiltonian = self.qubit_op

        self.logger.info('Initializing the ansatz...')
        ansatz = EfficientSU2(hamiltonian.num_qubits)
        self.logger.info('Ansatz initialized.')

        param_num = ansatz.num_parameters
        x0 = 2 * np.pi * np.random.random(param_num)
        self.logger.info('Initial ansatz parameters:\n{}'.format(x0))

        self.logger.info('Optimizing ansatz and hamiltonian...')
        ansatz_isa, hamiltonian_isa = self._optimize_circuits(ansatz, hamiltonian, backend)
        self.logger.info('Ansatz and hamiltonian optimized.')

        self.init_data = VQEInitialData(
            backend=backend.name,
            num_qubits=hamiltonian_isa.num_qubits,
            hamiltonian=hamiltonian_isa.to_matrix(),
            num_parameters=ansatz_isa.num_parameters,
            initial_parameters=x0,
            optimizer=self.optimizer,
            basis_set=settings.BASIS_SET,
            ansatz=ansatz_isa,
        )

        estimator = EstimatorV2(mode=backend)
        estimator.options.default_shots = self.default_shots

        optimization_params = {
            'maxiter': self.max_iterations,
            'disp': False,
        }

        res = minimize(
            self.computeEnergy,
            x0,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method=self.optimizer,
            options=optimization_params,
        )

        result = VQEResult(
            vqe_initial_data=self.init_data,
            vqe_iterations=self.vqe_process,
            minimum=res.fun,
            iterations=res.nfev,
            optimal_parameters=res.x,
            maxcv=res.maxcv,
        )

        self.logger.info('Simulation via Aer completed.')
        return result

    def solve(self):
        """Run the VQE simulation and return the result."""
        backend = self.get_backend()

        result = None
        if isinstance(backend, AerBackend):
            result = self.viaAer(backend)
        else:
            result = self.viaIBMQ(backend)

        return result
