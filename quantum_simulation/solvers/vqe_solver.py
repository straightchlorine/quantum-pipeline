"""
vqe_solver.py

This module contains a function to solve a quantum operator using the
Variational Quantum Eigensolver (VQE). The VQE algorithm combines quantum and
classical optimization to find the minimum eigenvalue of a Hamiltonian.
"""

import json
import numpy as np
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.primitives import Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_ibm_runtime import EstimatorV2, Session
from scipy.optimize import minimize

from quantum_simulation.configs import settings
from quantum_simulation.report.report_generator import ReportGenerator
from quantum_simulation.utils.logger import get_logger
from quantum_simulation.visual.ansatz import AnsatzViewer
from quantum_simulation.visual.energy import EnergyPlotter

logger = get_logger('VQESolver')

cost_history_dict = {
    'prev_vector': None,
    'iters': 0,
    'cost_history': [],
}


class VQESolver:
    SUPPORTED_OPTIMIZERS = {
        'COBYLA': COBYLA(maxiter=100),
        'SPSA': SPSA(maxiter=50),
    }

    energy_plotter: EnergyPlotter

    def __init__(
        self,
        qubit_op,
        ansatz_reps=None,
        optimizer_name=None,
        max_iterations=100,
        custom_optimizer=None,
        report_generator=None,
        report_name='vqe_report.pdf',
        plot_convergence=True,
        symbols=None,
        provider=None,
        backend_name='qasm_simulator',
    ):
        self.report_name = report_name

        self.report: ReportGenerator = report_generator

        self.qubit_op = qubit_op
        self.ansatz_reps = ansatz_reps
        self.optimizer_name = optimizer_name
        self.max_iterations = max_iterations
        self.custom_optimizer = custom_optimizer
        self.plot_convergence = plot_convergence
        self.symbols = symbols
        self.provider = provider
        self.backend_name = backend_name

        if self.plot_convergence:
            self.energy_plotter = EnergyPlotter()

    def _prepare_circuits(self, ansatz, hamiltonian, backend):
        """Prepare ISA-compatible circuits and observables"""
        target = backend.target
        pm = generate_preset_pass_manager(target=target, optimization_level=3)
        ansatz_isa = pm.run(ansatz)

        AnsatzViewer.save_circuit(ansatz_isa, self.symbols)

        hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)
        return ansatz_isa, hamiltonian_isa

    def cost_func(self, params, ansatz, hamiltonian, estimator):
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
        energy = result[0].data.evs[0]

        cost_history_dict['iters'] += 1
        cost_history_dict['prev_vector'] = params
        cost_history_dict['cost_history'].append(energy)
        print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

        return energy

    def solve(self):
        """
        Solves the given qubit operator using VQE and generates a report.

        Args:
            qubit_op (QubitOperator): The qubit operator to solve.
            ansatz_reps (int): Number of repetitions for the ansatz.
            optimizer_name (str): Name of the optimizer to use.
            max_iterations (int): Maximum number of iterations for the
                optimizer.
            custom_optimizer (Optimizer): Custom optimizer to use.
            report_generator (ReportGenerator): Report generator instance.
            report_name (str): Name of the report file.
            plot_convergence (bool): Whether to plot the energy convergence.
            symbols (list): List of symbols to use for the ansatz.

        Returns:
            float: The minimum eigenvalue found by VQE.

        Raises:
            ValueError: If an unsupported optimizer is passed and no custom
                optimizer is provided.
            RuntimeError: If VQE fails to converge or produces an invalid
                result.
        """
        # set to default if not provided
        ansatz_reps = self.ansatz_reps or settings.ANSATZ_REPS
        optimizer_name = self.optimizer_name or settings.OPTIMIZER

        predefined_optimizers = {
            'COBYLA': COBYLA(maxiter=self.max_iterations),
            'SPSA': SPSA(maxiter=self.max_iterations),
        }

        if self.custom_optimizer:
            optimizer = self.custom_optimizer
            logger.info('Using custom optimizer.')
        elif optimizer_name in predefined_optimizers:
            optimizer = predefined_optimizers[optimizer_name]
            logger.info(f'Using predefined optimizer: {optimizer_name}')
        else:
            supported = list(predefined_optimizers.keys())
            raise ValueError(
                f'Unsupported optimizer: {optimizer_name}' + f' Expected: [{supported}]'
            )

        ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=ansatz_reps)
        logger.debug(f'Created ansatz with {ansatz_reps} repetitions.')

        def callback(iteration, parameters, energy, *args):
            """
            Callback function to log and track energy values during
            optimization.
            """
            self.energy_plotter.add_iteration(iteration, energy)

        if self.provider:
            logger.info('Attempting to find least busy backend...')
            backend = self.provider.least_busy(operational=True, simulator=False)
            logger.info(f'Backend {backend.name} found.')

            hamiltonian = self.qubit_op
            num_qubits = hamiltonian.num_qubits

            ansatz = EfficientSU2(hamiltonian.num_qubits)

            AnsatzViewer.save_circuit(ansatz, self.symbols)

            ansatz_isa, hamiltonian_isa = self._prepare_circuits(ansatz, hamiltonian, backend)
            number_of_parameters = ansatz.num_parameters

            x0 = 2 * np.pi * np.random.random(number_of_parameters)
            with Session(backend=backend) as session:
                estimator = EstimatorV2(mode=session)
                estimator.options.default_shots = 10000

                res = minimize(
                    self.cost_func,
                    x0,
                    args=(ansatz_isa, hamiltonian_isa, estimator),
                    method='cobyla',
                )

                res_dict = {
                    'optimal_parameters': res.optimal_parameters,
                    'minimal_energy': res.minimal_energy,
                    'success': res.success,
                    'num_iterations': len(cost_history_dict['cost_history']),
                    'num_qubits': num_qubits,
                }
                json.dump(res_dict, open('res.json', 'w'))
                json.dump(cost_history_dict, open('cost_history.json', 'w'))
                return res

        else:
            backend = AerSimulator()
            logger.info('Aer simulator backend initialized.')

            hamiltonian = self.qubit_op
            num_qubits = hamiltonian.num_qubits
            ansatz = EfficientSU2(hamiltonian.num_qubits)
            AnsatzViewer.save_circuit(ansatz, self.symbols)
            ansatz_isa, hamiltonian_isa = self._prepare_circuits(ansatz, hamiltonian, backend)
            number_of_parameters = ansatz.num_parameters
            x0 = 2 * np.pi * np.random.random(number_of_parameters)

            # Using primitive directly with Aer backend
            estimator = EstimatorV2(mode=backend)
            estimator.options.default_shots = 10000
            res = minimize(
                self.cost_func,
                x0,
                args=(ansatz_isa, hamiltonian_isa, estimator),
                method='cobyla',
            )
            res_dict = {
                'optimal_parameters': res.optimal_parameters,
                'minimal_energy': res.minimal_energy,
                'success': res.success,
                'num_iterations': len(cost_history_dict['cost_history']),
                'num_qubits': num_qubits,
            }
            json.dump(res_dict, open('res.json', 'w'))
            json.dump(cost_history_dict, open('cost_history.json', 'w'))
            return res

            # try:
            #     logger.info('Starting VQE computation...')
            #     result = vqe.compute_minimum_eigenvalue(self.qubit_op)
            # except Exception as e:
            #     logger.error(f'VQE execution failed: {str(e)}')
            #     raise RuntimeError('VQE execution encountered an error.') from e
            #
            # if result.eigenvalue is None:
            #     logger.error('VQE did not converge to a valid result.')
            #     raise RuntimeError('VQE did not produce a valid eigenvalue.')
            #
            # min_energy = result.eigenvalue.real
            # logger.info(f'VQE Converged. Minimum energy: {min_energy:.6f}')
            #
            # try:
            #     AnsatzViewer().save_circuit(ansatz, self.symbols)
            #     self.energy_plotter.plot_convergence(self.symbols)
            #     self.report.add_insight('Energy analysis', 'Results of VQE algorihtm:')
            #     self.report.add_metrics(
            #         {
            #             'Minimum Energy': str(round(min_energy, 4)) + ' Heartree',
            #             'Optimizer': optimizer_name,
            #             'Iterations': self.max_iterations,
            #             'Ansatz Repetitions': ansatz_reps,
            #         }
            #     )
            # except Exception as e:
            #     logger.error(f'Failed to generate report: {str(e)}')
            #
            # return min_energy
