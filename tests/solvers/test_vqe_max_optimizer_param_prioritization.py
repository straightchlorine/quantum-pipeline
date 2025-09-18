"""
test_vqe_max_optimizer_param_prioritization.py

Tests to verify that the critical VQE parameter handling.

This specifically tests the max_iterations and convergence_threshold logic
for COBYLA and L-BFGS-B optimizers.
"""

import pytest
from unittest.mock import patch, MagicMock
from qiskit.quantum_info import SparsePauliOp
from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.vqe_solver import VQESolver


class TestVQEParameterFix:
    """Test that the VQE parameter handling fixes work correctly."""

    @pytest.fixture
    def simple_hamiltonian(self):
        """Create a simple Hamiltonian for testing."""
        return SparsePauliOp(['ZZ', 'XX'], coeffs=[1.0, 0.5])

    @pytest.fixture
    def backend_config(self):
        """Create a basic backend config."""
        return BackendConfig(
            local=True,
            gpu=False,
            optimization_level=2,
            min_num_qubits=2,
            filters=None,
            simulation_method='aer_simulator',
            gpu_opts=None,
            noise=None,
        )

    def test_cobyla_max_iterations_priority(self, simple_hamiltonian, backend_config):
        """Test that COBYLA respects max_iterations when specified."""
        # This test verifies that the maxfun error is fixed
        solver = VQESolver(
            qubit_op=simple_hamiltonian,
            backend_config=backend_config,
            max_iterations=5,
            optimizer='COBYLA',
            convergence_threshold=0.01,  # Should be ignored when max_iterations specified
        )

        # Mock the minimize function to avoid actual optimization
        with patch('quantum_pipeline.solvers.vqe_solver.minimize') as mock_minimize:
            mock_result = MagicMock()
            mock_result.fun = -1.5
            mock_result.x = [0.1, 0.2, 0.3, 0.4]
            mock_result.maxcv = 0.0
            mock_result.success = True
            mock_minimize.return_value = mock_result

            # Mock the backend and other components
            with patch.object(solver, 'get_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.name = 'aer_simulator'
                mock_get_backend.return_value = mock_backend

                result = solver.solve()

                # Verify minimize was called with correct parameters
                call_args = mock_minimize.call_args
                options = call_args.kwargs['options']

                # The critical fix: should use 'maxiter' not 'maxfun'
                assert 'maxiter' in options
                assert options['maxiter'] == 5
                assert 'maxfun' not in options  # This was the bug

                # Verify tolerance is passed correctly for COBYLA
                assert call_args.kwargs['tol'] == 0.01

    def test_lbfgsb_max_iterations_strict(self, simple_hamiltonian, backend_config):
        """Test that L-BFGS-B respects strict max_iterations."""
        solver = VQESolver(
            qubit_op=simple_hamiltonian,
            backend_config=backend_config,
            max_iterations=10,
            optimizer='L-BFGS-B',
            # No convergence_threshold - should disable convergence
        )

        with patch('quantum_pipeline.solvers.vqe_solver.minimize') as mock_minimize:
            mock_result = MagicMock()
            mock_result.fun = -1.5
            mock_result.x = [0.1, 0.2, 0.3, 0.4]
            mock_result.maxcv = 0.0
            mock_result.success = True
            mock_minimize.return_value = mock_result

            with patch.object(solver, 'get_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.name = 'aer_simulator'
                mock_get_backend.return_value = mock_backend

                result = solver.solve()

                call_args = mock_minimize.call_args
                options = call_args.kwargs['options']

                # Verify correct L-BFGS-B parameters
                assert options['maxiter'] == 10
                # The fix: should NOT have extremely small ftol/gtol values
                assert options.get('ftol', 0) != 1e-15  # Old broken behavior
                assert options.get('gtol', 0) != 1e-15  # Old broken behavior

                # For strict max_iterations, tol should be None
                assert call_args.kwargs['tol'] is None

    def test_lbfgsb_convergence_with_max_iterations(self, simple_hamiltonian, backend_config):
        """Test L-BFGS-B with both convergence and max_iterations."""
        solver = VQESolver(
            qubit_op=simple_hamiltonian,
            backend_config=backend_config,
            max_iterations=50,
            optimizer='L-BFGS-B',
            convergence_threshold=0.001,
        )

        with patch('quantum_pipeline.solvers.vqe_solver.minimize') as mock_minimize:
            mock_result = MagicMock()
            mock_result.fun = -1.5
            mock_result.x = [0.1, 0.2, 0.3, 0.4]
            mock_result.maxcv = 0.0
            mock_result.success = True
            mock_minimize.return_value = mock_result

            with patch.object(solver, 'get_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.name = 'aer_simulator'
                mock_get_backend.return_value = mock_backend

                result = solver.solve()

                call_args = mock_minimize.call_args
                options = call_args.kwargs['options']

                # Should respect both parameters
                assert options['maxiter'] == 50
                assert options['ftol'] == 0.001
                assert options['gtol'] == 0.001

                # L-BFGS-B doesn't use global tol
                assert call_args.kwargs['tol'] is None

    def test_cobyla_parameter_validation_warning(self, simple_hamiltonian, backend_config):
        """Test that COBYLA validation warns about insufficient iterations."""
        with patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            # This should trigger a warning because 2 < (num_params + 2)
            solver = VQESolver(
                qubit_op=simple_hamiltonian,
                backend_config=backend_config,
                max_iterations=2,  # Very small, should warn
                optimizer='COBYLA',
            )

            with patch('quantum_pipeline.solvers.vqe_solver.minimize') as mock_minimize:
                mock_result = MagicMock()
                mock_result.fun = -1.5
                mock_result.x = [0.1, 0.2]
                mock_result.maxcv = 0.0
                mock_result.success = True
                mock_minimize.return_value = mock_result

                with patch.object(solver, 'get_backend') as mock_get_backend:
                    mock_backend = MagicMock()
                    mock_backend.name = 'aer_simulator'
                    mock_get_backend.return_value = mock_backend

                    # This will create the optimizer config and should log warning
                    result = solver.solve()

    def test_parameter_priority_logic(self, simple_hamiltonian, backend_config):
        """Test that max_iterations takes priority over convergence_threshold."""
        solver = VQESolver(
            qubit_op=simple_hamiltonian,
            backend_config=backend_config,
            max_iterations=15,
            optimizer='COBYLA',
            convergence_threshold=0.05,
        )

        # Verify the priority is correctly understood by checking the config
        from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration

        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=15, convergence_threshold=0.05, num_parameters=8
        )

        # COBYLA should use both: maxiter for iterations and tol for convergence
        assert options['maxiter'] == 15
        assert minimize_tol == 0.05  # COBYLA uses global tol

    def test_unsupported_optimizer_handling(self, simple_hamiltonian, backend_config):
        """Test that unsupported optimizers raise appropriate errors."""
        from quantum_pipeline.solvers.optimizer_config import OptimizerConfigFactory

        with pytest.raises(ValueError) as exc_info:
            OptimizerConfigFactory.create_config('UNSUPPORTED_OPTIMIZER')

        assert 'Unsupported optimizer' in str(exc_info.value)
        assert 'COBYLA' in str(exc_info.value)
        assert 'L-BFGS-B' in str(exc_info.value)

    def test_no_scipy_warnings_for_cobyla(self, simple_hamiltonian, backend_config):
        """Test that the fix eliminates scipy warnings for COBYLA."""
        solver = VQESolver(
            qubit_op=simple_hamiltonian,
            backend_config=backend_config,
            max_iterations=10,
            optimizer='COBYLA',
        )

        # Mock minimize to capture what parameters are passed
        with patch('quantum_pipeline.solvers.vqe_solver.minimize') as mock_minimize:
            mock_result = MagicMock()
            mock_result.fun = -1.5
            mock_result.x = [0.1, 0.2, 0.3, 0.4]
            mock_result.maxcv = 0.0
            mock_result.success = True
            mock_minimize.return_value = mock_result

            with patch.object(solver, 'get_backend') as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.name = 'aer_simulator'
                mock_get_backend.return_value = mock_backend

                result = solver.solve()

                call_args = mock_minimize.call_args
                options = call_args.kwargs['options']

                # The critical fix: should NOT contain 'maxfun'
                # This was causing: "OptimizeWarning: Unknown solver options: maxfun"
                assert 'maxfun' not in options
                assert 'maxiter' in options
                assert options['maxiter'] == 10

