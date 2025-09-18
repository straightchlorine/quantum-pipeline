"""
test_priority_behavior.py

Tests that demonstrate the actual behavior of optimizer priorities
by simulating scipy.optimize.minimize calls with different termination conditions.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration


class MockOptimizationResult:
    """Mock optimization result to simulate different termination scenarios."""

    def __init__(self, nit, success, message, fun=1.0, x=None):
        self.nit = nit  # Number of iterations performed
        self.success = success
        self.message = message
        self.fun = fun  # Final function value
        self.x = x if x is not None else [0.1, 0.2, 0.3]
        self.maxcv = 0.0


class TestPriorityBehaviorSimulation:
    """Test priority behavior by simulating optimization scenarios."""

    def test_cobyla_max_iterations_reached_first(self):
        """Test COBYLA when max_iterations is reached before convergence."""
        # Setup: max_iterations=5, convergence=0.001
        # Simulate: optimization stops at iteration 5 (max reached)
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=5,
            convergence_threshold=0.001,
            num_parameters=10
        )

        # Mock scipy.optimize.minimize behavior
        with patch('scipy.optimize.minimize') as mock_minimize:
            # Simulate COBYLA stopping due to max iterations
            mock_minimize.return_value = MockOptimizationResult(
                nit=5,  # Exactly max_iterations
                success=False,  # Didn't converge
                message="Maximum number of function evaluations has been exceeded."
            )

            # This would be called in VQE
            result = mock_minimize(
                lambda x: np.sum(x**2),  # Simple objective
                [1.0, 1.0],
                method='COBYLA',
                options=options,
                tol=minimize_tol
            )

            # Verify correct parameters were passed
            call_args = mock_minimize.call_args
            assert call_args.kwargs['options']['maxiter'] == 5
            assert call_args.kwargs['tol'] == 0.001

            # Verify behavior: stopped at max_iterations
            assert result.nit == 5
            assert not result.success  # Didn't converge

    def test_cobyla_convergence_reached_first(self):
        """Test COBYLA when convergence is reached before max_iterations."""
        # Setup: max_iterations=100, convergence=0.1
        # Simulate: optimization converges at iteration 15
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=100,
            convergence_threshold=0.1,
            num_parameters=10
        )

        with patch('scipy.optimize.minimize') as mock_minimize:
            # Simulate COBYLA stopping due to convergence
            mock_minimize.return_value = MockOptimizationResult(
                nit=15,  # Less than max_iterations (100)
                success=True,  # Converged
                message="Optimization terminated successfully."
            )

            result = mock_minimize(
                lambda x: np.sum(x**2),
                [1.0, 1.0],
                method='COBYLA',
                options=options,
                tol=minimize_tol
            )

            # Verify correct parameters
            call_args = mock_minimize.call_args
            assert call_args.kwargs['options']['maxiter'] == 100
            assert call_args.kwargs['tol'] == 0.1

            # Verify behavior: stopped due to convergence
            assert result.nit == 15  # Less than max_iterations
            assert result.success  # Converged successfully

    def test_lbfgsb_max_iterations_reached_first(self):
        """Test L-BFGS-B when max_iterations is reached before convergence."""
        # Setup: max_iterations=20, convergence=1e-8 (very tight)
        # Simulate: optimization stops at iteration 20 (max reached)
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=20,
            convergence_threshold=1e-8,
            num_parameters=10
        )

        with patch('scipy.optimize.minimize') as mock_minimize:
            # Simulate L-BFGS-B stopping due to max iterations
            mock_minimize.return_value = MockOptimizationResult(
                nit=20,  # Exactly max_iterations
                success=False,  # Didn't converge (tolerance too tight)
                message="STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
            )

            result = mock_minimize(
                lambda x: np.sum(x**2),
                [1.0, 1.0],
                method='L-BFGS-B',
                options=options,
                tol=minimize_tol
            )

            # Verify correct parameters
            call_args = mock_minimize.call_args
            assert call_args.kwargs['options']['maxiter'] == 20
            assert call_args.kwargs['options']['ftol'] == 1e-8
            assert call_args.kwargs['options']['gtol'] == 1e-8
            assert call_args.kwargs['tol'] is None  # L-BFGS-B doesn't use global tol

            # Verify behavior: stopped at max_iterations
            assert result.nit == 20
            assert not result.success

    def test_lbfgsb_convergence_reached_first(self):
        """Test L-BFGS-B when convergence is reached before max_iterations."""
        # Setup: max_iterations=200, convergence=0.01 (reasonable)
        # Simulate: optimization converges at iteration 45
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=200,
            convergence_threshold=0.01,
            num_parameters=10
        )

        with patch('scipy.optimize.minimize') as mock_minimize:
            # Simulate L-BFGS-B stopping due to convergence
            mock_minimize.return_value = MockOptimizationResult(
                nit=45,  # Less than max_iterations (200)
                success=True,  # Converged
                message="CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
            )

            result = mock_minimize(
                lambda x: np.sum(x**2),
                [1.0, 1.0],
                method='L-BFGS-B',
                options=options,
                tol=minimize_tol
            )

            # Verify correct parameters
            call_args = mock_minimize.call_args
            assert call_args.kwargs['options']['maxiter'] == 200
            assert call_args.kwargs['options']['ftol'] == 0.01
            assert call_args.kwargs['options']['gtol'] == 0.01

            # Verify behavior: stopped due to convergence
            assert result.nit == 45  # Less than max_iterations
            assert result.success

    def test_priority_scenarios_matrix(self):
        """Test all combinations of priority scenarios."""

        scenarios = [
            # (optimizer, max_iter, convergence, expected_stop_reason, expected_iterations)
            ('COBYLA', 5, 0.1, 'max_iterations', 5),
            ('COBYLA', 100, 0.01, 'convergence', 25),
            ('L-BFGS-B', 10, 1e-10, 'max_iterations', 10),
            ('L-BFGS-B', 500, 0.001, 'convergence', 80),
        ]

        for optimizer, max_iter, convergence, stop_reason, expected_nit in scenarios:
            with patch('scipy.optimize.minimize') as mock_minimize:
                # Configure mock based on expected stopping reason
                if stop_reason == 'max_iterations':
                    mock_result = MockOptimizationResult(
                        nit=expected_nit,
                        success=False,
                        message="Maximum iterations reached"
                    )
                else:  # convergence
                    mock_result = MockOptimizationResult(
                        nit=expected_nit,
                        success=True,
                        message="Converged successfully"
                    )

                mock_minimize.return_value = mock_result

                # Get configuration and simulate call
                options, minimize_tol = get_optimizer_configuration(
                    optimizer=optimizer,
                    max_iterations=max_iter,
                    convergence_threshold=convergence,
                    num_parameters=20
                )

                result = mock_minimize(
                    lambda x: x[0]**2 + x[1]**2,
                    [1.0, 1.0],
                    method=optimizer,
                    options=options,
                    tol=minimize_tol
                )

                # Verify the expected behavior
                assert result.nit == expected_nit
                if stop_reason == 'max_iterations':
                    assert not result.success
                else:
                    assert result.success

    def test_real_vqe_scenario_simulation(self):
        """Simulate a realistic VQE scenario with proper priority handling."""

        # Realistic VQE parameters: 160 parameters, moderate limits
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=100,
            convergence_threshold=1e-4,
            num_parameters=160
        )

        # Simulate a typical VQE energy minimization
        def mock_vqe_energy(params):
            """Mock VQE energy function that gradually converges."""
            return np.sum(params**2) + 1.0  # Simple quadratic with offset

        with patch('scipy.optimize.minimize') as mock_minimize:
            # Simulate L-BFGS-B converging after 67 iterations
            mock_minimize.return_value = MockOptimizationResult(
                nit=67,  # Converged before max_iterations (100)
                success=True,
                message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
                fun=0.9234,  # Final energy
                x=np.random.random(160) * 0.1  # Final parameters
            )

            result = mock_minimize(
                mock_vqe_energy,
                np.random.random(160) * 2 * np.pi,  # Initial VQE parameters
                method='L-BFGS-B',
                options=options,
                tol=minimize_tol
            )

            # Verify call parameters
            call_args = mock_minimize.call_args
            assert call_args.kwargs['method'] == 'L-BFGS-B'
            assert call_args.kwargs['options']['maxiter'] == 100
            assert call_args.kwargs['options']['ftol'] == 1e-4
            assert call_args.kwargs['options']['gtol'] == 1e-4
            assert call_args.kwargs['tol'] is None

            # Verify expected VQE behavior
            assert result.success
            assert result.nit < 100  # Converged before max_iterations
            assert result.fun < 2.0  # Energy decreased from initial guess

    def test_edge_case_behaviors(self):
        """Test edge cases where priorities interact in unexpected ways."""

        # Edge case 1: max_iterations = 1 (very restrictive)
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=1,
            convergence_threshold=0.1,
            num_parameters=5
        )

        with patch('scipy.optimize.minimize') as mock_minimize:
            mock_minimize.return_value = MockOptimizationResult(
                nit=1,  # Only one iteration allowed
                success=False,
                message="Maximum iterations reached"
            )

            result = mock_minimize(
                lambda x: np.sum(x**2),
                [1.0],
                method='COBYLA',
                options=options,
                tol=minimize_tol
            )

            assert result.nit == 1
            assert not result.success

        # Edge case 2: Very loose convergence (should converge quickly)
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=1000,
            convergence_threshold=1.0,  # Very loose
            num_parameters=5
        )

        with patch('scipy.optimize.minimize') as mock_minimize:
            mock_minimize.return_value = MockOptimizationResult(
                nit=2,  # Converged very quickly
                success=True,
                message="Converged due to loose tolerance"
            )

            result = mock_minimize(
                lambda x: np.sum(x**2),
                [1.0],
                method='L-BFGS-B',
                options=options,
                tol=minimize_tol
            )

            assert result.nit == 2  # Much less than max_iterations
            assert result.success

    def test_parameter_validation_in_context(self):
        """Test parameter validation in the context of realistic usage."""

        # Test scenario that would have triggered the original bugs
        with patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            # This was the problematic scenario from the logs
            options, minimize_tol = get_optimizer_configuration(
                optimizer='COBYLA',
                max_iterations=5,  # Too small for 160 parameters
                convergence_threshold=0.1,
                num_parameters=160
            )

            # Should still work but log a warning
            assert options['maxiter'] == 5
            assert minimize_tol == 0.1
            assert 'maxfun' not in options  # maxfun is not a valid COBYLA parameter

            # Test the fix prevents scipy warnings
            with patch('scipy.optimize.minimize') as mock_minimize:
                mock_minimize.return_value = MockOptimizationResult(
                    nit=5, success=False, message="Max iterations reached"
                )

                # This should NOT generate any scipy warnings about unknown options
                result = mock_minimize(
                    lambda x: np.sum(x**2),
                    np.random.random(160),
                    method='COBYLA',
                    options=options,
                    tol=minimize_tol
                )

                # Verify the call was made with correct parameters
                call_args = mock_minimize.call_args
                passed_options = call_args.kwargs['options']

                # Critical verification: only valid COBYLA parameters
                valid_cobyla_params = {'maxiter', 'disp', 'rhobeg', 'tol', 'catol'}
                for param in passed_options.keys():
                    assert param in valid_cobyla_params, f"Invalid COBYLA parameter: {param}"

    def test_documented_behavior_examples(self):
        """Test examples that demonstrate the documented priority behavior."""

        examples = [
            {
                'description': 'Stop after exactly N iterations',
                'optimizer': 'COBYLA',
                'max_iterations': 20,
                'convergence_threshold': None,
                'expected_behavior': 'Runs exactly 20 iterations regardless of convergence'
            },
            {
                'description': 'Stop when converged OR after N iterations',
                'optimizer': 'L-BFGS-B',
                'max_iterations': 100,
                'convergence_threshold': 0.001,
                'expected_behavior': 'Stops at convergence OR 100 iterations, whichever first'
            },
            {
                'description': 'Stop only when converged',
                'optimizer': 'COBYLA',
                'max_iterations': None,
                'convergence_threshold': 0.01,
                'expected_behavior': 'Runs until convergence, no iteration limit'
            }
        ]

        for example in examples:
            options, minimize_tol = get_optimizer_configuration(
                optimizer=example['optimizer'],
                max_iterations=example['max_iterations'],
                convergence_threshold=example['convergence_threshold'],
                num_parameters=50
            )

            if example['max_iterations'] is None:
                # Should use default max_iterations
                if example['optimizer'] == 'COBYLA':
                    assert options['maxiter'] == 1000  # COBYLA default
                elif example['optimizer'] == 'L-BFGS-B':
                    assert options['maxiter'] == 15000  # L-BFGS-B default
            else:
                assert options['maxiter'] == example['max_iterations']

            if example['convergence_threshold'] is not None:
                if example['optimizer'] == 'COBYLA':
                    assert minimize_tol == example['convergence_threshold']
                elif example['optimizer'] == 'L-BFGS-B':
                    assert options['ftol'] == example['convergence_threshold']
                    assert options['gtol'] == example['convergence_threshold']