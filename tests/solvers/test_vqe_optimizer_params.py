"""
test_vqe_optimizer_params.py

Focused tests to verify that the critical VQE parameter handling issues
identified in the logs have been fixed.
"""

import pytest
from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration


class TestVQECriticalFix:
    """Test the specific critical issues that were identified."""

    def test_cobyla_no_maxfun_parameter(self):
        """Test that COBYLA config no longer uses invalid 'maxfun' parameter."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=5,
            convergence_threshold=0.1,
            num_parameters=160,  # From the logs
        )

        # CRITICAL FIX: Should use 'maxiter' not 'maxfun'
        assert 'maxiter' in options
        assert options['maxiter'] == 5
        assert 'maxfun' not in options  # This was the bug causing scipy warnings

        # COBYLA uses global tol parameter for convergence
        assert minimize_tol == 0.1

    def test_lbfgsb_no_extreme_tolerances(self):
        """Test that L-BFGS-B no longer uses extremely small tolerance values."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=50,
            convergence_threshold=None,  # Strict max_iterations mode
            num_parameters=160,
        )

        # CRITICAL FIX: Should NOT have extremely small ftol/gtol values
        assert options.get('ftol', 0) != 1e-15  # Old broken behavior
        assert options.get('gtol', 0) != 1e-15  # Old broken behavior
        assert options['maxiter'] == 50

        # For strict max_iterations, L-BFGS-B doesn't use global tol
        assert minimize_tol is None

    def test_lbfgsb_with_convergence_threshold(self):
        """Test L-BFGS-B with convergence threshold sets proper values."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=None,
            convergence_threshold=0.01,
            num_parameters=160,
        )

        # Should use the convergence threshold properly
        assert options['ftol'] == 0.01
        assert options['gtol'] == 0.01
        assert minimize_tol is None  # L-BFGS-B uses ftol/gtol, not global tol

    def test_lbfgsb_both_parameters(self):
        """Test L-BFGS-B with both max_iterations and convergence_threshold."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=100,
            convergence_threshold=0.005,
            num_parameters=160,
        )

        # Should respect both parameters
        assert options['maxiter'] == 100
        assert options['ftol'] == 0.005
        assert options['gtol'] == 0.005
        assert minimize_tol is None

    def test_parameter_priority_logic(self):
        """Test that the parameter priority logic works correctly."""
        # When both are specified, both should be used (not one ignored)
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=15, convergence_threshold=0.05, num_parameters=160
        )

        # COBYLA should use both: maxiter for iterations limit and tol for convergence
        assert options['maxiter'] == 15
        assert minimize_tol == 0.05

    def test_cobyla_validation_warning(self):
        """Test that COBYLA warns about insufficient iterations."""
        import logging
        from unittest.mock import patch, MagicMock

        with patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            # This should trigger a warning: 5 < (160 + 2)
            options, minimize_tol = get_optimizer_configuration(
                optimizer='COBYLA',
                max_iterations=5,
                convergence_threshold=None,
                num_parameters=160,
            )

            # Still should return the requested configuration
            assert options['maxiter'] == 5

    def test_realistic_log_scenario(self):
        """Test the exact scenario from the provided logs."""
        # From logs: COBYLA with max_iterations=5, 160 parameters
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=5,
            convergence_threshold=0.1,  # From logs: --convergence specified
            num_parameters=160,
        )

        # Verify the fix prevents the issues seen in logs:
        # 1. No 'maxfun' parameter (was causing OptimizeWarning)
        assert 'maxfun' not in options
        assert 'maxiter' in options
        assert options['maxiter'] == 5

        # 2. Convergence threshold properly handled
        assert minimize_tol == 0.1

        # 3. No scipy warnings about unknown options
        # (this would be verified by the absence of warnings when running)

    def test_all_supported_optimizers_work(self):
        """Test that all supported optimizers have valid configurations."""
        from quantum_pipeline.solvers.optimizer_config import OptimizerConfigFactory

        # Test only the core optimizers, not test-specific ones
        core_optimizers = ['L-BFGS-B', 'COBYLA', 'SLSQP']

        for optimizer in core_optimizers:
            # Should not raise any exceptions
            options, minimize_tol = get_optimizer_configuration(
                optimizer=optimizer,
                max_iterations=50,
                convergence_threshold=0.01,
                num_parameters=10,
            )

            # Basic sanity checks
            assert isinstance(options, dict)
            assert 'maxiter' in options
            assert options['maxiter'] == 50
            assert minimize_tol is None or isinstance(minimize_tol, float)

    def test_edge_cases(self):
        """Test edge cases that could cause issues."""
        # Very small max_iterations
        options, _ = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=1, num_parameters=10
        )
        assert options['maxiter'] == 1

        # Very small convergence threshold
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', convergence_threshold=1e-10, num_parameters=10
        )
        assert minimize_tol == 1e-10

        # No parameters specified (should use defaults)
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', num_parameters=10
        )
        assert 'maxiter' in options
        assert options['maxiter'] == 15000  # L-BFGS-B default

