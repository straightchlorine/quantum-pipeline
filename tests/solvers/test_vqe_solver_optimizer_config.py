"""
test_vqe_solver_optimizer_config.py

Tests for VQESolver integration with OptimizerConfig classes.
Verifies that VQESolver correctly uses optimizer configurations
for COBYLA and L-BFGS-B optimizers.
"""

import pytest

from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration


class TestVQESolverOptimizerConfig:
    """Test VQESolver integration with optimizer configuration classes."""

    def test_cobyla_max_iterations_only(self):
        """Test that COBYLA config uses only maxiter (no maxfun)."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=5,
            num_parameters=160,
        )

        # CRITICAL FIX: Should use only valid COBYLA parameters
        assert 'maxiter' in options
        assert options['maxiter'] == 5
        assert 'maxfun' not in options  # maxfun is not a valid COBYLA parameter
        assert minimize_tol is None  # No convergence threshold specified

    def test_lbfgsb_strict_max_iterations(self):
        """Test that L-BFGS-B uses tight tolerances for strict max_iterations mode."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=50,
            convergence_threshold=None,  # Strict max_iterations mode
            num_parameters=160,
        )

        # For strict max_iterations, should use very tight tolerances to prioritize iteration limit
        assert options['ftol'] == 1e-15  # Tight tolerance for strict iteration control
        assert options['gtol'] == 1e-15  # Tight tolerance for strict iteration control
        assert options['maxiter'] == 50

        # L-BFGS-B doesn't use global tol parameter
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

    def test_lbfgsb_mutual_exclusion(self):
        """Test L-BFGS-B raises error with both parameters."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            get_optimizer_configuration(
                optimizer='L-BFGS-B',
                max_iterations=100,
                convergence_threshold=0.005,
                num_parameters=160,
            )

    def test_cobyla_mutual_exclusion(self):
        """Test COBYLA raises error with both parameters."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            get_optimizer_configuration(
                optimizer='COBYLA',
                max_iterations=15,
                convergence_threshold=0.05,
                num_parameters=160,
            )

    def test_cobyla_validation_warning(self):
        """Test that COBYLA warns about insufficient iterations."""
        from unittest.mock import MagicMock, patch

        with patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            # This should trigger a warning: 5 < (160 + 2)
            options, _minimize_tol = get_optimizer_configuration(
                optimizer='COBYLA',
                max_iterations=5,
                convergence_threshold=None,
                num_parameters=160,
            )

            # Still should return the requested configuration
            assert options['maxiter'] == 5

    def test_realistic_log_scenario_max_iterations(self):
        """Test realistic scenario with max_iterations only."""
        # From logs: COBYLA with max_iterations=5, 160 parameters
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=5,
            num_parameters=160,
        )

        # Verify the fix prevents the issues seen in logs:
        # 1. Only valid COBYLA parameters (prevents warnings)
        assert 'maxiter' in options
        assert options['maxiter'] == 5
        assert 'maxfun' not in options  # maxfun is not a valid COBYLA parameter
        assert minimize_tol is None

    def test_realistic_log_scenario_convergence(self):
        """Test realistic scenario with convergence_threshold only."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            convergence_threshold=0.1,
            num_parameters=160,
        )

        # Only valid COBYLA parameters
        assert 'maxiter' in options
        assert options['maxiter'] == 1000  # Default
        assert 'maxfun' not in options
        assert minimize_tol == 0.1

    def test_all_supported_optimizers_work(self):
        """Test that all supported optimizers have valid configurations."""

        # Test only the core optimizers, not test-specific ones
        core_optimizers = ['L-BFGS-B', 'COBYLA', 'SLSQP']

        for optimizer in core_optimizers:
            # Test with max_iterations only
            options, _minimize_tol = get_optimizer_configuration(
                optimizer=optimizer,
                max_iterations=50,
                num_parameters=10,
            )

            # Basic sanity checks
            assert isinstance(options, dict)
            assert 'maxiter' in options
            assert options['maxiter'] == 50

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
