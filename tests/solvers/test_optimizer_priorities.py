"""
test_optimizer_priorities.py

Comprehensive tests for COBYLA and L-BFGS-B optimizer priority handling
when both --max-iterations and --convergence are specified.
"""

from unittest.mock import MagicMock, patch

import pytest

from quantum_pipeline.solvers.optimizer_config import (
    COBYLAConfig,
    LBFGSBConfig,
    get_optimizer_configuration,
)


class TestCOBYLAPriorities:
    """Test COBYLA priority handling with max_iterations and convergence."""

    def test_cobyla_max_iterations_only(self):
        """Test COBYLA with only max_iterations specified."""
        config = COBYLAConfig(max_iterations=20)
        options = config.get_options(num_parameters=10)
        minimize_tol = config.get_minimize_tol()

        assert options['maxiter'] == 20
        assert minimize_tol is None  # No convergence threshold

    def test_cobyla_convergence_only(self):
        """Test COBYLA with only convergence_threshold specified."""
        config = COBYLAConfig(convergence_threshold=0.01)
        options = config.get_options(num_parameters=10)
        minimize_tol = config.get_minimize_tol()

        assert options['maxiter'] == 1000  # Default
        assert minimize_tol == 0.01

    def test_cobyla_both_parameters_raises_error(self):
        """Test COBYLA raises error when both parameters are specified."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            COBYLAConfig(max_iterations=15, convergence_threshold=0.005)

    def test_cobyla_both_via_convenience_function_raises_error(self):
        """Test that the convenience function raises error when both parameters are specified."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            get_optimizer_configuration(
                optimizer='COBYLA', max_iterations=25, convergence_threshold=0.02, num_parameters=8
            )

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_cobyla_validation_warns_small_max_iterations(self, mock_logger):
        """Test COBYLA validation warnings when max_iterations is too small."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Test case where max_iterations is too small
        config = COBYLAConfig(max_iterations=3)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        # Should warn because 3 < (10 + 2)
        mock_logger_instance.warning.assert_called_once()
        warning_msg = mock_logger_instance.warning.call_args[0][0]
        assert 'less than recommended' in warning_msg
        assert '12' in warning_msg  # num_parameters + 2

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_cobyla_no_warning_sufficient_iterations(self, mock_logger):
        """Test COBYLA doesn't warn when max_iterations is sufficient."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = COBYLAConfig(max_iterations=20)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        # Should not warn because 20 >= (10 + 2)
        mock_logger_instance.warning.assert_not_called()

    def test_cobyla_realistic_scenario_max_iterations(self):
        """Test COBYLA with realistic VQE parameters using max_iterations."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=50,
            num_parameters=160,  # Typical VQE parameter count
        )

        assert options['maxiter'] == 50
        assert minimize_tol is None
        assert options['disp'] is False

    def test_cobyla_realistic_scenario_convergence(self):
        """Test COBYLA with realistic VQE parameters using convergence_threshold."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            convergence_threshold=0.001,
            num_parameters=160,  # Typical VQE parameter count
        )

        assert options['maxiter'] == 1000  # Default
        assert minimize_tol == 0.001
        assert options['disp'] is False


class TestLBFGSBPriorities:
    """Test L-BFGS-B priority handling with max_iterations and convergence."""

    def test_lbfgsb_max_iterations_only(self):
        """Test L-BFGS-B with only max_iterations specified."""
        config = LBFGSBConfig(max_iterations=30)
        options = config.get_options(num_parameters=10)
        minimize_tol = config.get_minimize_tol()

        assert options['maxiter'] == 30
        # For strict max_iterations mode, should use tight tolerances
        assert options['ftol'] == 1e-15  # Tight tolerance for strict iteration control
        assert options['gtol'] == 1e-15  # Tight tolerance for strict iteration control
        assert minimize_tol is None  # L-BFGS-B uses ftol/gtol, not global tol

    def test_lbfgsb_convergence_only(self):
        """Test L-BFGS-B with only convergence_threshold specified."""
        config = LBFGSBConfig(convergence_threshold=0.005)
        options = config.get_options(num_parameters=10)
        minimize_tol = config.get_minimize_tol()

        assert options['maxiter'] == 15000  # Default
        assert options['ftol'] == 0.005
        assert options['gtol'] == 0.005
        assert minimize_tol is None

    def test_lbfgsb_both_parameters_raises_error(self):
        """Test L-BFGS-B raises error when both parameters are specified."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            LBFGSBConfig(max_iterations=40, convergence_threshold=0.01)

    def test_lbfgsb_both_via_convenience_function_raises_error(self):
        """Test that the convenience function raises error when both parameters are specified."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            get_optimizer_configuration(
                optimizer='L-BFGS-B',
                max_iterations=100,
                convergence_threshold=0.001,
                num_parameters=50,
            )

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_lbfgsb_validation_invalid_max_iterations(self, mock_logger):
        """Test L-BFGS-B warns for invalid max_iterations."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = LBFGSBConfig(max_iterations=0)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        # Should warn about max_iterations <= 0
        mock_logger_instance.warning.assert_called_once()
        warning_msg = mock_logger_instance.warning.call_args[0][0]
        assert 'should be >= 1' in warning_msg

    def test_lbfgsb_realistic_scenario_max_iterations(self):
        """Test L-BFGS-B with realistic VQE parameters using max_iterations."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', max_iterations=200, num_parameters=160
        )

        assert options['maxiter'] == 200
        assert options['ftol'] == 1e-15  # Tight tolerance
        assert options['gtol'] == 1e-15
        assert minimize_tol is None

    def test_lbfgsb_realistic_scenario_convergence(self):
        """Test L-BFGS-B with realistic VQE parameters using convergence_threshold."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', convergence_threshold=0.0001, num_parameters=160
        )

        assert options['maxiter'] == 15000  # High default
        assert options['ftol'] == 0.0001
        assert options['gtol'] == 0.0001
        assert minimize_tol is None
        assert options['disp'] is False


class TestOptimizerPriorityIntegration:
    """Integration tests for mutually exclusive parameters across different scenarios."""

    @pytest.mark.parametrize(
        'optimizer,max_iter',
        [
            ('COBYLA', 10),
            ('COBYLA', 50),
            ('L-BFGS-B', 30),
            ('L-BFGS-B', 100),
        ],
    )
    def test_max_iterations_only(self, optimizer, max_iter):
        """Test that max_iterations is properly configured for all optimizers."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer=optimizer, max_iterations=max_iter, num_parameters=20
        )

        # All optimizers should respect max_iterations
        assert options['maxiter'] == max_iter

        if optimizer == 'COBYLA':
            # COBYLA doesn't set tolerance when using max_iterations
            assert minimize_tol is None
        elif optimizer == 'L-BFGS-B':
            # L-BFGS-B uses tight ftol/gtol for strict iteration control
            assert options['ftol'] == 1e-15
            assert options['gtol'] == 1e-15
            assert minimize_tol is None

    @pytest.mark.parametrize(
        'optimizer,convergence',
        [
            ('COBYLA', 0.01),
            ('COBYLA', 0.001),
            ('L-BFGS-B', 0.005),
            ('L-BFGS-B', 0.0001),
        ],
    )
    def test_convergence_threshold_only(self, optimizer, convergence):
        """Test that convergence_threshold is properly configured for all optimizers."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer=optimizer, convergence_threshold=convergence, num_parameters=20
        )

        if optimizer == 'COBYLA':
            # COBYLA uses default maxiter and global tolerance
            assert options['maxiter'] == 1000
            assert minimize_tol == convergence
        elif optimizer == 'L-BFGS-B':
            # L-BFGS-B uses high default maxiter and ftol/gtol
            assert options['maxiter'] == 15000
            assert options['ftol'] == convergence
            assert options['gtol'] == convergence
            assert minimize_tol is None

    def test_mutually_exclusive_behavior(self):
        """Test and document the mutually exclusive behavior."""
        # max_iterations and convergence_threshold are mutually exclusive

        # COBYLA example with max_iterations only
        cobyla_options, cobyla_tol = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=25, num_parameters=10
        )
        assert cobyla_options['maxiter'] == 25
        assert cobyla_tol is None

        # COBYLA example with convergence_threshold only
        cobyla_options, cobyla_tol = get_optimizer_configuration(
            optimizer='COBYLA', convergence_threshold=0.02, num_parameters=10
        )
        assert cobyla_options['maxiter'] == 1000  # Default
        assert cobyla_tol == 0.02

        # L-BFGS-B example with max_iterations only
        lbfgsb_options, _lbfgsb_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', max_iterations=100, num_parameters=50
        )
        assert lbfgsb_options['maxiter'] == 100
        assert lbfgsb_options['ftol'] == 1e-15  # Tight tolerance
        assert lbfgsb_options['gtol'] == 1e-15

        # L-BFGS-B example with convergence_threshold only
        lbfgsb_options, _lbfgsb_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', convergence_threshold=0.001, num_parameters=50
        )
        assert lbfgsb_options['maxiter'] == 15000  # High default
        assert lbfgsb_options['ftol'] == 0.001
        assert lbfgsb_options['gtol'] == 0.001

    def test_edge_cases_mutually_exclusive(self):
        """Test edge cases with mutually exclusive parameters."""
        # Very small max_iterations only
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=2, num_parameters=5
        )
        assert options['maxiter'] == 2
        assert minimize_tol is None

        # Very tight convergence only
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', convergence_threshold=1e-12, num_parameters=10
        )
        assert options['maxiter'] == 15000
        assert options['ftol'] == 1e-12
        assert options['gtol'] == 1e-12

    def test_real_world_thesis_scenarios(self):
        """Test realistic scenarios for thesis research."""

        # Scenario 1: Quick exploratory run with iteration limit
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=10, num_parameters=40
        )
        assert options['maxiter'] == 10
        assert minimize_tol is None

        # Scenario 2: Production L-BFGS-B run with convergence threshold
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', convergence_threshold=1e-6, num_parameters=160
        )
        assert options['maxiter'] == 15000  # High default for convergence
        assert options['ftol'] == 1e-6
        assert options['gtol'] == 1e-6
        assert minimize_tol is None

        # Scenario 3: COBYLA with convergence threshold
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', convergence_threshold=1e-4, num_parameters=80
        )
        assert options['maxiter'] == 1000  # Default
        assert minimize_tol == 1e-4

    def test_no_parameters_vs_single_parameter(self):
        """Test difference between no parameters and single parameter specified."""

        # No parameters - should use defaults
        no_params_options, no_params_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', num_parameters=10
        )
        assert no_params_options['maxiter'] == 15000  # Default
        assert no_params_tol is None

        # max_iterations only - should use specified value with tight tolerances
        max_iter_options, max_iter_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', max_iterations=50, num_parameters=10
        )
        assert max_iter_options['maxiter'] == 50
        assert max_iter_options['ftol'] == 1e-15
        assert max_iter_options['gtol'] == 1e-15
        assert max_iter_tol is None

        # convergence_threshold only - should use default iterations
        conv_options, conv_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', convergence_threshold=0.01, num_parameters=10
        )
        assert conv_options['maxiter'] == 15000
        assert conv_options['ftol'] == 0.01
        assert conv_options['gtol'] == 0.01
        assert conv_tol is None

    def test_documentation_examples(self):
        """Test examples that could be used in documentation."""

        # Example 1: "Stop after exactly 20 iterations"
        options, tol = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=20, num_parameters=10
        )
        assert options['maxiter'] == 20
        assert tol is None  # No convergence threshold

        # Example 2: "Stop only when converged"
        options, tol = get_optimizer_configuration(
            optimizer='COBYLA',
            # No max_iterations
            convergence_threshold=0.01,
            num_parameters=10,
        )
        assert options['maxiter'] == 1000  # Default, very high
        assert tol == 0.01  # Will stop when converged
