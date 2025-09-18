"""
test_optimizer_priorities.py

Comprehensive tests for COBYLA and L-BFGS-B optimizer priority handling
when both --max-iterations and --convergence are specified.
"""

import pytest
from unittest.mock import patch, MagicMock
from quantum_pipeline.solvers.optimizer_config import (
    COBYLAConfig,
    LBFGSBConfig,
    get_optimizer_configuration
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

    def test_cobyla_both_parameters_priority(self):
        """Test COBYLA priority when both max_iterations and convergence are specified."""
        config = COBYLAConfig(max_iterations=15, convergence_threshold=0.005)
        options = config.get_options(num_parameters=10)
        minimize_tol = config.get_minimize_tol()

        # COBYLA should respect BOTH parameters:
        # - max_iterations sets the iteration limit
        # - convergence_threshold sets the convergence criterion
        # Whichever is reached first will stop the optimization
        assert options['maxiter'] == 15
        assert minimize_tol == 0.005

    def test_cobyla_priority_via_convenience_function(self):
        """Test COBYLA priority using the convenience function."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=25,
            convergence_threshold=0.02,
            num_parameters=8
        )

        assert options['maxiter'] == 25
        assert minimize_tol == 0.02
        assert 'maxfun' not in options  # Verify the critical bug fix

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_cobyla_validation_with_both_parameters(self, mock_logger):
        """Test COBYLA validation warnings when both parameters are specified."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Test case where max_iterations is too small
        config = COBYLAConfig(max_iterations=3, convergence_threshold=0.01)
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

        config = COBYLAConfig(max_iterations=20, convergence_threshold=0.01)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        # Should not warn because 20 >= (10 + 2)
        mock_logger_instance.warning.assert_not_called()

    def test_cobyla_realistic_scenario(self):
        """Test COBYLA with realistic VQE parameters."""
        # Scenario: User wants max 50 iterations OR convergence at 0.001
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=50,
            convergence_threshold=0.001,
            num_parameters=160  # Typical VQE parameter count
        )

        assert options['maxiter'] == 50
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
        # Should use scipy defaults for ftol/gtol, not extreme values
        assert options.get('ftol', 0) != 1e-15  # Critical fix verification
        assert options.get('gtol', 0) != 1e-15  # Critical fix verification
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

    def test_lbfgsb_both_parameters_priority(self):
        """Test L-BFGS-B priority when both max_iterations and convergence are specified."""
        config = LBFGSBConfig(max_iterations=40, convergence_threshold=0.01)
        options = config.get_options(num_parameters=10)
        minimize_tol = config.get_minimize_tol()

        # L-BFGS-B should respect BOTH parameters:
        # - max_iterations sets the iteration limit
        # - convergence_threshold sets ftol and gtol
        # Whichever condition is met first will stop the optimization
        assert options['maxiter'] == 40
        assert options['ftol'] == 0.01
        assert options['gtol'] == 0.01
        assert minimize_tol is None

    def test_lbfgsb_priority_via_convenience_function(self):
        """Test L-BFGS-B priority using the convenience function."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=100,
            convergence_threshold=0.001,
            num_parameters=50
        )

        assert options['maxiter'] == 100
        assert options['ftol'] == 0.001
        assert options['gtol'] == 0.001
        assert minimize_tol is None

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_lbfgsb_validation_with_both_parameters(self, mock_logger):
        """Test L-BFGS-B validation with both parameters."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = LBFGSBConfig(max_iterations=50, convergence_threshold=0.01)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        # L-BFGS-B should not warn for reasonable parameters
        mock_logger_instance.warning.assert_not_called()

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_lbfgsb_validation_invalid_max_iterations(self, mock_logger):
        """Test L-BFGS-B warns for invalid max_iterations."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = LBFGSBConfig(max_iterations=0, convergence_threshold=0.01)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        # Should warn about max_iterations <= 0
        mock_logger_instance.warning.assert_called_once()
        warning_msg = mock_logger_instance.warning.call_args[0][0]
        assert 'should be >= 1' in warning_msg

    def test_lbfgsb_realistic_scenario(self):
        """Test L-BFGS-B with realistic VQE parameters for thesis work."""
        # Scenario: User wants max 200 iterations OR convergence at 0.0001
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=200,
            convergence_threshold=0.0001,
            num_parameters=160
        )

        assert options['maxiter'] == 200
        assert options['ftol'] == 0.0001
        assert options['gtol'] == 0.0001
        assert minimize_tol is None
        assert options['disp'] is False


class TestOptimizerPriorityIntegration:
    """Integration tests for optimizer priorities across different scenarios."""

    @pytest.mark.parametrize("optimizer,max_iter,convergence", [
        ('COBYLA', 10, 0.01),
        ('COBYLA', 50, 0.001),
        ('L-BFGS-B', 30, 0.005),
        ('L-BFGS-B', 100, 0.0001),
    ])
    def test_both_parameters_respected(self, optimizer, max_iter, convergence):
        """Test that both parameters are properly configured for all optimizers."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer=optimizer,
            max_iterations=max_iter,
            convergence_threshold=convergence,
            num_parameters=20
        )

        # All optimizers should respect max_iterations
        assert options['maxiter'] == max_iter

        if optimizer == 'COBYLA':
            # COBYLA uses global tolerance
            assert minimize_tol == convergence
        elif optimizer == 'L-BFGS-B':
            # L-BFGS-B uses ftol/gtol
            assert options['ftol'] == convergence
            assert options['gtol'] == convergence
            assert minimize_tol is None

    def test_priority_behavior_explanation(self):
        """Test and document the expected priority behavior."""
        # When both max_iterations and convergence_threshold are specified:
        # The optimization will stop when EITHER condition is met first

        # COBYLA example
        cobyla_options, cobyla_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=25,
            convergence_threshold=0.02,
            num_parameters=10
        )

        # Expected behavior: COBYLA will run up to 25 iterations,
        # but may stop earlier if convergence criterion (0.02) is met
        assert cobyla_options['maxiter'] == 25  # Iteration limit
        assert cobyla_tol == 0.02  # Convergence criterion

        # L-BFGS-B example
        lbfgsb_options, lbfgsb_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=100,
            convergence_threshold=0.001,
            num_parameters=50
        )

        # Expected behavior: L-BFGS-B will run up to 100 iterations,
        # but may stop earlier if function or gradient tolerance (0.001) is met
        assert lbfgsb_options['maxiter'] == 100  # Iteration limit
        assert lbfgsb_options['ftol'] == 0.001   # Function tolerance
        assert lbfgsb_options['gtol'] == 0.001   # Gradient tolerance
        assert lbfgsb_tol is None  # L-BFGS-B doesn't use global tol

    def test_edge_case_priorities(self):
        """Test edge cases for priority handling."""
        # Very small max_iterations with reasonable convergence
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=2,
            convergence_threshold=0.1,
            num_parameters=5
        )
        assert options['maxiter'] == 2
        assert minimize_tol == 0.1

        # Very tight convergence with reasonable max_iterations
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=1000,
            convergence_threshold=1e-12,
            num_parameters=10
        )
        assert options['maxiter'] == 1000
        assert options['ftol'] == 1e-12
        assert options['gtol'] == 1e-12

    def test_real_world_thesis_scenarios(self):
        """Test realistic scenarios for thesis research."""

        # Scenario 1: Quick exploratory run
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=10,
            convergence_threshold=0.1,
            num_parameters=40
        )
        assert options['maxiter'] == 10
        assert minimize_tol == 0.1

        # Scenario 2: Production L-BFGS-B run for thesis
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=500,
            convergence_threshold=1e-6,
            num_parameters=160
        )
        assert options['maxiter'] == 500
        assert options['ftol'] == 1e-6
        assert options['gtol'] == 1e-6
        assert minimize_tol is None

        # Scenario 3: Conservative COBYLA run
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=200,
            convergence_threshold=1e-4,
            num_parameters=80
        )
        assert options['maxiter'] == 200
        assert minimize_tol == 1e-4

    def test_no_parameters_vs_both_parameters(self):
        """Test difference between no parameters and both parameters specified."""

        # No parameters - should use defaults
        no_params_options, no_params_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            num_parameters=10
        )
        assert no_params_options['maxiter'] == 15000  # Default
        assert no_params_tol is None

        # Both parameters - should use specified values
        both_params_options, both_params_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=50,
            convergence_threshold=0.01,
            num_parameters=10
        )
        assert both_params_options['maxiter'] == 50
        assert both_params_options['ftol'] == 0.01
        assert both_params_options['gtol'] == 0.01
        assert both_params_tol is None

    def test_priority_documentation_examples(self):
        """Test examples that could be used in documentation."""

        # Example 1: "Stop after exactly 20 iterations"
        options, tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=20,
            # No convergence_threshold
            num_parameters=10
        )
        assert options['maxiter'] == 20
        assert tol is None  # No early stopping

        # Example 2: "Stop when converged OR after 100 iterations"
        options, tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=100,
            convergence_threshold=0.001,
            num_parameters=10
        )
        assert options['maxiter'] == 100
        assert options['ftol'] == 0.001
        assert options['gtol'] == 0.001

        # Example 3: "Stop only when converged"
        options, tol = get_optimizer_configuration(
            optimizer='COBYLA',
            # No max_iterations
            convergence_threshold=0.01,
            num_parameters=10
        )
        assert options['maxiter'] == 1000  # Default, very high
        assert tol == 0.01  # Will stop when converged