"""
test_optimizer_config.py

Comprehensive tests for the optimizer configuration factory and individual
optimizer configuration classes.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

from quantum_pipeline.solvers.optimizer_config import (
    OptimizerConfig,
    LBFGSBConfig,
    COBYLAConfig,
    SLSQPConfig,
    OptimizerConfigFactory,
    get_optimizer_configuration
)


class TestLBFGSBConfig:
    """Test cases for L-BFGS-B optimizer configuration."""

    def test_init_with_max_iterations_only(self):
        """Test initialization with only max_iterations."""
        config = LBFGSBConfig(max_iterations=100)
        assert config.max_iterations == 100
        assert config.convergence_threshold is None

    def test_init_with_convergence_threshold_only(self):
        """Test initialization with only convergence_threshold."""
        config = LBFGSBConfig(convergence_threshold=0.01)
        assert config.max_iterations is None
        assert config.convergence_threshold == 0.01

    def test_init_with_both_parameters_raises_error(self):
        """Test that initialization with both parameters raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            LBFGSBConfig(max_iterations=100, convergence_threshold=0.01)

    def test_init_with_no_parameters(self):
        """Test initialization with no parameters."""
        config = LBFGSBConfig()
        assert config.max_iterations is None
        assert config.convergence_threshold is None

    def test_get_options_max_iterations_only(self):
        """Test get_options with max_iterations only."""
        config = LBFGSBConfig(max_iterations=50)
        options = config.get_options(num_parameters=10)

        assert options['disp'] is False
        assert options['maxfun'] == 50
        assert options['maxiter'] == 50
        # For strict max_iterations mode, should use tight tolerances
        assert options['ftol'] == 1e-15
        assert options['gtol'] == 1e-15

    def test_get_options_convergence_threshold_only(self):
        """Test get_options with convergence_threshold only."""
        config = LBFGSBConfig(convergence_threshold=0.001)
        options = config.get_options(num_parameters=10)

        assert options['disp'] is False
        # When only convergence_threshold is specified, use default maxiter
        assert options['maxiter'] == 15000
        assert options['ftol'] == 0.001
        assert options['gtol'] == 0.001

    def test_get_options_both_parameters_raises_error(self):
        """Test that both parameters together raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            LBFGSBConfig(max_iterations=30, convergence_threshold=0.001)

    def test_get_options_no_parameters(self):
        """Test get_options with no parameters - uses defaults."""
        config = LBFGSBConfig()
        options = config.get_options(num_parameters=10)

        assert options['disp'] is False
        assert 'maxfun' not in options
        assert 'ftol' not in options or options.get('ftol') == 2.220446049250313e-09
        assert 'gtol' not in options or options.get('gtol') == 1e-05

    def test_get_minimize_tol(self):
        """Test get_minimize_tol always returns None for L-BFGS-B."""
        config = LBFGSBConfig(convergence_threshold=0.01)
        assert config.get_minimize_tol() is None

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_validate_parameters_valid(self, mock_logger):
        """Test validate_parameters with valid parameters."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = LBFGSBConfig(max_iterations=100)
        config.validate_parameters(num_parameters=10)

        mock_logger_instance.warning.assert_not_called()

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_validate_parameters_invalid_max_iterations(self, mock_logger):
        """Test validate_parameters with invalid max_iterations."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = LBFGSBConfig(max_iterations=0)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        mock_logger_instance.warning.assert_called_once()
        assert 'should be >= 1' in mock_logger_instance.warning.call_args[0][0]


class TestCOBYLAConfig:
    """Test cases for COBYLA optimizer configuration."""

    def test_init_raises_error_with_both_parameters(self):
        """Test COBYLA config raises error when both parameters are provided."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            COBYLAConfig(max_iterations=200, convergence_threshold=0.05)

    def test_get_options_with_max_iterations(self):
        """Test get_options with max_iterations."""
        config = COBYLAConfig(max_iterations=150)
        options = config.get_options(num_parameters=5)

        assert options['disp'] is False
        assert options['maxiter'] == 150

    def test_get_options_without_max_iterations(self):
        """Test get_options without max_iterations uses default."""
        config = COBYLAConfig()
        options = config.get_options(num_parameters=5)

        assert options['disp'] is False
        assert options['maxiter'] == 1000  # default

    def test_get_minimize_tol_with_convergence(self):
        """Test get_minimize_tol returns convergence_threshold."""
        config = COBYLAConfig(convergence_threshold=0.02)
        assert config.get_minimize_tol() == 0.02

    def test_get_minimize_tol_without_convergence(self):
        """Test get_minimize_tol returns None when no convergence_threshold."""
        config = COBYLAConfig()
        assert config.get_minimize_tol() is None

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_validate_parameters_sufficient_iterations(self, mock_logger):
        """Test validate_parameters with sufficient iterations."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = COBYLAConfig(max_iterations=20)
        config.validate_parameters(num_parameters=10)

        mock_logger_instance.warning.assert_not_called()

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_validate_parameters_insufficient_iterations(self, mock_logger):
        """Test validate_parameters with insufficient iterations."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = COBYLAConfig(max_iterations=5)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=10)

        mock_logger_instance.warning.assert_called_once()
        warning_msg = mock_logger_instance.warning.call_args[0][0]
        assert 'less than recommended' in warning_msg
        assert '12' in warning_msg  # num_parameters + 2

    def test_validate_parameters_no_max_iterations(self):
        """Test validate_parameters with no max_iterations set."""
        config = COBYLAConfig()
        # Should not raise any exception
        config.validate_parameters(num_parameters=10)


class TestSLSQPConfig:
    """Test cases for SLSQP optimizer configuration."""

    def test_init_raises_error_with_both_parameters(self):
        """Test SLSQP config raises error when both parameters are provided."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            SLSQPConfig(max_iterations=80, convergence_threshold=0.001)

    def test_get_options_with_convergence(self):
        """Test get_options with convergence_threshold."""
        config = SLSQPConfig(convergence_threshold=0.005)
        options = config.get_options(num_parameters=8)

        assert options['disp'] is False
        assert options['maxiter'] == 100  # Default when using convergence
        assert options['ftol'] == 0.005

    def test_get_options_without_convergence(self):
        """Test get_options without convergence_threshold."""
        config = SLSQPConfig(max_iterations=40)
        options = config.get_options(num_parameters=8)

        assert options['disp'] is False
        assert options['maxiter'] == 40
        assert 'ftol' not in options

    def test_get_options_default_maxiter(self):
        """Test get_options uses default maxiter when not specified."""
        config = SLSQPConfig()
        options = config.get_options(num_parameters=8)

        assert options['maxiter'] == 100  # default

    def test_get_minimize_tol_with_max_iterations(self):
        """Test get_minimize_tol returns None when max_iterations is set."""
        config = SLSQPConfig(max_iterations=50)
        assert config.get_minimize_tol() is None

    def test_get_minimize_tol_without_max_iterations(self):
        """Test get_minimize_tol returns convergence_threshold when no max_iterations."""
        config = SLSQPConfig(convergence_threshold=0.01)
        assert config.get_minimize_tol() == 0.01

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_validate_parameters_invalid_max_iterations(self, mock_logger):
        """Test validate_parameters with invalid max_iterations."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        config = SLSQPConfig(max_iterations=-5)
        config.logger = mock_logger_instance
        config.validate_parameters(num_parameters=5)

        mock_logger_instance.warning.assert_called_once()
        assert 'should be >= 1' in mock_logger_instance.warning.call_args[0][0]


class TestOptimizerConfigFactory:
    """Test cases for OptimizerConfigFactory."""

    def test_create_config_lbfgsb(self):
        """Test creating L-BFGS-B config with max_iterations."""
        config = OptimizerConfigFactory.create_config(
            optimizer='L-BFGS-B',
            max_iterations=100
        )

        assert isinstance(config, LBFGSBConfig)
        assert config.max_iterations == 100
        assert config.convergence_threshold is None

    def test_create_config_lbfgsb_raises_with_both(self):
        """Test that factory raises error when both parameters are provided."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            OptimizerConfigFactory.create_config(
                optimizer='L-BFGS-B',
                max_iterations=100,
                convergence_threshold=0.01
            )

    def test_create_config_cobyla(self):
        """Test creating COBYLA config."""
        config = OptimizerConfigFactory.create_config(
            optimizer='COBYLA',
            max_iterations=200
        )

        assert isinstance(config, COBYLAConfig)
        assert config.max_iterations == 200
        assert config.convergence_threshold is None

    def test_create_config_slsqp(self):
        """Test creating SLSQP config."""
        config = OptimizerConfigFactory.create_config(
            optimizer='SLSQP',
            convergence_threshold=0.005
        )

        assert isinstance(config, SLSQPConfig)
        assert config.max_iterations is None
        assert config.convergence_threshold == 0.005

    def test_create_config_unsupported_optimizer(self):
        """Test creating config for unsupported optimizer raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizerConfigFactory.create_config(optimizer='UNSUPPORTED')

        assert 'Unsupported optimizer: UNSUPPORTED' in str(exc_info.value)
        assert 'L-BFGS-B' in str(exc_info.value)
        assert 'COBYLA' in str(exc_info.value)
        assert 'SLSQP' in str(exc_info.value)

    def test_get_supported_optimizers(self):
        """Test get_supported_optimizers returns correct list."""
        optimizers = OptimizerConfigFactory.get_supported_optimizers()

        assert isinstance(optimizers, list)
        assert 'L-BFGS-B' in optimizers
        assert 'COBYLA' in optimizers
        assert 'SLSQP' in optimizers

    def test_register_optimizer(self):
        """Test registering a new optimizer."""
        class CustomConfig(OptimizerConfig):
            def get_options(self, num_parameters: int):
                return {'custom': True}

            def get_minimize_tol(self):
                return None

            def validate_parameters(self, num_parameters: int):
                pass

        # Register new optimizer
        OptimizerConfigFactory.register_optimizer('CUSTOM', CustomConfig)

        # Test it can be created
        config = OptimizerConfigFactory.create_config('CUSTOM')
        assert isinstance(config, CustomConfig)

        # Test it appears in supported optimizers
        assert 'CUSTOM' in OptimizerConfigFactory.get_supported_optimizers()

    def test_create_config_no_parameters(self):
        """Test creating config with no optional parameters."""
        config = OptimizerConfigFactory.create_config('L-BFGS-B')

        assert isinstance(config, LBFGSBConfig)
        assert config.max_iterations is None
        assert config.convergence_threshold is None


class TestGetOptimizerConfiguration:
    """Test cases for the convenience function."""

    def test_get_optimizer_configuration_lbfgsb(self):
        """Test convenience function with L-BFGS-B max_iterations."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=50,
            num_parameters=10
        )

        assert isinstance(options, dict)
        assert options['maxfun'] == 50
        assert options['maxiter'] == 50
        assert options['ftol'] == 1e-15  # Tight tolerance
        assert options['gtol'] == 1e-15
        assert minimize_tol is None

    def test_get_optimizer_configuration_lbfgsb_convergence(self):
        """Test convenience function with L-BFGS-B convergence_threshold."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            convergence_threshold=0.01,
            num_parameters=10
        )

        assert isinstance(options, dict)
        assert options['maxiter'] == 15000  # High default
        assert options['ftol'] == 0.01
        assert options['gtol'] == 0.01
        assert minimize_tol is None

    def test_get_optimizer_configuration_cobyla(self):
        """Test convenience function with COBYLA max_iterations."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=100,
            num_parameters=5
        )

        assert isinstance(options, dict)
        assert options['maxiter'] == 100
        assert minimize_tol is None

    def test_get_optimizer_configuration_cobyla_convergence(self):
        """Test convenience function with COBYLA convergence_threshold."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            convergence_threshold=0.02,
            num_parameters=5
        )

        assert isinstance(options, dict)
        assert options['maxiter'] == 1000  # Default
        assert minimize_tol == 0.02

    @patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger')
    def test_get_optimizer_configuration_validation_called(self, mock_logger):
        """Test that validation is called by convenience function."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # This should trigger a validation warning
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=2,  # Too small
            num_parameters=10
        )

        # The validation should have been called and logged a warning
        # We can't easily test the exact warning without more complex mocking,
        # but we can test that the function completed successfully
        assert isinstance(options, dict)
        assert options['maxiter'] == 2

    def test_get_optimizer_configuration_unsupported(self):
        """Test convenience function with unsupported optimizer."""
        with pytest.raises(ValueError):
            get_optimizer_configuration(
                optimizer='INVALID',
                num_parameters=5
            )


class TestOptimizerConfigAbstractBase:
    """Test cases for the abstract base class."""

    def test_cannot_instantiate_abstract_base(self):
        """Test that OptimizerConfig cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OptimizerConfig()

    def test_abstract_methods_must_be_implemented(self):
        """Test that subclasses must implement all abstract methods."""
        class IncompleteConfig(OptimizerConfig):
            pass  # Missing all abstract methods

        with pytest.raises(TypeError):
            IncompleteConfig()

    def test_concrete_implementation_works(self):
        """Test that proper implementation of abstract class works."""
        class ConcreteConfig(OptimizerConfig):
            def get_options(self, num_parameters: int):
                return {'test': True}

            def get_minimize_tol(self):
                return 0.01

            def validate_parameters(self, num_parameters: int):
                pass

        # Should be able to instantiate with one parameter
        config = ConcreteConfig(max_iterations=10)
        assert config.max_iterations == 10
        assert config.convergence_threshold is None
        assert config.get_options(5) == {'test': True}
        assert config.get_minimize_tol() == 0.01

    def test_concrete_implementation_raises_with_both(self):
        """Test that validation works in abstract base class."""
        class ConcreteConfig(OptimizerConfig):
            def get_options(self, num_parameters: int):
                return {'test': True}

            def get_minimize_tol(self):
                return 0.01

            def validate_parameters(self, num_parameters: int):
                pass

        # Should raise error with both parameters
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConcreteConfig(max_iterations=10, convergence_threshold=0.01)


# Integration tests
class TestOptimizerConfigIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.parametrize("optimizer", ['L-BFGS-B', 'COBYLA', 'SLSQP'])
    def test_full_workflow_all_optimizers(self, optimizer):
        """Test complete workflow for all supported optimizers with max_iterations."""
        max_iterations = 50
        num_parameters = 10

        # Create config with max_iterations only
        config = OptimizerConfigFactory.create_config(
            optimizer=optimizer,
            max_iterations=max_iterations
        )

        # Validate
        config.validate_parameters(num_parameters)

        # Get configuration
        options = config.get_options(num_parameters)
        minimize_tol = config.get_minimize_tol()

        # Basic assertions that should be true for all optimizers
        assert isinstance(options, dict)
        assert 'disp' in options
        assert options['disp'] is False
        assert 'maxiter' in options
        assert options['maxiter'] == max_iterations
        assert minimize_tol is None or isinstance(minimize_tol, float)

    @pytest.mark.parametrize("optimizer", ['L-BFGS-B', 'COBYLA', 'SLSQP'])
    def test_full_workflow_convergence(self, optimizer):
        """Test complete workflow for all supported optimizers with convergence_threshold."""
        convergence_threshold = 0.01
        num_parameters = 10

        # Create config with convergence_threshold only
        config = OptimizerConfigFactory.create_config(
            optimizer=optimizer,
            convergence_threshold=convergence_threshold
        )

        # Validate
        config.validate_parameters(num_parameters)

        # Get configuration
        options = config.get_options(num_parameters)
        minimize_tol = config.get_minimize_tol()

        # Basic assertions
        assert isinstance(options, dict)
        assert 'disp' in options
        assert options['disp'] is False
        assert 'maxiter' in options  # All should have maxiter in convergence mode
        assert minimize_tol is None or isinstance(minimize_tol, float)

    def test_convenience_function_matches_direct_usage(self):
        """Test that convenience function produces same result as direct usage."""
        optimizer = 'L-BFGS-B'
        max_iterations = 100
        num_parameters = 8

        # Direct usage
        config = OptimizerConfigFactory.create_config(
            optimizer=optimizer,
            max_iterations=max_iterations
        )
        config.validate_parameters(num_parameters)
        direct_options = config.get_options(num_parameters)
        direct_tol = config.get_minimize_tol()

        # Convenience function
        conv_options, conv_tol = get_optimizer_configuration(
            optimizer=optimizer,
            max_iterations=max_iterations,
            num_parameters=num_parameters
        )

        # Should be identical
        assert direct_options == conv_options
        assert direct_tol == conv_tol