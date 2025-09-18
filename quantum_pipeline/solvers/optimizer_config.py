"""
optimizer_config.py

This module provides a factory for creating optimizer-specific configurations
for scipy.optimize.minimize. It handles the proper parameter mapping and
validation for different optimizers used in VQE optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging


class OptimizerConfig(ABC):
    """Abstract base class for optimizer configurations."""

    def __init__(self, max_iterations: Optional[int] = None,
                 convergence_threshold: Optional[float] = None):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_options(self, num_parameters: int) -> Dict[str, Any]:
        """Get optimizer-specific options dict for scipy.optimize.minimize."""
        pass

    @abstractmethod
    def get_minimize_tol(self) -> Optional[float]:
        """Get the tolerance parameter for scipy.optimize.minimize."""
        pass

    @abstractmethod
    def validate_parameters(self, num_parameters: int) -> None:
        """Validate parameters and log warnings if needed."""
        pass


class LBFGSBConfig(OptimizerConfig):
    """Configuration for L-BFGS-B optimizer."""

    def get_options(self, num_parameters: int) -> Dict[str, Any]:
        options = {
            'disp': False,
            'maxiter': self.max_iterations or 15000  # scipy default
        }

        # Handle convergence criteria based on user priorities
        if self.max_iterations and not self.convergence_threshold:
            # Strict iteration limit - use very tight convergence criteria
            # This forces the optimizer to run closer to maxiter iterations
            options.update({
                'ftol': 1e-15,  # Very tight function tolerance
                'gtol': 1e-15,  # Very tight gradient tolerance
            })
            self.logger.debug(f'L-BFGS-B configured for strict max_iterations={self.max_iterations} with tight tolerances')

        elif self.convergence_threshold:
            # Use convergence threshold
            options.update({
                'ftol': self.convergence_threshold,
                'gtol': self.convergence_threshold,
            })
            if self.max_iterations:
                self.logger.debug(
                    f'L-BFGS-B will stop at convergence {self.convergence_threshold} '
                    f'or max_iterations {self.max_iterations}, whichever comes first'
                )
            else:
                self.logger.debug(f'L-BFGS-B will stop at convergence threshold {self.convergence_threshold}')

        return options

    def get_minimize_tol(self) -> Optional[float]:
        """L-BFGS-B uses ftol/gtol in options, not the global tol parameter."""
        return None

    def validate_parameters(self, num_parameters: int) -> None:
        """L-BFGS-B generally doesn't have strict parameter requirements."""
        if self.max_iterations is not None and self.max_iterations < 1:
            self.logger.warning(f'L-BFGS-B max_iterations {self.max_iterations} should be >= 1')


class COBYLAConfig(OptimizerConfig):
    """Configuration for COBYLA optimizer."""

    def get_options(self, num_parameters: int) -> Dict[str, Any]:
        max_iter = self.max_iterations or 1000  # scipy default
        options = {
            'disp': False,
        }

        # For COBYLA, use maxfun instead of maxiter to avoid scipy's internal override
        # maxfun controls the maximum number of function evaluations
        if self.max_iterations is not None:
            options['maxfun'] = max_iter
        else:
            options['maxiter'] = max_iter

        return options

    def get_minimize_tol(self) -> Optional[float]:
        """COBYLA uses the global tol parameter for convergence."""
        return self.convergence_threshold

    def validate_parameters(self, num_parameters: int) -> None:
        """Validate COBYLA-specific requirements."""
        if self.max_iterations:
            # COBYLA documentation suggests it needs reasonable number of iterations
            min_recommended = num_parameters + 2
            if self.max_iterations < min_recommended:
                self.logger.warning(
                    f'COBYLA max_iterations {self.max_iterations} is less than recommended '
                    f'{min_recommended} for {num_parameters} parameters. This may cause early termination.'
                )


class SLSQPConfig(OptimizerConfig):
    """Configuration for SLSQP optimizer."""

    def get_options(self, num_parameters: int) -> Dict[str, Any]:
        options = {
            'disp': False,
            'maxiter': self.max_iterations or 100  # scipy default
        }

        if self.convergence_threshold:
            options['ftol'] = self.convergence_threshold

        return options

    def get_minimize_tol(self) -> Optional[float]:
        """SLSQP can use both options and global tol."""
        return self.convergence_threshold if not self.max_iterations else None

    def validate_parameters(self, num_parameters: int) -> None:
        """SLSQP validation."""
        if self.max_iterations is not None and self.max_iterations < 1:
            self.logger.warning(f'SLSQP max_iterations {self.max_iterations} should be >= 1')


class OptimizerConfigFactory:
    """Factory class for creating optimizer-specific configurations."""

    _configs = {
        'L-BFGS-B': LBFGSBConfig,
        'COBYLA': COBYLAConfig,
        'SLSQP': SLSQPConfig,
    }

    @classmethod
    def create_config(cls, optimizer: str, max_iterations: Optional[int] = None,
                     convergence_threshold: Optional[float] = None) -> OptimizerConfig:
        """
        Create optimizer-specific configuration.

        Args:
            optimizer: Name of the optimizer ('L-BFGS-B', 'COBYLA', 'SLSQP', etc.)
            max_iterations: Maximum number of iterations (takes priority over convergence)
            convergence_threshold: Convergence threshold for optimization

        Returns:
            OptimizerConfig: Configured optimizer instance

        Raises:
            ValueError: If optimizer is not supported
        """
        if optimizer not in cls._configs:
            raise ValueError(f"Unsupported optimizer: {optimizer}. "
                           f"Supported optimizers: {list(cls._configs.keys())}")

        config_class = cls._configs[optimizer]
        return config_class(max_iterations=max_iterations,
                          convergence_threshold=convergence_threshold)

    @classmethod
    def get_supported_optimizers(cls) -> list[str]:
        """Get list of supported optimizers."""
        return list(cls._configs.keys())

    @classmethod
    def register_optimizer(cls, name: str, config_class: type[OptimizerConfig]) -> None:
        """Register a new optimizer configuration class."""
        cls._configs[name] = config_class


def get_optimizer_configuration(optimizer: str, max_iterations: Optional[int] = None,
                               convergence_threshold: Optional[float] = None,
                               num_parameters: int = 0) -> Tuple[Dict[str, Any], Optional[float]]:
    """
    Convenience function to get optimizer configuration.

    Args:
        optimizer: Name of the optimizer
        max_iterations: Maximum number of iterations
        convergence_threshold: Convergence threshold
        num_parameters: Number of parameters being optimized

    Returns:
        Tuple of (options_dict, minimize_tol)
    """
    config = OptimizerConfigFactory.create_config(
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold
    )

    config.validate_parameters(num_parameters)
    options = config.get_options(num_parameters)
    minimize_tol = config.get_minimize_tol()

    return options, minimize_tol