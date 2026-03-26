"""
optimizer_config.py

This module provides a factory for creating optimizer-specific configurations
for scipy.optimize.minimize. It handles the proper parameter mapping and
validation for different optimizers used in VQE optimization.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from quantum_pipeline.configs.constants import (
    COBYLA_DEFAULT_MAXITER,
    LBFGSB_DEFAULT_MAXITER,
    LBFGSB_TIGHT_TOL,
    SLSQP_DEFAULT_MAXITER,
)


class OptimizerConfig(ABC):
    """Abstract base class for optimizer configurations."""

    def __init__(
        self, max_iterations: int | None = None, convergence_threshold: float | None = None
    ):
        # Validate mutually exclusive parameters
        if max_iterations is not None and convergence_threshold is not None:
            raise ValueError(
                'max_iterations and convergence_threshold are mutually exclusive. '
                'Please specify only one.'
            )

        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_options(self, num_parameters: int) -> dict[str, Any]:
        """Get optimizer-specific options dict for scipy.optimize.minimize."""

    @abstractmethod
    def get_minimize_tol(self) -> float | None:
        """Get the tolerance parameter for scipy.optimize.minimize."""

    @abstractmethod
    def validate_parameters(self, num_parameters: int) -> None:
        """Validate parameters and log warnings if needed."""


class LBFGSBConfig(OptimizerConfig):
    """Configuration for L-BFGS-B optimizer.

    Simplified behavior:
    - If max_iterations is set: Use it with tight tolerances (1e-15) to enforce iteration limit
    - If convergence_threshold is set: Use it with high max iterations (15000) to let it converge
    - If neither is set: Use defaults (15000 iterations, standard scipy tolerances)
    """

    def get_options(self, num_parameters: int) -> dict[str, Any]:
        options = {'disp': False}

        if self.max_iterations is not None:
            # Mode: Strict iteration control
            # Set both maxfun and maxiter to prevent hanging
            options['maxfun'] = self.max_iterations
            options['maxiter'] = self.max_iterations
            # Use tight tolerances to ensure iteration limit is respected
            options['ftol'] = LBFGSB_TIGHT_TOL
            options['gtol'] = LBFGSB_TIGHT_TOL

        elif self.convergence_threshold is not None:
            # Mode: Convergence-based optimization
            # Use high iteration limit to allow convergence
            options['maxiter'] = LBFGSB_DEFAULT_MAXITER
            options['ftol'] = self.convergence_threshold
            options['gtol'] = self.convergence_threshold

        else:
            # Mode: Defaults
            options['maxiter'] = LBFGSB_DEFAULT_MAXITER
            # Let scipy use its default tolerances

        return options

    def get_minimize_tol(self) -> float | None:
        return None

    def validate_parameters(self, num_parameters: int) -> None:
        if self.max_iterations is not None and self.max_iterations < 1:
            self.logger.warning(f'L-BFGS-B max_iterations {self.max_iterations} should be >= 1')


class COBYLAConfig(OptimizerConfig):
    """Configuration for COBYLA optimizer.

    Simplified behavior:
    - If max_iterations is set: Use it as the iteration limit
    - If convergence_threshold is set: Use it with default iterations (1000)
    - If neither is set: Use scipy defaults (1000 iterations)
    """

    def get_options(self, num_parameters: int) -> dict[str, Any]:
        if self.max_iterations is not None:
            maxiter = self.max_iterations
        elif self.convergence_threshold is not None:
            maxiter = COBYLA_DEFAULT_MAXITER  # Default when using convergence
        else:
            maxiter = COBYLA_DEFAULT_MAXITER  # scipy default

        return {
            'disp': False,
            'maxiter': maxiter,
        }

    def get_minimize_tol(self) -> float | None:
        # COBYLA uses global tolerance parameter
        return self.convergence_threshold

    def validate_parameters(self, num_parameters: int) -> None:
        if self.max_iterations is not None:
            min_recommended = num_parameters + 2
            if self.max_iterations < min_recommended:
                self.logger.warning(
                    f'COBYLA max_iterations {self.max_iterations} is less than recommended '
                    f'{min_recommended} for {num_parameters} parameters. This may cause early termination.'
                )


class SLSQPConfig(OptimizerConfig):
    """Configuration for SLSQP optimizer.

    Simplified behavior:
    - If max_iterations is set: Use it as the iteration limit
    - If convergence_threshold is set: Use it with default iterations (100)
    - If neither is set: Use scipy defaults (100 iterations)
    """

    def get_options(self, num_parameters: int) -> dict[str, Any]:
        if self.max_iterations is not None:
            maxiter = self.max_iterations
        elif self.convergence_threshold is not None:
            maxiter = SLSQP_DEFAULT_MAXITER  # Default when using convergence
        else:
            maxiter = SLSQP_DEFAULT_MAXITER  # scipy default

        options = {'disp': False, 'maxiter': maxiter}

        if self.convergence_threshold is not None:
            options['ftol'] = self.convergence_threshold

        return options

    def get_minimize_tol(self) -> float | None:
        # SLSQP uses global tolerance when convergence_threshold is set
        return self.convergence_threshold

    def validate_parameters(self, num_parameters: int) -> None:
        if self.max_iterations is not None and self.max_iterations < 1:
            self.logger.warning(f'SLSQP max_iterations {self.max_iterations} should be >= 1')


class GenericConfig(OptimizerConfig):
    """Generic configuration for scipy optimizers not requiring custom logic.

    Passes `maxiter` into the options dict and returns `convergence_threshold`
    as the global `tol` argument for `scipy.optimize.minimize`.

    Optimizer-specific default `maxiter values.
    """

    # Research-backed default maxiter values per optimizer
    _DEFAULT_MAXITER: ClassVar[dict[str, int]] = {
        'Nelder-Mead': 5000,  # gradient-free simplex; slow - needs high budget
        'Powell': 10000,  # gradient-free conjugate-directions; one iter /approx n line searches
        'BFGS': 1000,  # quasi-Newton; fast convergence, limit by outer iterations
        'CG': 2000,  # conjugate gradient; moderate convergence
        'TNC': 500,  # truncated Newton; each step is expensive (inner CG)
    }

    # Optimizers that use 'maxfun' instead of 'maxiter'
    _USES_MAXFUN: ClassVar[set[str]] = {'TNC'}

    def __init__(
        self,
        optimizer_name: str,
        max_iterations: int | None = None,
        convergence_threshold: float | None = None,
    ):
        super().__init__(
            max_iterations=max_iterations, convergence_threshold=convergence_threshold
        )
        self.optimizer_name = optimizer_name

    def _effective_maxiter(self) -> int:
        """Return the effective maxiter for the options dict."""
        if self.max_iterations is not None:
            return self.max_iterations
        return self._DEFAULT_MAXITER.get(self.optimizer_name, 1000)

    def get_options(self, num_parameters: int) -> dict[str, Any]:
        key = 'maxfun' if self.optimizer_name in self._USES_MAXFUN else 'maxiter'
        return {
            'disp': False,
            key: self._effective_maxiter(),
        }

    def get_minimize_tol(self) -> float | None:
        return self.convergence_threshold

    def validate_parameters(self, num_parameters: int) -> None:
        if self.max_iterations is not None and self.max_iterations < 1:
            self.logger.warning(
                f'{self.optimizer_name} max_iterations {self.max_iterations} should be >= 1'
            )


class OptimizerConfigFactory:
    """Factory class for creating optimizer-specific configurations."""

    _configs: ClassVar[dict] = {
        'L-BFGS-B': LBFGSBConfig,
        'COBYLA': COBYLAConfig,
        'SLSQP': SLSQPConfig,
        'Nelder-Mead': lambda **kw: GenericConfig('Nelder-Mead', **kw),
        'Powell': lambda **kw: GenericConfig('Powell', **kw),
        'BFGS': lambda **kw: GenericConfig('BFGS', **kw),
        'CG': lambda **kw: GenericConfig('CG', **kw),
        'TNC': lambda **kw: GenericConfig('TNC', **kw),
    }

    @classmethod
    def create_config(
        cls,
        optimizer: str,
        max_iterations: int | None = None,
        convergence_threshold: float | None = None,
    ) -> OptimizerConfig:
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
            raise ValueError(
                f'Unsupported optimizer: {optimizer}. '
                f'Supported optimizers: {list(cls._configs.keys())}'
            )

        config_class = cls._configs[optimizer]
        return config_class(
            max_iterations=max_iterations, convergence_threshold=convergence_threshold
        )

    @classmethod
    def get_supported_optimizers(cls) -> list[str]:
        """Get list of supported optimizers."""
        return list(cls._configs.keys())

    @classmethod
    def register_optimizer(cls, name: str, config_class: type[OptimizerConfig]) -> None:
        """Register a new optimizer configuration class."""
        cls._configs[name] = config_class


def get_optimizer_configuration(
    optimizer: str,
    max_iterations: int | None = None,
    convergence_threshold: float | None = None,
    num_parameters: int = 0,
) -> tuple[dict[str, Any], float | None]:
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
        convergence_threshold=convergence_threshold,
    )

    config.validate_parameters(num_parameters)
    options = config.get_options(num_parameters)
    minimize_tol = config.get_minimize_tol()

    return options, minimize_tol
