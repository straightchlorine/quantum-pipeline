"""Factory for optimizer-specific scipy.optimize.minimize configurations used by VQE."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from quantum_pipeline.configs.constants import (
    COBYLA_DEFAULT_MAXITER,
    LBFGSB_DEFAULT_MAXITER,
    SLSQP_DEFAULT_MAXITER,
)


class OptimizerConfig(ABC):
    """Abstract base class for optimizer configurations."""

    def __init__(
        self, max_iterations: int | None = None, convergence_threshold: float | None = None
    ):
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

    `max_iterations` sets `maxiter`/`maxfun` with `ftol`/`gtol` at 1e-15 so the whole budget
    is spent; `convergence_threshold` sets `ftol`/`gtol` under a high `maxiter` cap; with
    neither, only `maxiter` is set and scipy tolerances apply.
    """

    def get_options(self, num_parameters: int) -> dict[str, Any]:
        options: dict[str, Any] = {'disp': False}

        if self.max_iterations is not None:
            # Set both maxfun and maxiter to prevent hanging.
            # Use tight tolerances to ensure the full iteration budget is used
            # (scipy's defaults would cause early stopping).
            options['maxfun'] = self.max_iterations
            options['maxiter'] = self.max_iterations
            options['ftol'] = 1e-15
            options['gtol'] = 1e-15

        elif self.convergence_threshold is not None:
            # Cap iterations high so ftol/gtol decide when to stop.
            options['maxiter'] = LBFGSB_DEFAULT_MAXITER
            options['ftol'] = self.convergence_threshold
            options['gtol'] = self.convergence_threshold

        else:
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

    Only `maxiter` goes into the options; `convergence_threshold` is passed as scipy `tol`.
    """

    def get_options(self, num_parameters: int) -> dict[str, Any]:
        maxiter = (
            self.max_iterations if self.max_iterations is not None else COBYLA_DEFAULT_MAXITER
        )

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

    `maxiter` always goes into the options; `convergence_threshold` is set both as option
    `ftol` and as scipy `tol`.
    """

    def get_options(self, num_parameters: int) -> dict[str, Any]:
        maxiter = self.max_iterations if self.max_iterations is not None else SLSQP_DEFAULT_MAXITER

        options: dict[str, Any] = {'disp': False, 'maxiter': maxiter}

        if self.convergence_threshold is not None:
            options['ftol'] = self.convergence_threshold

        return options

    def get_minimize_tol(self) -> float | None:
        return self.convergence_threshold

    def validate_parameters(self, num_parameters: int) -> None:
        if self.max_iterations is not None and self.max_iterations < 1:
            self.logger.warning(f'SLSQP max_iterations {self.max_iterations} should be >= 1')


class GenericConfig(OptimizerConfig):
    """Generic configuration for scipy optimizers not requiring custom logic.

    Sets `maxiter` (`maxfun` for TNC) from `max_iterations` or a per-optimizer default;
    `convergence_threshold` is passed as scipy `tol`.
    """

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

    _configs: ClassVar[dict[str, Callable[..., OptimizerConfig]]] = {
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
            max_iterations: Maximum number of iterations (mutually exclusive with convergence_threshold)
            convergence_threshold: Convergence threshold for optimization

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
        return list(cls._configs.keys())

    @classmethod
    def register_optimizer(cls, name: str, config_class: Callable[..., OptimizerConfig]) -> None:
        cls._configs[name] = config_class


def get_optimizer_configuration(
    optimizer: str,
    max_iterations: int | None = None,
    convergence_threshold: float | None = None,
    num_parameters: int = 0,
) -> tuple[dict[str, Any], float | None]:
    """Return (options dict, scipy `tol`) for the given optimizer.

    Args:
        num_parameters: Only used to warn when a COBYLA budget is below `num_parameters + 2`.

    Raises:
        ValueError: If the optimizer is unsupported or both `max_iterations` and
            `convergence_threshold` are set.
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
