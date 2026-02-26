"""
test_optimizer_maxfun_fix.py

Tests to verify that L-BFGS-B properly respects maxfun limits for VQE optimization.
This addresses the issue where maxiter was being ignored in favor of tight tolerances.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration


class SimpleTestFunction:
    """Simple test function that counts calls"""

    def __init__(self):
        self.call_count = 0

    def reset(self):
        self.call_count = 0

    def quadratic(self, x):
        """Simple quadratic function"""
        self.call_count += 1
        return np.sum(x**2)

    def noisy_quadratic(self, x):
        """Quadratic with noise to simulate VQE measurement uncertainty"""
        self.call_count += 1
        clean_val = np.sum(x**2)
        noise = np.random.normal(0, 0.01 * np.sqrt(abs(clean_val) + 1e-8))
        return clean_val + noise


class TestLBFGSBMaxfunFix:
    """Test that L-BFGS-B respects maxfun parameter correctly"""

    def setup_method(self):
        """Setup for each test"""
        self.test_func = SimpleTestFunction()
        np.random.seed(42)  # For reproducible tests

    def test_lbfgsb_respects_maxfun_clean_function(self):
        """Test that L-BFGS-B respects maxfun with clean function"""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=10,  # Very low limit to force early termination
            convergence_threshold=None,
            num_parameters=5,
        )

        # Should have maxfun set
        assert 'maxfun' in options
        assert options['maxfun'] == 10

        # Start far from optimum to prevent quick convergence
        x0 = np.random.uniform(-10, 10, 5)
        self.test_func.reset()

        result = minimize(
            self.test_func.quadratic, x0, method='L-BFGS-B', options=options, tol=minimize_tol
        )

        # Should stop due to maxfun limit or be close to it
        assert (
            not result.success and 'EXCEEDS LIMIT' in result.message
        ) or self.test_func.call_count <= 15

    def test_lbfgsb_respects_maxfun_noisy_function(self):
        """Test that L-BFGS-B respects maxfun with noisy function (VQE-like)"""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=50,
            convergence_threshold=None,
            num_parameters=5,  # Smaller problem
        )

        assert options['maxfun'] == 50

        x0 = np.random.uniform(-2, 2, 5)  # Smaller range
        self.test_func.reset()

        result = minimize(
            self.test_func.noisy_quadratic,
            x0,
            method='L-BFGS-B',
            options=options,
            tol=minimize_tol,
        )

        # The fix is working if:
        # 1. It stops due to function limit (not natural convergence)
        # 2. Function calls are significantly less than the old default (15000)
        assert not result.success  # Should fail due to maxfun limit
        assert 'EXCEEDS LIMIT' in result.message
        assert self.test_func.call_count < 500  # Much better than 15000 default

    def test_lbfgsb_with_convergence_threshold(self):
        """Test L-BFGS-B with convergence threshold instead of maxfun"""
        options, _minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', max_iterations=None, convergence_threshold=1e-4, num_parameters=5
        )

        # Should not have maxfun when only convergence threshold is specified
        assert 'maxfun' not in options
        assert 'ftol' in options
        assert options['ftol'] == 1e-4
        assert options['gtol'] == 1e-4

    def test_lbfgsb_mutual_exclusion(self):
        """Test that both parameters raises error"""
        with pytest.raises(ValueError, match='mutually exclusive'):
            get_optimizer_configuration(
                optimizer='L-BFGS-B',
                max_iterations=25,
                convergence_threshold=1e-6,
                num_parameters=5,
            )

    def test_cobyla_unchanged(self):
        """Test that COBYLA behavior is unchanged"""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=30, convergence_threshold=None, num_parameters=5
        )

        # COBYLA should still use maxiter, not maxfun
        assert 'maxiter' in options
        assert options['maxiter'] == 30
        assert 'maxfun' not in options

        x0 = np.random.random(5)
        self.test_func.reset()

        minimize(self.test_func.quadratic, x0, method='COBYLA', options=options, tol=minimize_tol)

        # COBYLA should respect maxiter exactly
        assert self.test_func.call_count <= 30

    def test_configuration_backward_compatibility(self):
        """Test that existing code calling patterns still work"""
        # Test the old way of calling should still work
        config_result = get_optimizer_configuration(
            optimizer='L-BFGS-B', max_iterations=100, num_parameters=20
        )

        assert len(config_result) == 2  # options, minimize_tol
        options, _minimize_tol = config_result
        assert isinstance(options, dict)
        assert 'maxfun' in options
        assert options['maxfun'] == 100


if __name__ == '__main__':
    pytest.main([__file__])
