"""Tests for basis_sets module — validate_basis_set()."""

import pytest

from quantum_pipeline.configs.settings import SUPPORTED_BASIS_SETS
from quantum_pipeline.drivers.basis_sets import validate_basis_set


class TestValidateBasisSet:
    """Tests for the validate_basis_set function."""

    @pytest.mark.parametrize("basis_set", SUPPORTED_BASIS_SETS)
    def test_valid_basis_sets_accepted(self, basis_set):
        """Each supported basis set should pass validation without error."""
        validate_basis_set(basis_set)  # should not raise

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "sto-3g",        # close but wrong format
            "STO3G",         # wrong case
            "6-31g*",        # unsupported variant
            "cc-pvtz",       # unsupported set
            "",              # empty string
            "nonexistent",   # arbitrary string
        ],
    )
    def test_invalid_basis_sets_rejected(self, invalid_input):
        """Unsupported basis sets must raise ValueError."""
        with pytest.raises(ValueError):
            validate_basis_set(invalid_input)

    def test_none_input_raises(self):
        """None should raise (not silently pass)."""
        with pytest.raises((ValueError, TypeError)):
            validate_basis_set(None)

    def test_numeric_input_raises(self):
        """Numeric input should raise."""
        with pytest.raises((ValueError, TypeError)):
            validate_basis_set(631)

    def test_list_input_raises(self):
        """A list input should raise."""
        with pytest.raises((ValueError, TypeError)):
            validate_basis_set(["sto3g"])
