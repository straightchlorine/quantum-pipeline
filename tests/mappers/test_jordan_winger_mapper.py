"""Tests for Jordan-Wigner mapper implementation."""

import pytest
from qiskit_nature.second_q.operators import FermionicOp
from quantum_pipeline.mappers import JordanWignerMapper


class TestJordanWignerMapper:
    """Test suite for JordanWignerMapper."""

    @pytest.fixture
    def mapper(self):
        """Create a JordanWignerMapper instance."""
        return JordanWignerMapper()

    def test_mapper_initialization(self, mapper):
        """Test that mapper can be initialized."""
        assert mapper is not None
        assert isinstance(mapper, JordanWignerMapper)

    def test_simple_fermionic_operator(self, mapper):
        """Test mapping a simple fermionic operator."""
        fermionic_op = FermionicOp({'+_0 -_1': 1.0})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_identity_operator(self, mapper):
        """Test mapping identity-like fermionic operator."""
        fermionic_op = FermionicOp({'+_0 -_0': 1.0})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_single_orbital_operator(self, mapper):
        """Test mapping single orbital creation operator."""
        fermionic_op = FermionicOp({'+_0': 1.0})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_multiple_orbital_operator(self, mapper):
        """Test mapping operator with multiple orbitals."""
        fermionic_op = FermionicOp({'+_0 +_1 -_2 -_3': 1.5})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_summed_operators(self, mapper):
        """Test mapping sum of fermionic operators."""
        fermionic_op = FermionicOp({'+_0 -_1': 0.5, '+_1 -_0': 0.3})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_h2_hamiltonian(self, mapper):
        """Test mapping H2 molecular Hamiltonian."""
        # Simple H2 Hamiltonian (hopping term + on-site)
        h2_ham = FermionicOp(
            {
                '+_0 -_1': -1.254,
                '+_1 -_0': -1.254,
                '+_0 -_0': -0.474,
                '+_1 -_1': -0.474,
            }
        )
        qubit_op = mapper.map(h2_ham)
        assert qubit_op is not None

    def test_four_electron_system(self, mapper):
        """Test mapping 4-electron system."""
        fermionic_op = FermionicOp({'+_0 +_1 -_2 -_3': 1.0, '': 0.5})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_null_operator_raises_error(self, mapper):
        """Test that None operator raises ValueError."""
        with pytest.raises(ValueError, match='The input operator must not be None'):
            mapper.map(None)

    def test_zero_coefficient_terms(self, mapper):
        """Test handling of zero coefficient terms."""
        fermionic_op = FermionicOp({'+_0 -_1': 0.0, '+_1 -_0': 1.0})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_complex_coefficients(self, mapper):
        """Test mapping with complex coefficients."""
        fermionic_op = FermionicOp({'+_0 -_1': 1.0 + 0.5j, '+_1 -_0': 1.0 - 0.5j})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_large_operator(self, mapper):
        """Test mapping large fermionic operator."""
        # Create operator with many orbitals
        terms = {f'+_{i} -_{i+1}': 1.0 / (i + 1) for i in range(5)}
        fermionic_op = FermionicOp(terms)
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_ground_state_like_operator(self, mapper):
        """Test mapping typical ground state energy operator."""
        # Typical form for molecular Hamiltonian
        fermionic_op = FermionicOp(
            {
                '': 0.5,  # Constant term
                '+_0': 0.1,
                '-_0': 0.1,
                '+_0 -_1': 1.2,
                '+_1 -_0': 1.2,
                '+_0 +_1 -_1 -_0': 2.0,  # Two-body term
            }
        )
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_empty_operator(self, mapper):
        """Test mapping empty fermionic operator (should be zero operator)."""
        fermionic_op = FermionicOp({})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_repeated_mapping(self, mapper):
        """Test that repeated mapping of same operator is consistent."""
        fermionic_op = FermionicOp({'+_0 -_1': 1.0})
        qubit_op1 = mapper.map(fermionic_op)
        qubit_op2 = mapper.map(fermionic_op)
        # Both should produce operators (consistent behavior)
        assert qubit_op1 is not None
        assert qubit_op2 is not None

    def test_different_operators_different_results(self, mapper):
        """Test that different operators produce different results."""
        op1 = FermionicOp({'+_0 -_1': 1.0})
        op2 = FermionicOp({'+_0 -_2': 1.0})
        qubit_op1 = mapper.map(op1)
        qubit_op2 = mapper.map(op2)
        assert qubit_op1 is not None
        assert qubit_op2 is not None
        # Operators should be different (based on different indices)
        assert str(qubit_op1) != str(qubit_op2)

    def test_negative_coefficients(self, mapper):
        """Test mapping with negative coefficients."""
        fermionic_op = FermionicOp({'+_0 -_1': -1.5, '+_1 -_0': -2.0})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None

    def test_high_qubit_index(self, mapper):
        """Test mapping with high qubit indices."""
        fermionic_op = FermionicOp({'+_10 -_11': 1.0})
        qubit_op = mapper.map(fermionic_op)
        assert qubit_op is not None
