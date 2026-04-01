"""Tests for the ScientificReferenceDatabase accuracy metrics."""

import pytest

from quantum_pipeline.monitoring.scientific_references import (
    ScientificReferenceDatabase,
    get_reference_database,
)


class TestAccuracyScore:
    """Test the logarithmic accuracy score formula."""

    @pytest.fixture
    def ref_db(self):
        return ScientificReferenceDatabase()

    def _score_for_error(self, ref_db, energy_error_hartree):
        """Compute accuracy score for H2 with a given error."""
        ref = ref_db.get_reference('H2')
        vqe_energy = ref.ground_state_energy_hartree + energy_error_hartree
        result = ref_db.calculate_accuracy_metrics('H2', vqe_energy)
        return result['accuracy_score']

    def test_zero_error_gives_100(self, ref_db):
        score = self._score_for_error(ref_db, 0.0)
        assert score == 100.0

    def test_tiny_error_gives_near_100(self, ref_db):
        score = self._score_for_error(ref_db, 1e-12)
        assert score == 100.0

    def test_chemical_accuracy_gives_high_score(self, ref_db):
        # 1 mHa = 0.001 Ha (chemical accuracy threshold)
        score = self._score_for_error(ref_db, 0.001)
        assert 90 < score < 98

    def test_moderate_error(self, ref_db):
        # 10 mHa = 0.01 Ha
        score = self._score_for_error(ref_db, 0.01)
        assert 75 < score < 85

    def test_large_error(self, ref_db):
        # 100 mHa = 0.1 Ha
        score = self._score_for_error(ref_db, 0.1)
        assert 55 < score < 65

    def test_very_large_error(self, ref_db):
        # 1000 mHa = 1 Ha
        score = self._score_for_error(ref_db, 1.0)
        assert 35 < score < 45

    def test_extreme_error_gives_zero(self, ref_db):
        # 100 Ha - absurdly bad
        score = self._score_for_error(ref_db, 100.0)
        assert score == 0.0

    def test_negative_error_same_as_positive(self, ref_db):
        # VQE below reference (negative error)
        score_pos = self._score_for_error(ref_db, 0.01)
        score_neg = self._score_for_error(ref_db, -0.01)
        assert score_pos == score_neg

    def test_score_monotonically_decreasing(self, ref_db):
        """Larger errors should give lower scores."""
        errors = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        scores = [self._score_for_error(ref_db, e) for e in errors]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_score_capped_at_100(self, ref_db):
        score = self._score_for_error(ref_db, 0.0)
        assert score <= 100.0

    def test_score_never_negative(self, ref_db):
        score = self._score_for_error(ref_db, 1000.0)
        assert score >= 0.0


class TestAccuracyMetrics:
    """Test the full calculate_accuracy_metrics method."""

    @pytest.fixture
    def ref_db(self):
        return ScientificReferenceDatabase()

    def test_no_reference_returns_none(self, ref_db):
        result = ref_db.calculate_accuracy_metrics('UnknownMolecule', -1.0)
        assert result['reference_available'] is False
        assert result['accuracy_score'] is None

    def test_energy_error_fields(self, ref_db):
        ref = ref_db.get_reference('H2')
        error_hartree = 0.005
        vqe_energy = ref.ground_state_energy_hartree + error_hartree
        result = ref_db.calculate_accuracy_metrics('H2', vqe_energy)
        assert abs(result['energy_error_hartree'] - error_hartree) < 1e-10
        assert abs(result['energy_error_millihartree'] - 5.0) < 1e-7

    def test_reference_fields_populated(self, ref_db):
        ref = ref_db.get_reference('H2')
        result = ref_db.calculate_accuracy_metrics('H2', ref.ground_state_energy_hartree)
        assert result['reference_available'] is True
        assert result['reference_energy_hartree'] == ref.ground_state_energy_hartree
        assert result['reference_method'] is not None
        assert result['reference_source'] is not None


class TestGetReferenceDatabase:
    """Test the global singleton accessor."""

    def test_returns_same_instance(self):
        db1 = get_reference_database()
        db2 = get_reference_database()
        assert db1 is db2

    def test_has_known_molecules(self):
        db = get_reference_database()
        molecules = db.get_all_molecules()
        assert 'H2' in molecules
        assert 'LiH' in molecules
