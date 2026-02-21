"""Tests for operator coefficient visualization (OperatorViewer)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


class TestPlotOperatorCoefficients:
    """Test OperatorViewer.plot_operator_coefficients."""

    @pytest.fixture
    def simple_qubit_op(self):
        """Qubit op with a few significant terms (list of (label, coeff) tuples)."""
        return [
            ('IZ', 0.5 + 0.1j),
            ('ZI', -0.3 + 0.2j),
            ('XX', 0.8 + 0.0j),
        ]

    @pytest.fixture
    def op_with_negligible(self):
        """Qubit op containing terms below the default threshold (1e-2)."""
        return [
            ('IZ', 0.5 + 0.0j),
            ('ZI', 0.001 + 0.001j),   # negligible
            ('XX', 0.002 + 0.0j),     # negligible
            ('YY', -0.4 + 0.0j),
        ]

    @pytest.fixture
    def large_qubit_op(self):
        """Qubit op with > 50 significant terms to trigger subsampling."""
        return [(f'P{i}', 0.5 + 0.1j) for i in range(80)]

    # ------------------------------------------------------------------ #
    # Happy-path tests
    # ------------------------------------------------------------------ #

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_returns_plot_path(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        result = OperatorViewer.plot_operator_coefficients(
            simple_qubit_op, symbols='H2'
        )

        assert result == '/tmp/plot.png'
        mock_save.assert_called_once()
        mock_plt.close.assert_called_once_with(mock_fig)

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_creates_horizontal_bar_chart(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        OperatorViewer.plot_operator_coefficients(simple_qubit_op, symbols='H2')

        # Two barh calls: real parts and imaginary parts
        assert mock_ax.barh.call_count == 2
        mock_ax.set_title.assert_called_once()
        mock_ax.legend.assert_called_once()

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_custom_title(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        OperatorViewer.plot_operator_coefficients(
            simple_qubit_op, symbols='H2', title='My Title'
        )

        mock_ax.set_title.assert_called_once_with('My Title', fontsize=14)

    # ------------------------------------------------------------------ #
    # Filtering / threshold tests
    # ------------------------------------------------------------------ #

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_negligible_terms_grouped_as_other(self, mock_plt, mock_save, op_with_negligible):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        OperatorViewer.plot_operator_coefficients(op_with_negligible, symbols='H2')

        # yticklabels should include 'Other' for grouped negligible terms
        ytick_call = mock_ax.set_yticklabels.call_args
        labels = ytick_call[0][0]
        assert 'Other' in labels
        # significant: IZ, YY → 2 significant + 1 Other = 3
        assert len(labels) == 3

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_high_threshold_groups_all(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        OperatorViewer.plot_operator_coefficients(
            simple_qubit_op, symbols='H2', threshold=10.0
        )

        labels = mock_ax.set_yticklabels.call_args[0][0]
        # All terms are below threshold=10 → only 'Other'
        assert labels == ['Other']

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_zero_threshold_shows_all(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        OperatorViewer.plot_operator_coefficients(
            simple_qubit_op, symbols='H2', threshold=0.0
        )

        labels = mock_ax.set_yticklabels.call_args[0][0]
        assert len(labels) == 3  # all terms, no 'Other'
        assert 'Other' not in labels

    # ------------------------------------------------------------------ #
    # Subsampling (max_terms) tests
    # ------------------------------------------------------------------ #

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_max_terms_limits_output(self, mock_plt, mock_save, large_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        OperatorViewer.plot_operator_coefficients(
            large_qubit_op, symbols='H2', max_terms=20
        )

        labels = mock_ax.set_yticklabels.call_args[0][0]
        assert len(labels) <= 20

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_small_op_no_subsampling(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        OperatorViewer.plot_operator_coefficients(
            simple_qubit_op, symbols='H2', max_terms=50
        )

        labels = mock_ax.set_yticklabels.call_args[0][0]
        assert len(labels) == 3  # no subsampling needed

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_single_term(self, mock_plt, mock_save):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        op = [('ZZ', 1.0 + 0.0j)]
        result = OperatorViewer.plot_operator_coefficients(op, symbols='H2')
        assert result == '/tmp/plot.png'

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_purely_imaginary_coefficients(self, mock_plt, mock_save):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_save.return_value = '/tmp/plot.png'

        op = [('XY', 0.0 + 0.5j), ('YX', 0.0 + 0.3j)]
        OperatorViewer.plot_operator_coefficients(op, symbols='LiH')

        mock_plt.tight_layout.assert_called_once()


class TestPlotComplexCoefficientsPolar:
    """Test OperatorViewer.plot_complex_coefficients_polar."""

    @pytest.fixture
    def simple_qubit_op(self):
        return [
            ('IZ', 0.5 + 0.1j),
            ('ZI', -0.3 + 0.2j),
            ('XX', 0.8 + 0.0j),
        ]

    @pytest.fixture
    def op_with_negligible(self):
        return [
            ('IZ', 0.5 + 0.0j),
            ('ZI', 0.001 + 0.001j),
            ('XX', 0.002 + 0.0j),
            ('YY', -0.4 + 0.0j),
        ]

    @pytest.fixture
    def large_qubit_op(self):
        return [(f'P{i}', 0.5 + 0.1j) for i in range(80)]

    # ------------------------------------------------------------------ #
    # Happy-path tests
    # ------------------------------------------------------------------ #

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_returns_plot_path(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        result = OperatorViewer.plot_complex_coefficients_polar(
            simple_qubit_op, symbols='H2'
        )

        assert result == '/tmp/polar.png'
        mock_save.assert_called_once()
        mock_plt.close.assert_called_once_with(mock_fig)

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_creates_magnitude_and_phase_subplots(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        OperatorViewer.plot_complex_coefficients_polar(simple_qubit_op, symbols='H2')

        # magnitude bar on ax1, phase bar on ax2
        mock_ax1.bar.assert_called_once()
        mock_ax2.bar.assert_called_once()
        mock_ax1.set_title.assert_called_once()
        mock_ax2.set_title.assert_called_once()

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_custom_title_propagated(self, mock_plt, mock_save, simple_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        OperatorViewer.plot_complex_coefficients_polar(
            simple_qubit_op, symbols='H2', title='Polar View'
        )

        mag_title = mock_ax1.set_title.call_args[0][0]
        phase_title = mock_ax2.set_title.call_args[0][0]
        assert 'Polar View' in mag_title
        assert 'Polar View' in phase_title

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_negligible_terms_grouped_polar(self, mock_plt, mock_save, op_with_negligible):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        OperatorViewer.plot_complex_coefficients_polar(op_with_negligible, symbols='H2')

        labels = mock_ax1.set_xticklabels.call_args[0][0]
        assert 'Other' in labels

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_max_terms_limits_polar(self, mock_plt, mock_save, large_qubit_op):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        OperatorViewer.plot_complex_coefficients_polar(
            large_qubit_op, symbols='H2', max_terms=15
        )

        labels = mock_ax1.set_xticklabels.call_args[0][0]
        assert len(labels) <= 15

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_all_real_coefficients(self, mock_plt, mock_save):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        op = [('ZZ', 1.0 + 0.0j), ('XI', -0.5 + 0.0j)]
        result = OperatorViewer.plot_complex_coefficients_polar(op, symbols='H2')
        assert result == '/tmp/polar.png'

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_single_term_polar(self, mock_plt, mock_save):
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        op = [('II', 0.3 + 0.4j)]
        OperatorViewer.plot_complex_coefficients_polar(op, symbols='LiH')
        mock_plt.tight_layout.assert_called_once()

    @patch('quantum_pipeline.visual.operator.savePlot')
    @patch('quantum_pipeline.visual.operator.plt')
    def test_all_negligible_no_crash(self, mock_plt, mock_save):
        """When every term is below threshold, no 'Other' added if sums are zero."""
        from quantum_pipeline.visual.operator import OperatorViewer

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_save.return_value = '/tmp/polar.png'

        op = [('ZZ', 0.005 + 0.005j)]
        # magnitude ≈ 0.007 < default threshold 0.01 but sum > 0 → 'Other'
        OperatorViewer.plot_complex_coefficients_polar(op, symbols='H2')
        labels = mock_ax1.set_xticklabels.call_args[0][0]
        assert 'Other' in labels
