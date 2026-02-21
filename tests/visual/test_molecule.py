"""Tests for 3D molecule visualization (MoleculePlotter)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_molecule(symbols, coords, masses):
    """Create a lightweight molecule mock."""
    mol = MagicMock()
    mol.symbols = symbols
    mol.coords = coords
    mol.masses = masses
    return mol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def h2_molecule():
    """Simple H2 molecule."""
    return _make_molecule(
        symbols=['H', 'H'],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        masses=[1.008, 1.008],
    )


@pytest.fixture
def water_molecule():
    """H2O molecule with three distinct atom types."""
    return _make_molecule(
        symbols=['O', 'H', 'H'],
        coords=[
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
        masses=[15.999, 1.008, 1.008],
    )


@pytest.fixture
def multi_element_molecule():
    """Molecule with many different elements."""
    return _make_molecule(
        symbols=['C', 'N', 'O', 'S', 'H'],
        coords=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        masses=[12.011, 14.007, 15.999, 32.06, 1.008],
    )


# ---------------------------------------------------------------------------
# hex_to_rgb
# ---------------------------------------------------------------------------

class TestHexToRgb:
    """Test MoleculePlotter.hex_to_rgb static method."""

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_basic_colors(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        assert MoleculePlotter.hex_to_rgb('#FF0000') == (255, 0, 0)
        assert MoleculePlotter.hex_to_rgb('#00FF00') == (0, 255, 0)
        assert MoleculePlotter.hex_to_rgb('#0000FF') == (0, 0, 255)

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_black_and_white(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        assert MoleculePlotter.hex_to_rgb('#000000') == (0, 0, 0)
        assert MoleculePlotter.hex_to_rgb('#FFFFFF') == (255, 255, 255)

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_without_hash(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        assert MoleculePlotter.hex_to_rgb('FF0000') == (255, 0, 0)

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_invalid_hex_raises(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        with pytest.raises(ValueError):
            MoleculePlotter.hex_to_rgb('#GG0000')

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_short_hex_raises(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        with pytest.raises(ValueError):
            MoleculePlotter.hex_to_rgb('#FFF')


# ---------------------------------------------------------------------------
# get_text_color
# ---------------------------------------------------------------------------

class TestGetTextColor:
    """Test MoleculePlotter.get_text_color static method."""

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_dark_background_returns_white(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        assert MoleculePlotter.get_text_color('#000000') == 'white'
        assert MoleculePlotter.get_text_color('#0000FF') == 'white'

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_light_background_returns_black(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        assert MoleculePlotter.get_text_color('#FFFFFF') == 'black'
        assert MoleculePlotter.get_text_color('#FFFF00') == 'black'

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_invalid_color_returns_black(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()

        assert MoleculePlotter.get_text_color('#ZZZ') == 'black'


# ---------------------------------------------------------------------------
# _get_atom_color
# ---------------------------------------------------------------------------

class TestGetAtomColor:
    """Test MoleculePlotter._get_atom_color."""

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_known_elements(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        assert plotter._get_atom_color('H') == '#FFFFFF'
        assert plotter._get_atom_color('O') == '#FF0000'
        assert plotter._get_atom_color('C') == '#808080'

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_unknown_element_returns_default(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        assert plotter._get_atom_color('Unobtanium') == MoleculePlotter.DEFAULT_COLOR


# ---------------------------------------------------------------------------
# _validate_molecule
# ---------------------------------------------------------------------------

class TestValidateMolecule:
    """Test MoleculePlotter._validate_molecule."""

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_valid_data_passes(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        coords = np.array([[0, 0, 0], [1, 0, 0]])
        plotter._validate_molecule(coords, ['H', 'H'], [1.0, 1.0])

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_mismatched_lengths_raises(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        coords = np.array([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(ValueError, match='Mismatch'):
            plotter._validate_molecule(coords, ['H'], [1.0, 1.0])

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_mismatched_masses_raises(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        coords = np.array([[0, 0, 0]])
        with pytest.raises(ValueError, match='Mismatch'):
            plotter._validate_molecule(coords, ['H'], [1.0, 2.0])


# ---------------------------------------------------------------------------
# _calculate_atom_sizes
# ---------------------------------------------------------------------------

class TestCalculateAtomSizes:
    """Test MoleculePlotter._calculate_atom_sizes."""

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_equal_masses_same_size(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        sizes = plotter._calculate_atom_sizes([1.0, 1.0, 1.0])
        assert len(sizes) == 3
        assert sizes[0] == sizes[1] == sizes[2]

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_heavier_atoms_larger(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        sizes = plotter._calculate_atom_sizes([1.0, 16.0])
        assert sizes[1] > sizes[0]

    @patch('quantum_pipeline.visual.molecule.plt')
    def test_custom_base_size(self, mock_plt):
        from quantum_pipeline.visual.molecule import MoleculePlotter
        mock_plt.figure.return_value = MagicMock()
        mock_plt.figure.return_value.add_subplot.return_value = MagicMock()

        plotter = MoleculePlotter()
        sizes_default = plotter._calculate_atom_sizes([1.0], base_size=500)
        sizes_small = plotter._calculate_atom_sizes([1.0], base_size=100)
        assert sizes_default[0] > sizes_small[0]


# ---------------------------------------------------------------------------
# plot_molecule (integration-level, heavily mocked)
# ---------------------------------------------------------------------------

class TestPlotMolecule:
    """Test MoleculePlotter.plot_molecule with mocked matplotlib."""

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_returns_plot_path(self, mock_plt, mock_save, h2_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        result = plotter.plot_molecule(h2_molecule)

        assert result == '/tmp/mol.png'
        mock_save.assert_called_once()

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_scatter_called_per_unique_symbol(self, mock_plt, mock_save, water_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        plotter.plot_molecule(water_molecule)

        # H2O has 2 unique symbols (O, H) → 2 scatter calls
        assert mock_ax.scatter.call_count == 2

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_scatter_per_unique_multi_element(self, mock_plt, mock_save, multi_element_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        plotter.plot_molecule(multi_element_molecule)

        # C, N, O, S, H → 5 unique symbols → 5 scatter calls
        assert mock_ax.scatter.call_count == 5

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_axes_configured(self, mock_plt, mock_save, h2_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        plotter.plot_molecule(h2_molecule)

        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        mock_ax.set_zlabel.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_ax.view_init.assert_called_once_with(elev=20, azim=45)

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_legend_created(self, mock_plt, mock_save, water_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        plotter.plot_molecule(water_molecule)

        mock_ax.legend.assert_called_once()

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_tight_layout_called(self, mock_plt, mock_save, h2_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        plotter.plot_molecule(h2_molecule)

        mock_plt.tight_layout.assert_called_once()

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_invalid_molecule_raises(self, mock_plt, mock_save):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig

        bad_mol = _make_molecule(
            symbols=['H'],
            coords=[[0, 0, 0], [1, 0, 0]],  # 2 coords but 1 symbol
            masses=[1.0],
        )

        plotter = MoleculePlotter()
        with pytest.raises(ValueError, match='Mismatch'):
            plotter.plot_molecule(bad_mol)

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_atom_labels_added(self, mock_plt, mock_save, h2_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        plotter.plot_molecule(h2_molecule)

        # H2 has 2 atoms, both H, one scatter call, 2 text calls for labels
        assert mock_ax.text.call_count == 2

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_plot_bounds_set(self, mock_plt, mock_save, h2_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter()
        plotter.plot_molecule(h2_molecule)

        mock_ax.set_box_aspect.assert_called_once_with([1.0, 1.0, 1.0])

    @patch('quantum_pipeline.visual.molecule.savePlot')
    @patch('quantum_pipeline.visual.molecule.plt')
    def test_custom_figsize(self, mock_plt, mock_save, h2_molecule):
        from quantum_pipeline.visual.molecule import MoleculePlotter

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_save.return_value = '/tmp/mol.png'

        plotter = MoleculePlotter(figsize=(20, 16))
        mock_plt.figure.assert_called_with(figsize=(20, 16))
