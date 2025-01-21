import pathlib
import numpy as np

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from quantum_pipeline.configs import settings
from quantum_pipeline.utils.dir import savePlot


class MoleculePlotter:
    """A class to handle 3D molecular structure visualization."""

    ATOM_COLORS: dict[str, str] = {
        'H': '#FFFFFF',  # white
        'C': '#808080',  # gray
        'N': '#0000FF',  # blue
        'O': '#FF0000',  # red
        'F': '#FFFF00',  # yellow
        'Cl': '#00FF00',  # green
        'Br': '#A52A2A',  # brown
        'I': '#800080',  # purple
        'P': '#FFA500',  # orange
        'S': '#FFD700',  # goldenYellow
        'Na': '#AAB7B8',  # silver
        'K': '#8E44AD',  # purple
        'Ca': '#BDC3C7',  # light gray
        'Fe': '#E74C3C',  # dark red
    }
    DEFAULT_COLOR = '#1f77b4'  # default blue for unknown elements

    def __init__(self, figsize: tuple[int, int] = (12, 10), legend_marker_size: int = 10):
        """
        Initialize the plotter with figure size and legend marker size.
        """
        self.fig = plt.figure(figsize=figsize)
        # subplot is dynamicaly typed, hinting to avoid errors later
        self.ax: Axes3D = self.fig.add_subplot(111, projection='3d')
        self.legend_marker_size = legend_marker_size

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB values."""
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                raise ValueError('Invalid hex color format.')
            return (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
        except Exception as e:
            raise ValueError(f'Error converting hex color: {hex_color}') from e

    @staticmethod
    def get_text_color(bg_color: str) -> str:
        """Determine optimal text color based on background color brightness."""
        try:
            rgb = MoleculePlotter.hex_to_rgb(bg_color)
            brightness = sum(rgb) / (3 * 255)
            return 'black' if brightness > 0.5 else 'white'
        except ValueError:
            return 'black'

    def _get_atom_color(self, symbol: str) -> str:
        """Get the color for an atom or the default color."""
        return self.ATOM_COLORS.get(symbol, self.DEFAULT_COLOR)

    def _validate_molecule(
        self, coords: np.ndarray, symbols: list[str], masses: list[float]
    ) -> None:
        """Validate that molecule data is consistent."""
        if len(coords) != len(symbols) or len(coords) != len(masses):
            raise ValueError('Mismatch in the lengths of coordinates, symbols, and masses.')

    def _calculate_atom_sizes(self, masses: list[float], base_size: float = 500) -> list[float]:
        """Calculate atom sizes based on masses with appropriate scaling."""
        min_mass = min(masses)
        return list(base_size * (1 + np.log(np.array(masses) / min_mass)))

    def _setup_axes(self) -> None:
        """Configure axes labels, title, and viewing angle."""
        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_zlabel('Z (Å)')
        self.ax.set_title('3D Molecular Structure')
        self.ax.view_init(elev=20, azim=45)
        self.ax.grid(True, alpha=0.3)

    def _set_plot_bounds(self, coords: np.ndarray, padding_factor: float = 0.3) -> None:
        """Set plot boundaries with padding."""
        ranges = [coords[:, i].max() - coords[:, i].min() for i in range(3)]
        max_range = max(ranges) / 2.0
        means = [coords[:, i].mean() for i in range(3)]
        padding = max_range * padding_factor

        for i, mean in enumerate(means):
            getattr(self.ax, f'set_{["x", "y", "z"][i]}lim')(
                mean - max_range - padding, mean + max_range + padding
            )
        self.ax.set_box_aspect([1.0, 1.0, 1.0])

    def plot_molecule(self, molecule) -> pathlib.Path:
        """
        Plot 3D structure of a molecule with improved visualization.

        Args:
            molecule: MoleculeInfo object.
        """
        coords = np.array(molecule.coords)
        symbols = molecule.symbols
        masses = molecule.masses

        # validate data
        self._validate_molecule(coords, symbols, masses)

        # setup basic plot parameters
        self._setup_axes()

        # calculate atom sizes with improved scaling
        sizes = self._calculate_atom_sizes(masses)

        # plot unique atoms
        legend_elements = []
        processed_symbols = set()

        for _, symbol in enumerate(symbols):
            if symbol in processed_symbols:
                continue
            indices = [j for j, s in enumerate(symbols) if s == symbol]
            symbol_coords = coords[indices]
            symbol_color = self._get_atom_color(symbol)

            # plot atoms
            self.ax.scatter(
                symbol_coords[:, 0],
                symbol_coords[:, 1],
                symbol_coords[:, 2],
                s=[sizes[j] for j in indices],
                c=[symbol_color],
                alpha=0.6,
                edgecolors='black',
                linewidth=2,
            )

            # Add to legend with fixed size marker
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    markerfacecolor=symbol_color,
                    markeredgecolor='black',
                    markersize=self.legend_marker_size,
                    label=symbol,
                )
            )
            processed_symbols.add(symbol)

            # Add atom labels
            for idx in indices:
                text_color = self.get_text_color(symbol_color)
                text = self.ax.text(
                    coords[idx, 0],
                    coords[idx, 1],
                    coords[idx, 2],
                    symbol,
                    fontsize=14,
                    color=text_color,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='semibold',
                )
                text.set_path_effects([pe.withStroke(linewidth=0.1, foreground='black')])

        # Set plot bounds
        self._set_plot_bounds(coords)

        # Add legend with consistent marker sizes
        self.ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True,
            edgecolor='black',
        )

        plt.tight_layout()
        plot_path = savePlot(
            plt,
            settings.MOLECULE_PLOT_DIR,
            settings.MOLECULE_PLOT,
            molecule.symbols,
        )

        return plot_path
