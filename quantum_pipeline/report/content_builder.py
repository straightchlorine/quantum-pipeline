import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from quantum_pipeline.report.configuration import ReportConfiguration
from quantum_pipeline.structures.vqe_observation import VQEProcess
from quantum_pipeline.utils.logger import get_logger
from quantum_pipeline.visual.energy import EnergyPlotter
from quantum_pipeline.visual.molecule import MoleculePlotter
from quantum_pipeline.visual.operator import OperatorViewer


class ReportContentBuilder:
    """Manages the content and structure of the report."""

    def __init__(self, styles=None):
        """
        Initialize the content builder.

        Args:
            styles (dict, optional): Custom paragraph styles.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.report_config = ReportConfiguration()
        self.content: List[Union[Paragraph, Spacer, Table, Tuple]] = []
        self.styles = styles or getSampleStyleSheet()
        self._add_custom_styles()

    def _add_custom_styles(self):
        """Add custom paragraph styles."""
        self.styles.add(
            ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                textColor=colors.darkblue,
                spaceAfter=12,
            )
        )

    def add_insight(self, title: str, insight: str):
        """
        Add a textual insight section to the report.

        Args:
            title (str): Title of the insight section.
            insight (str): Text describing the insight.
        """
        self.content.extend(
            [
                Paragraph(title, self.styles['CustomHeading']),
                Spacer(1, 10),
                Paragraph(insight, self.styles['BodyText']),
                Spacer(1, 5),
            ]
        )

    def add_header(self, header: str):
        """
        Add a textual header section to the report.

        Args:
            header (str): Header of the section.
        """
        self.content.extend(
            [
                Paragraph(header, self.styles['CustomHeading']),
            ]
        )

    def add_table(
        self,
        data: List[List],
        col_widths: Optional[List[float]] = None,
        custom_styles: Optional[List[Tuple]] = None,
    ):
        """
        Add a table to the report.

        Args:
            data (List[List]): 2D list representing table contents.
            col_widths (List[float], optional): Column widths.
            custom_styles (List[Tuple], optional): Custom table styles.
        """
        col_widths = col_widths or [4 * cm, 5 * cm]
        table = Table(data, colWidths=col_widths)

        # Use custom styles if provided, otherwise use default
        styles = custom_styles or self.report_config.table_styles
        table.setStyle(TableStyle(styles))

        self.content.extend([table, Spacer(1, 20)])

    def add_metrics(self, metrics: Dict[str, Union[str, float, int]]):
        """
        Add a metrics table to the report.

        Args:
            metrics (Dict): Dictionary of metrics to include.
        """
        data = [['Metric', 'Value']] + [[str(k), str(v)] for k, v in metrics.items()]
        self.add_table(data)

    def append_plot_path(self, plot_path: str | Path, sizes: Tuple[int, int]):
        if os.path.exists(plot_path):
            self.content.append(('plot', plot_path, sizes))
        else:
            raise FileNotFoundError(f'Plot path {plot_path} does not exist.')

    def add_molecule_plot(self, molecule: MoleculeInfo, plotter: Optional[MoleculePlotter] = None):
        """
        Add a molecule visualization to the report.

        Args:
            molecule (MoleculeInfo): Molecule to visualize.
            plotter (MoleculePlotter, optional): Custom molecule plotter.
        """
        plotter = plotter or MoleculePlotter()
        plot_path = plotter.plot_molecule(molecule)
        self.append_plot_path(plot_path, self.report_config.molecule_size)

    def add_convergence_plot(self, iterations: list[VQEProcess], symbols, max_points=100):
        """
        Add a molecule visualization to the report.

        Args:
            molecule (MoleculeInfo): Molecule to visualize.
            plotter (MoleculePlotter, optional): Custom molecule plotter.
        """
        plot_path = EnergyPlotter(iterations, symbols, max_points).plot_convergence()
        self.append_plot_path(plot_path, self.report_config.convergence_size)

    def add_operator_coeff_plot(self, qubit_op, symbols, title='Real Operator Coefficients'):
        """
        Add a plot of the operator coefficients to the report.
        Args:
            qubit_op: Qiskit's PauliSumOp or similar operator.
            title (str, optional): Title of the plot.
        """
        plot_path = OperatorViewer().plot_operator_coefficients(qubit_op, symbols, title)
        self.append_plot_path(plot_path, self.report_config.op_coeff_size)

    def add_complex_coeff_plot(self, qubit_op, symbols, title='Polar Operator Coefficients'):
        """
        Add a plot of the operator coefficients in polar coordinates to
        the report.
        Args:
            qubit_op: Qiskit's PauliSumOp or similar operator.
            title (str, optional): Title of the plot.
        """
        plot_path = OperatorViewer().plot_complex_coefficients_polar(qubit_op, symbols, title)
        self.append_plot_path(plot_path, self.report_config.complex_op_coeff_size)

    def new_page(self):
        """Insert a new page marker in the report content."""
        self.content.append(('new_page',))

    def get_content(self):
        """
        Get the current report content.

        Returns:
            List: Report content items.
        """
        return self.content
