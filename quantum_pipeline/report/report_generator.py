"""
report_generator.py

Generates PDF reports with improved modularity and flexibility.
"""

import os
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from quantum_pipeline.configs import settings
from quantum_pipeline.utils.dir import getGraphPath
from quantum_pipeline.utils.observation import VQEProcess
from quantum_pipeline.visual.energy import EnergyPlotter
from quantum_pipeline.visual.molecule import MoleculePlotter
from quantum_pipeline.utils.logger import get_logger
from quantum_pipeline.visual.operator import OperatorViewer


class ReportConfiguration:
    """Configuration parameters for report generation."""

    def __init__(self):
        self.margin = 50
        self.page_size = letter
        self.image_width = 500
        self.image_height = 400
        self.table_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]
        self.molecule_size = (500, 400)
        self.op_coeff_size = (415, 244)
        self.complex_op_coeff_size = (420, 300)
        self.convergence_size = (600, 360)


class ReportContentBuilder:
    """Manages the content and structure of the report."""

    def __init__(self, styles=None):
        """
        Initialize the content builder.

        Args:
            styles (dict, optional): Custom paragraph styles.
        """
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


class PDFRenderer:
    """Handles PDF rendering for the report."""

    def __init__(self, config: Optional[ReportConfiguration] = None):
        """
        Initialize the PDF renderer.

        Args:
            config (ReportConfiguration, optional): Report configuration.
        """
        self.config = config or ReportConfiguration()
        self.styles = getSampleStyleSheet()

    def render(self, content: List, output_path: str):
        """
        Render report content to a PDF.

        Args:
            content (List): Report content to render.
            output_path (str): Path to save the PDF.
        """
        canvas_obj = canvas.Canvas(output_path, pagesize=self.config.page_size)
        _, max_height = self.config.page_size
        y_position = max_height - self.config.margin

        for item in content:
            y_position = self._render_item(canvas_obj, item, y_position, max_height)

        canvas_obj.save()

    def _render_item(self, canvas_obj, item, y_position, max_height):
        """
        Render individual report items.

        Args:
            canvas_obj (Canvas): PDF canvas.
            item: Report content item.
            y_position (float): Current vertical position.
            max_height (float): Maximum page height.

        Returns:
            float: Updated vertical position.
        """
        if item == ('new_page',):
            canvas_obj.showPage()
            return max_height - self.config.margin

        if isinstance(item, (Paragraph, Spacer)):
            return self._render_text_element(canvas_obj, item, y_position)

        if isinstance(item, Table):
            return self._render_table(canvas_obj, item, y_position, max_height)

        if isinstance(item, tuple) and item[0] == 'plot':
            try:
                size = item[2]
            except IndexError:
                size = (self.config.image_width, self.config.image_height)

            return self._render_image(canvas_obj, item[1], y_position, max_height, size)

        return y_position

    def _render_text_element(self, canvas_obj, element, y_position):
        """
        Render text-based elements.

        Args:
            canvas_obj (Canvas): PDF canvas.
            element (Paragraph/Spacer): Text element to render.
            y_position (float): Current vertical position.

        Returns:
            float: Updated vertical position.
        """
        element.wrapOn(canvas_obj, (A4[0] - self.config.margin), y_position)

        element.drawOn(canvas_obj, self.config.margin, y_position - element.height)
        return y_position - element.height

    def _render_table(self, canvas_obj, table, y_position, max_height):
        """
        Render a table to the PDF.

        Args:
            canvas_obj (Canvas): PDF canvas.
            table (Table): Table to render.
            y_position (float): Current vertical position.
            max_height (float): Maximum page height.

        Returns:
            float: Updated vertical position.
        """
        table_width, table_height = table.wrap(0, 0)
        table_x = (letter[0] - table_width) / 2
        table.drawOn(canvas_obj, table_x, y_position - table_height)
        return y_position - table_height - 20

    def _render_image(self, canvas_obj, plot_path, y_position, max_height, size):
        """
        Render an image to the PDF.

        Args:
            canvas_obj (Canvas): PDF canvas.
            plot_path (str): Path to the image.
            y_position (float): Current vertical position.
            max_height (float): Maximum page height.

        Returns:
            float: Updated vertical position.
        """
        if y_position < self.config.image_height + self.config.margin:
            canvas_obj.showPage()
            y_position = max_height - self.config.margin

        page_width = A4[0]
        x_position = (page_width - size[0]) / 2

        canvas_obj.drawImage(
            plot_path,
            x_position,
            y_position - size[1],
            size[0],
            size[1],
        )
        return y_position - size[1] - 10


class ReportGenerator:
    """
    Facade class for generating reports with simplified interface.
    """

    def __init__(
        self,
        report_name: str = 'quantum_report.pdf',
        logger=None,
        config: Optional[ReportConfiguration] = None,
    ):
        """
        Initialize the report generator.

        Args:
            report_name (str): Name of the output PDF file.
            logger (Logger, optional): Custom logger.
            config (ReportConfiguration, optional): Report configuration.
        """
        self.logger = logger or get_logger('ReportGenerator')
        self.config = config or ReportConfiguration()
        self.report_name = report_name
        self.content_builder = ReportContentBuilder()
        self.pdf_renderer = PDFRenderer(self.config)

    def add_insight(self, title: str, insight: str):
        """Add an insight to the report."""
        self.content_builder.add_insight(title, insight)

    def add_header(self, header: str):
        """Add a header to the report."""
        self.content_builder.add_header(header)

    def add_table(self, data: List[List], col_widths=None):
        """Add a table to the report."""
        self.content_builder.add_table(data, col_widths)

    def add_metrics(self, metrics: Dict):
        """Add metrics to the report."""
        self.content_builder.add_metrics(metrics)

    def add_molecule_plot(self, molecule: MoleculeInfo):
        """Add a molecule plot to the report."""
        self.content_builder.add_molecule_plot(molecule)

    def add_convergence_plot(self, iterations: list[VQEProcess], symbols, max_points=100):
        """Add a convergence plot to the report."""
        self.content_builder.add_convergence_plot(iterations, symbols, max_points)

    def add_operator_coefficients_plot(self, qubit_op, symbols, title='Operator Coefficients'):
        """Add an operator coefficients plot to the report."""
        self.content_builder.add_operator_coeff_plot(qubit_op, symbols, title)

    def add_complex_operator_coefficients_plot(
        self, qubit_op, symbols, title='Operator Coefficients'
    ):
        """Add a operator coefficients plot in polar coordinates to the report."""
        self.content_builder.add_complex_coeff_plot(qubit_op, symbols, title)

    def new_page(self):
        """Insert a new page in the report."""
        self.content_builder.new_page()

    def generate_report(self):
        """Generate the final PDF report."""
        try:
            content = self.content_builder.get_content()
            self.pdf_renderer.render(content, str(Path(settings.GEN_DIR, self.report_name)))
            self.logger.info(f'Report successfully generated: {self.report_name}')
        except Exception as e:
            self.logger.error(f'Report generation failed: {e}')
            raise
