"""
report_generator.py

Generates PDF reports with improved modularity and flexibility.
"""

from pathlib import Path
from typing import Dict, List, Optional

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo

from quantum_pipeline.configs import settings
from quantum_pipeline.report.configuration import ReportConfiguration
from quantum_pipeline.report.content_builder import ReportContentBuilder
from quantum_pipeline.report.renderer import PDFRenderer
from quantum_pipeline.structures.vqe_observation import VQEProcess
from quantum_pipeline.utils.logger import get_logger


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
        self.logger = logger or get_logger(self.__class__.__name__)
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
