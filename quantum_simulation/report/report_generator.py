"""
report_generator.py

Generates PDF reports summarizing visualizations and key insights.
"""

import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle


class ReportGenerator:
    """
    A class to compile plots and metrics into a PDF report.
    """

    def __init__(self, report_name='quantum_report.pdf'):
        """
        Initializes the report generator.

        Args:
            report_name: Name of the PDF file to generate.
        """
        self.report_name = report_name
        self.content = []  # Stores paths to generated plots

    def add_plot(self, plot_path, description=None):
        """
        Adds a plot to the report.

        Args:
            plot_path: Path to the saved plot image.
            description: Optional text description of the plot.
        """
        self.content.append((plot_path, description))

    def add_text_summary(self, canvas, text, x, y):
        """
        Adds a text summary to the PDF.

        Args:
            canvas: ReportLab canvas object.
            text: Text string to add.
            x: X-coordinate.
            y: Y-coordinate.
        """
        canvas.drawString(x, y, text)

    def generate_report(self, metrics=None):
        """
        Compiles the report into a PDF.

        Args:
            metrics: Optional dictionary of metrics to include as a table.
        """
        c = canvas.Canvas(self.report_name, pagesize=letter)
        width, height = letter
        margin = 50

        # Title
        c.setFont('Helvetica-Bold', 16)
        c.drawString(margin, height - margin, 'Quantum Experiment Report')

        # Add metrics as a table if provided
        if metrics:
            self._add_metrics_table(c, metrics, margin, height - 100)

        # Add plots
        y_position = height - 250
        for plot_path, description in self.content:
            if y_position < 150:  # Add new page if space runs out
                c.showPage()
                y_position = height - margin

            # Add plot image
            c.drawImage(
                plot_path, margin, y_position - 150, width=400, height=150
            )
            y_position -= 180

            # Add description
            if description:
                c.setFont('Helvetica', 10)
                c.drawString(margin, y_position, description)
                y_position -= 20

        c.save()
        print(f'Report generated: {self.report_name}')

    def _add_metrics_table(self, c, metrics, x, y):
        """
        Adds a metrics table to the PDF.

        Args:
            c: ReportLab canvas object.
            metrics: Dictionary of metrics to include.
            x: X-coordinate.
            y: Y-coordinate.
        """
        data = [['Metric', 'Value']] + [
            [key, str(value)] for key, value in metrics.items()
        ]
        table = Table(data)
        style = TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]
        )
        table.setStyle(style)

        table.wrapOn(c, x, y)
        table.drawOn(c, x, y - 100)

    def add_error_metrics(self, error_metrics):
        """
        Adds error metrics to the report.

        Args:
            error_metrics: Dictionary of error metrics to include.
        """
        self.content.append(('error_metrics', error_metrics))

    def add_timing_analysis(self, timing_metrics):
        """
        Adds timing analysis to the report.

        Args:
            timing_metrics: Dictionary of timing metrics to include.
        """
        self.content.append(('timing_metrics', timing_metrics))
