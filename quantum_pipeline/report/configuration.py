from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

from quantum_pipeline.utils.logger import get_logger


class ReportConfiguration:
    """Configuration parameters for report generation."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

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

        self.logger.debug('ReportConfiguration initialized with: %s', self.to_dict())

    def to_dict(self):
        """Converts the configuration to a dictionary format."""
        return {
            'margin': self.margin,
            'page_size': self.page_size,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'table_styles': self.table_styles,
            'molecule_size': self.molecule_size,
            'op_coeff_size': self.op_coeff_size,
            'complex_op_coeff_size': self.complex_op_coeff_size,
            'convergence_size': self.convergence_size,
        }
