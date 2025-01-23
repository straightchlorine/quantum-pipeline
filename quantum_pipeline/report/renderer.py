from typing import List, Optional

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Spacer, Table

from quantum_pipeline.report.configuration import ReportConfiguration


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
        canvas_obj = Canvas(output_path, pagesize=self.config.page_size)
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
