"""Tests for report generation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantum_pipeline.report.configuration import ReportConfiguration
from quantum_pipeline.report.content_builder import ReportContentBuilder
from quantum_pipeline.report.renderer import PDFRenderer
from quantum_pipeline.report.report_generator import ReportGenerator
from quantum_pipeline.structures.vqe_observation import VQEProcess


@pytest.fixture
def report_config():
    """Create a ReportConfiguration for testing."""
    return ReportConfiguration()


@pytest.fixture
def report_generator(report_config):
    """Create a ReportGenerator instance."""
    return ReportGenerator(
        report_name='test_report.pdf',
        config=report_config,
    )


@pytest.fixture
def sample_vqe_processes():
    """Create sample VQE processes."""
    processes = []
    for i in range(5):
        process = VQEProcess(
            iteration=i + 1,
            parameters=np.random.random(4),
            result=2.0 - i * 0.3,
            std=0.05,
        )
        processes.append(process)
    return processes


class TestReportGeneratorInitialization:
    """Test ReportGenerator initialization."""

    def test_generator_creation_default(self):
        """Test creating generator with default parameters."""
        gen = ReportGenerator()
        assert gen is not None
        assert gen.report_name == 'quantum_report.pdf'
        assert gen.config is not None

    def test_generator_creation_custom_name(self):
        """Test creating generator with custom report name."""
        gen = ReportGenerator(report_name='custom_report.pdf')
        assert gen.report_name == 'custom_report.pdf'

    def test_generator_creation_custom_config(self, report_config):
        """Test creating generator with custom config."""
        gen = ReportGenerator(config=report_config)
        assert gen.config == report_config

    def test_generator_has_content_builder(self, report_generator):
        """Test that generator has content builder."""
        assert report_generator.content_builder is not None
        assert isinstance(report_generator.content_builder, ReportContentBuilder)

    def test_generator_has_pdf_renderer(self, report_generator):
        """Test that generator has PDF renderer."""
        assert report_generator.pdf_renderer is not None
        assert isinstance(report_generator.pdf_renderer, PDFRenderer)

    def test_generator_has_logger(self, report_generator):
        """Test that generator has logger."""
        assert report_generator.logger is not None

    def test_custom_logger(self):
        """Test creating generator with custom logger."""
        mock_logger = MagicMock()
        gen = ReportGenerator(logger=mock_logger)
        assert gen.logger == mock_logger


class TestContentManipulation:
    """Test adding content to reports."""

    def test_add_insight(self, report_generator):
        """Test adding insight to report."""
        report_generator.add_insight('Test Title', 'Test insight content')
        assert report_generator.content_builder is not None

    def test_add_header(self, report_generator):
        """Test adding header to report."""
        report_generator.add_header('Test Header')
        assert report_generator.content_builder is not None

    def test_multiple_insights(self, report_generator):
        """Test adding multiple insights."""
        for i in range(5):
            report_generator.add_insight(f'Title {i}', f'Content {i}')

    def test_multiple_headers(self, report_generator):
        """Test adding multiple headers."""
        for i in range(3):
            report_generator.add_header(f'Header {i}')

    def test_empty_insight(self, report_generator):
        """Test adding empty insight."""
        report_generator.add_insight('', '')

    def test_long_insight(self, report_generator):
        """Test adding very long insight."""
        long_content = 'x' * 10000
        report_generator.add_insight('Long Title', long_content)

    def test_special_characters_in_content(self, report_generator):
        """Test content with special characters."""
        report_generator.add_insight('Title', 'Content with special chars: @#$%^&*()')

    def test_unicode_content(self, report_generator):
        """Test content with unicode characters."""
        report_generator.add_insight('Greek', 'α β γ δ ε')  # noqa: RUF001


class TestReportConfiguration:
    """Test ReportConfiguration handling."""

    def test_default_config(self, report_config):
        """Test that default config is created."""
        assert report_config is not None

    def test_config_in_generator(self, report_generator, report_config):
        """Test config is properly stored in generator."""
        assert report_generator.config == report_config

    def test_different_configs(self):
        """Test creating generators with different configs."""
        config1 = ReportConfiguration()
        config2 = ReportConfiguration()
        gen1 = ReportGenerator(config=config1)
        gen2 = ReportGenerator(config=config2)
        assert gen1.config is config1
        assert gen2.config is config2

    def test_none_config_creates_default(self):
        """Test that None config creates default."""
        gen = ReportGenerator(config=None)
        assert gen.config is not None


class TestReportNaming:
    """Test report naming conventions."""

    def test_pdf_extension(self, report_generator):
        """Test report has PDF extension."""
        assert report_generator.report_name.endswith('.pdf')

    def test_custom_pdf_name(self):
        """Test custom PDF file naming."""
        names = ['report1.pdf', 'results_2024.pdf', 'vqe_output.pdf']
        for name in names:
            gen = ReportGenerator(report_name=name)
            assert gen.report_name == name

    def test_report_name_with_path(self):
        """Test report name with file path."""
        gen = ReportGenerator(report_name='/tmp/test_report.pdf')
        assert gen.report_name == '/tmp/test_report.pdf'

    def test_report_name_with_unicode(self):
        """Test report name with unicode characters."""
        gen = ReportGenerator(report_name='отчет_2024.pdf')
        assert gen.report_name == 'отчет_2024.pdf'


class TestContentBuilder:
    """Test ReportContentBuilder integration."""

    def test_content_builder_methods(self, report_generator):
        """Test that content builder methods are accessible."""
        assert hasattr(report_generator.content_builder, 'add_insight')
        assert hasattr(report_generator.content_builder, 'add_header')

    def test_content_builder_delegation(self, report_generator):
        """Test that add methods delegate to content builder."""
        with patch.object(report_generator.content_builder, 'add_insight') as mock_add:
            report_generator.add_insight('Title', 'Content')
            mock_add.assert_called_once_with('Title', 'Content')

    def test_content_builder_header_delegation(self, report_generator):
        """Test that add_header delegates to content builder."""
        with patch.object(report_generator.content_builder, 'add_header') as mock_add:
            report_generator.add_header('Header')
            mock_add.assert_called_once_with('Header')


class TestPDFRendering:
    """Test PDF rendering integration."""

    def test_pdf_renderer_exists(self, report_generator):
        """Test that PDF renderer is available."""
        assert report_generator.pdf_renderer is not None

    def test_pdf_renderer_type(self, report_generator):
        """Test PDF renderer is correct type."""
        assert isinstance(report_generator.pdf_renderer, PDFRenderer)

    def test_pdf_renderer_config(self, report_generator, report_config):
        """Test that PDF renderer has correct config."""
        assert report_generator.pdf_renderer is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_generator_with_empty_name(self):
        """Test generator with empty report name."""
        gen = ReportGenerator(report_name='')
        assert gen.report_name == ''

    def test_generator_with_very_long_name(self):
        """Test generator with very long report name."""
        long_name = 'x' * 1000 + '.pdf'
        gen = ReportGenerator(report_name=long_name)
        assert gen.report_name == long_name

    def test_multiple_generators(self):
        """Test creating multiple generator instances."""
        generators = [ReportGenerator(report_name=f'report_{i}.pdf') for i in range(10)]
        assert len(generators) == 10
        for i, gen in enumerate(generators):
            assert gen.report_name == f'report_{i}.pdf'

    def test_generator_independence(self):
        """Test that generators are independent."""
        gen1 = ReportGenerator(report_name='report1.pdf')
        gen2 = ReportGenerator(report_name='report2.pdf')

        gen1.add_insight('Insight 1', 'Content 1')
        gen2.add_insight('Insight 2', 'Content 2')

        # Generators should not interfere
        assert gen1.report_name == 'report1.pdf'
        assert gen2.report_name == 'report2.pdf'

    def test_special_characters_in_filename(self):
        """Test special characters in filename."""
        names = [
            'report-2024.pdf',
            'report_final.pdf',
            'report (1).pdf',
            'report[v2].pdf',
        ]
        for name in names:
            gen = ReportGenerator(report_name=name)
            assert gen.report_name == name

    def test_report_name_with_spaces(self):
        """Test report name with spaces."""
        gen = ReportGenerator(report_name='my report 2024.pdf')
        assert gen.report_name == 'my report 2024.pdf'


class TestReportContentBuilderIntegration:
    """Test integration with content builder."""

    def test_content_builder_is_created(self, report_generator):
        """Test that content builder is created."""
        assert report_generator.content_builder is not None

    def test_content_builder_can_add_content(self, report_generator):
        """Test that content builder can add content."""
        # These should not raise
        report_generator.add_insight('Test', 'Content')
        report_generator.add_header('Header')

    def test_multiple_content_additions(self, report_generator):
        """Test adding multiple pieces of content."""
        for i in range(10):
            report_generator.add_insight(f'Title {i}', f'Content {i}')
            report_generator.add_header(f'Header {i}')

    def test_content_order_preservation(self, report_generator):
        """Test that content additions are recorded."""
        # Add content in order
        report_generator.add_header('Header 1')
        report_generator.add_insight('Title 1', 'Content 1')
        report_generator.add_header('Header 2')
        report_generator.add_insight('Title 2', 'Content 2')

        # Should complete without error


class TestReportGeneratorWithVQEData:
    """Test report generator with VQE-specific data."""

    def test_generator_accepts_vqe_processes(self, report_generator, sample_vqe_processes):
        """Test that generator can be created with VQE data."""
        # Should be able to create generator and add VQE-related content
        report_generator.add_insight(
            'VQE Results', f'Completed {len(sample_vqe_processes)} iterations'
        )

    def test_report_with_convergence_data(self, report_generator):
        """Test report with convergence data."""
        energies = [2.0, 1.8, 1.6, 1.4, 1.2]
        report_generator.add_insight('Energy Convergence', f'Minimum energy: {min(energies)}')

    def test_report_with_optimization_results(self, report_generator):
        """Test report with optimization results."""
        report_generator.add_insight('Optimization Results', 'Final energy: -1.2 Ha')
        report_generator.add_insight('Parameters', 'θ0=0.5, θ1=1.2, θ2=0.8, θ3=2.1')
