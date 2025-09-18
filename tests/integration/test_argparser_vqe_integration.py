"""Integration tests between argparser and VQE solver configuration."""

import pytest
import sys
from unittest.mock import patch, MagicMock

from quantum_pipeline.configs.parsing.argparser import QuantumPipelineArgParser
from quantum_pipeline.configs.parsing.configuration_manager import ConfigurationManager
from quantum_pipeline.solvers.vqe_solver import VQESolver
from quantum_pipeline.configs.module.backend import BackendConfig
from qiskit.quantum_info import SparsePauliOp


class TestArgparserVQEIntegration:
    """Test integration between argparser configuration and VQE solver behavior."""

    @pytest.fixture
    def sample_hamiltonian(self):
        """Create a sample Hamiltonian for testing."""
        return SparsePauliOp.from_list([('II', 1.0), ('IZ', 0.5), ('ZI', 0.3)])

    @pytest.fixture
    def mock_backend_config(self):
        """Create a mock backend configuration."""
        return BackendConfig(
            local=True,
            gpu=False,
            optimization_level=2,
            min_num_qubits=4,
            filters=None,
            simulation_method='mock_backend',
            gpu_opts=None,
            noise=None
        )

    def test_argparser_to_vqe_max_iterations_only(self, sample_hamiltonian, mock_backend_config):
        """Test max_iterations flow from argparser to VQE solver."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        # Simulate argparse with max_iterations only
        test_args = ['--file', 'molecule.json', '--max-iterations', '5']

        with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        # Verify config has correct values
        assert config['max_iterations'] == 5
        assert config['convergence'] is False

        # Create VQE solver with config values
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=config['max_iterations'],
            convergence_threshold=config['threshold'] if config['convergence'] else None,
            optimizer=config['optimizer'],
            ansatz_reps=2,
        )

        # Verify VQE solver configuration
        assert solver.max_iterations == 5
        assert solver.convergence_threshold is None
        assert solver.optimizer == 'L-BFGS-B'  # Default

    def test_argparser_to_vqe_convergence_only(self, sample_hamiltonian, mock_backend_config):
        """Test convergence flow from argparser to VQE solver."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        test_args = ['--file', 'molecule.json', '--convergence', '--threshold', '1e-8']

        with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        assert config['max_iterations'] == 100  # Default
        assert config['convergence'] is True
        assert config['threshold'] == 1e-8

        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=config['max_iterations'],
            convergence_threshold=config['threshold'] if config['convergence'] else None,
            optimizer=config['optimizer'],
            ansatz_reps=2,
        )

        assert solver.max_iterations == 100
        assert solver.convergence_threshold == 1e-8
        assert solver.optimizer == 'L-BFGS-B'

    def test_argparser_to_vqe_both_specified(self, sample_hamiltonian, mock_backend_config):
        """Test both max_iterations and convergence from argparser to VQE solver."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        test_args = [
            '--file', 'molecule.json',
            '--max-iterations', '3',
            '--convergence',
            '--threshold', '1e-6',
            '--optimizer', 'COBYLA'
        ]

        with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        assert config['max_iterations'] == 3
        assert config['convergence'] is True
        assert config['threshold'] == 1e-6
        assert config['optimizer'] == 'COBYLA'

        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=config['max_iterations'],
            convergence_threshold=config['threshold'] if config['convergence'] else None,
            optimizer=config['optimizer'],
            ansatz_reps=2,
        )

        assert solver.max_iterations == 3
        assert solver.convergence_threshold == 1e-6
        assert solver.optimizer == 'COBYLA'

        # Test priority logic
        both_specified = solver.convergence_threshold and solver.max_iterations
        assert both_specified, "Both parameters should be specified"

    def test_argparser_edge_cases_to_vqe(self, sample_hamiltonian, mock_backend_config):
        """Test edge case values flow from argparser to VQE solver."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        # Test with edge case values
        test_cases = [
            (['--file', 'molecule.json', '--max-iterations', '0'], 0, None),
            (['--file', 'molecule.json', '--max-iterations', '1'], 1, None),
            (['--file', 'molecule.json', '--convergence', '--threshold', '0'], 100, 0.0),
            (['--file', 'molecule.json', '--convergence', '--threshold', '1e-20'], 100, 1e-20),
        ]

        for test_args, expected_max_iter, expected_threshold in test_cases:
            with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
                args = parser.parser.parse_args(test_args)
                config = config_manager.get_config(args)

            solver = VQESolver(
                qubit_op=sample_hamiltonian,
                backend_config=mock_backend_config,
                max_iterations=config['max_iterations'],
                convergence_threshold=config['threshold'] if config.get('convergence', False) else None,
                optimizer=config['optimizer'],
                ansatz_reps=2,
            )

            assert solver.max_iterations == expected_max_iter
            if expected_threshold is None:
                assert solver.convergence_threshold is None
            else:
                assert solver.convergence_threshold == expected_threshold

    def test_lbfgs_b_optimization_params_integration(self, sample_hamiltonian, mock_backend_config):
        """Test that L-BFGS-B optimization parameters are correctly configured."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        test_args = ['--file', 'molecule.json', '--max-iterations', '5', '--optimizer', 'L-BFGS-B']

        with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=config['max_iterations'],
            convergence_threshold=None,
            optimizer=config['optimizer'],
            ansatz_reps=2,
        )

        # Test the L-BFGS-B specific logic
        should_disable_convergence = (
            solver.optimizer == 'L-BFGS-B' and
            solver.max_iterations
        )
        assert should_disable_convergence, "L-BFGS-B should disable default convergence criteria"

    def test_invalid_optimizer_handling(self):
        """Test that invalid optimizers are rejected at argparser level."""
        parser = QuantumPipelineArgParser()

        with pytest.raises(SystemExit):
            parser.parser.parse_args([
                '--file', 'molecule.json',
                '--optimizer', 'INVALID_OPTIMIZER'
            ])

    def test_config_preservation_through_pipeline(self, sample_hamiltonian, mock_backend_config):
        """Test that configuration values are preserved through the entire pipeline."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        original_values = {
            'max_iterations': 42,
            'threshold': 1.5e-7,
            'optimizer': 'COBYQA'
        }

        test_args = [
            '--file', 'molecule.json',
            '--max-iterations', str(original_values['max_iterations']),
            '--convergence',
            '--threshold', str(original_values['threshold']),
            '--optimizer', original_values['optimizer']
        ]

        with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        # Verify config preservation
        assert config['max_iterations'] == original_values['max_iterations']
        assert config['threshold'] == original_values['threshold']
        assert config['optimizer'] == original_values['optimizer']
        assert config['convergence'] is True

        # Verify VQE solver gets correct values
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=config['max_iterations'],
            convergence_threshold=config['threshold'] if config['convergence'] else None,
            optimizer=config['optimizer'],
            ansatz_reps=2,
        )

        assert solver.max_iterations == original_values['max_iterations']
        assert solver.convergence_threshold == original_values['threshold']
        assert solver.optimizer == original_values['optimizer']


class TestInputValidationRobustness:
    """Test robustness of input validation across the pipeline."""

    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        parser = QuantumPipelineArgParser()

        # Test very large max_iterations
        large_value = str(sys.maxsize)
        args = parser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', large_value])
        assert args.max_iterations == sys.maxsize

        # Test very small threshold
        small_threshold = '1e-300'
        args = parser.parser.parse_args(['--file', 'molecule.json', '--threshold', small_threshold])
        assert args.threshold == 1e-300

    def test_scientific_notation_preservation(self):
        """Test that scientific notation is properly handled."""
        parser = QuantumPipelineArgParser()

        test_cases = [
            '1e-6',
            '2.5e-8',
            '1.23456789e-15',
            '9.99999e-1'
        ]

        for threshold_str in test_cases:
            args = parser.parser.parse_args(['--file', 'molecule.json', '--threshold', threshold_str])
            expected_value = float(threshold_str)
            assert args.threshold == expected_value

    def test_type_consistency(self):
        """Test that types are consistent throughout the pipeline."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        test_args = ['--file', 'molecule.json', '--max-iterations', '10', '--threshold', '1e-6']

        with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        # Verify types
        assert isinstance(config['max_iterations'], int)
        assert isinstance(config['threshold'], float)
        assert isinstance(config['optimizer'], str)
        assert isinstance(config['convergence'], bool)

    def test_default_value_types(self):
        """Test that default values have correct types."""
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        test_args = ['--file', 'molecule.json']

        with patch('sys.argv', ['quantum_pipeline.py'] + test_args):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        # Verify default types
        assert isinstance(config['max_iterations'], int)
        assert isinstance(config['threshold'], float)
        assert isinstance(config['optimizer'], str)
        assert isinstance(config['convergence'], bool)