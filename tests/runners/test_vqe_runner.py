from unittest.mock import MagicMock, Mock, patch

import pytest
from kafka.errors import KafkaError

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig
from quantum_pipeline.runners.vqe_runner import VQERunner
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult


class TestVQERunner:
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for consistent testing."""
        with (
            patch('quantum_pipeline.drivers.molecule_loader.load_molecule') as mock_load,
            patch('quantum_pipeline.drivers.basis_sets.validate_basis_set') as mock_validate,
            patch('quantum_pipeline.solvers.vqe_solver.VQESolver') as mock_solver,
            patch('quantum_pipeline.report.report_generator.ReportGenerator') as mock_report,
            patch('quantum_pipeline.stream.kafka_interface.VQEKafkaProducer') as mock_kafka,
            patch(
                'qiskit_nature.second_q.drivers.pyscfd.pyscfdriver.PySCFDriver.from_molecule'
            ) as mock_driver,
            patch('quantum_pipeline.utils.timer.Timer') as mock_timer,
        ):
            # mock molecule
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            # mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # mock solver results
            mock_result = Mock()
            mock_result.minimum = -1.0
            mock_result.iteration_list = [0.5, 0.0, -0.5, -1.0]
            mock_result.initial_data = Mock()
            mock_result.initial_data.optimizer = 'COBYLA'
            mock_result.initial_data.ansatz_reps = 3
            mock_result.initial_data.hamiltonian = Mock()
            mock_result.initial_data.ansatz = Mock()
            mock_solver.return_value.solve.return_value = mock_result

            # mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            yield {
                'load_molecule': mock_load,
                'validate_basis_set': mock_validate,
                'vqe_solver': mock_solver,
                'report_generator': mock_report,
                'kafka_producer': mock_kafka,
                'mock_molecule': mock_molecule,
                'mock_result': mock_result,
                'mock_second_q_op': mock_second_q_op,
                'mock_driver': mock_driver,
                'mock_timer': mock_timer,
            }

    @pytest.fixture
    def single_molecule_file(self, tmp_path):
        """Create a temporary file with a single H2 molecule in XYZ format."""
        file_path = tmp_path / 'h2_molecule.xyz'
        with open(file_path, 'w') as f:
            f.write('[\n')
            f.write('{\n')
            f.write('"symbols": ["H", "H"],')
            f.write('"coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],')
            f.write('"multiplicity": 1,')
            f.write('"charge": 0,')
            f.write('"units": "angstrom",')
            f.write('"masses": [1.008, 1.008]')
            f.write('}\n')
            f.write(']')
        return file_path

    @pytest.fixture
    def multiple_molecules_file(self, tmp_path):
        """Create a temporary file with multiple molecules in XYZ format."""
        file_path = tmp_path / 'multiple_molecules.xyz'
        with open(file_path, 'w') as f:
            f.write('[\n')
            f.write('{\n')
            f.write('"symbols": ["H", "H"],')
            f.write('"coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],')
            f.write('"multiplicity": 1,')
            f.write('"charge": 0,')
            f.write('"units": "angstrom",')
            f.write('"masses": [1.008, 1.008]')
            f.write('},\n')
            f.write('{\n')
            f.write('"symbols": ["O", "H", "H"],')
            f.write('"coords": [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],')
            f.write('"multiplicity": 1,')
            f.write('"charge": 0,')
            f.write('"units": "angstrom",')
            f.write('"masses": [15.999, 1.008, 1.008]')
            f.write('}\n')
            f.write(']')
        return file_path

    @pytest.fixture
    def invalid_molecule_file(self, tmp_path):
        """Create a temporary file with invalid molecule data."""
        file_path = tmp_path / 'invalid_molecule.xyz'
        with open(file_path, 'w') as f:
            f.write('Invalid XYZ data\n')
        return file_path

    def test_initialization_with_minimal_params(self):
        """Test VQERunner initialization with minimal parameters."""
        runner = VQERunner(filepath='/path/to/molecules.xyz', basis_set='sto3g')
        assert runner.filepath == '/path/to/molecules.xyz'
        assert runner.basis_set == 'sto3g'
        assert runner.max_iterations == 100  # Default value
        assert runner.kafka is False
        assert runner.report is False
        assert runner.run_results == []
        assert runner.backend_config is not None

    def test_initialization_with_custom_backend(self):
        """Test VQERunner initialization with custom backend config."""
        custom_backend = BackendConfig(
            local=True,
            optimization_level=2,
            min_num_qubits=None,
            filters=None,
            gpu=None,
            simulation_method=None,
            gpu_opts=None,
            noise=None,
        )

        runner = VQERunner(
            filepath='/path/to/molecules.xyz', basis_set='sto3g', backend_config=custom_backend
        )

        assert runner.backend_config == custom_backend
        assert runner.backend_config.optimization_level == 2

    def test_initialization_with_kafka_config(self):
        """Test VQERunner initialization with Kafka config."""
        security_config = SecurityConfig.get_default()
        kafka_config = ProducerConfig(
            servers='test-server:9092',
            topic='test-topic',
            security=security_config,
            retries=5,
            timeout=30,
        )

        runner = VQERunner(
            filepath='/path/to/molecules.xyz',
            basis_set='sto3g',
            kafka=True,
            kafka_config=kafka_config,
        )

        assert runner.kafka is True
        assert runner.kafka_config == kafka_config
        assert runner.kafka_config.servers == 'test-server:9092'
        assert runner.kafka_config.topic == 'test-topic'

    def test_load_molecules(self, single_molecule_file):
        """Test molecule loading functionality with a real temporary file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set') as mock_validate,
        ):
            # mock values
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            runner = VQERunner(filepath=str(single_molecule_file), basis_set='sto3g')
            molecules = runner.load_molecules()

            # check if mocks were called correctly
            mock_load.assert_called_once_with(str(single_molecule_file))
            mock_validate.assert_called_once_with('sto3g')
            assert molecules == [mock_molecule]

    def test_load_molecules_error(self, invalid_molecule_file):
        """Test error handling during molecule loading with invalid file."""
        with patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load:
            mock_load.side_effect = ValueError('Invalid XYZ format')

            runner = VQERunner(filepath=str(invalid_molecule_file), basis_set='sto3g')

            with pytest.raises(ValueError):
                runner.load_molecules()

    def test_provide_hamiltonian(self):
        """Test Hamiltonian generation."""
        with patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver:
            mock_molecule = Mock()
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            runner = VQERunner(filepath='dummy.xyz', basis_set='sto3g')
            result = runner.provide_hamiltonian(mock_molecule)

            mock_driver.assert_called_once_with(mock_molecule, basis='sto3g')
            assert result == mock_second_q_op

    def test_run_method(self, single_molecule_file):
        """Test the full VQE run method with a real file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set'),
            patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver,
            patch('quantum_pipeline.runners.vqe_runner.JordanWignerMapper') as mock_mapper,
            patch('quantum_pipeline.runners.vqe_runner.VQESolver') as mock_solver,
            patch('quantum_pipeline.runners.vqe_runner.Timer') as mock_timer,
        ):
            # mock molecule
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            # mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # mock mapper
            mock_qubit_op = Mock()
            mock_mapper.return_value.map.return_value = mock_qubit_op

            # mock solver results
            mock_result = Mock()
            mock_result.minimum = -1.0
            mock_result.iteration_list = [0.5, 0.0, -0.5, -1.0]
            mock_result.initial_data = Mock()
            mock_result.initial_data.optimizer = 'COBYLA'
            mock_result.initial_data.ansatz_reps = 3
            mock_result.initial_data.hamiltonian = Mock()
            mock_result.initial_data.ansatz = Mock()
            mock_solver.return_value.solve.return_value = mock_result

            # mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            # run the test
            runner = VQERunner(filepath=str(single_molecule_file), basis_set='sto3g')
            runner.run()

            # see if molecules and the solver were loaded
            mock_load.assert_called_once()
            mock_solver.assert_called_once()

            # check if results were returned and if timing data was set
            assert len(runner.run_results) == 1
            assert isinstance(runner.run_results[0], VQEDecoratedResult)
            assert runner.hamiltonian_time == 0.5
            assert runner.mapping_time == 0.5
            assert runner.vqe_time == 0.5

    def test_run_with_kafka_enabled(self, single_molecule_file):
        """Test running with Kafka enabled using a real file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set'),
            patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver,
            patch('quantum_pipeline.runners.vqe_runner.JordanWignerMapper') as mock_mapper,
            patch('quantum_pipeline.runners.vqe_runner.VQESolver') as mock_solver,
            patch('quantum_pipeline.runners.vqe_runner.VQEKafkaProducer') as mock_kafka,
            patch('quantum_pipeline.runners.vqe_runner.Timer') as mock_timer,
        ):
            # mock molecules
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            # mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # mock mapper
            mock_qubit_op = Mock()
            mock_mapper.return_value.map.return_value = mock_qubit_op

            # mock solver
            mock_result = Mock()
            mock_result.minimum = -1.0
            mock_result.iteration_list = [0.5, 0.0, -0.5, -1.0]
            mock_result.initial_data = Mock()
            mock_result.initial_data.optimizer = 'COBYLA'
            mock_result.initial_data.ansatz_reps = 3
            mock_result.initial_data.hamiltonian = Mock()
            mock_result.initial_data.ansatz = Mock()
            mock_solver.return_value.solve.return_value = mock_result

            # mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            # run the test
            runner = VQERunner(
                filepath=str(single_molecule_file),
                basis_set='sto3g',
                kafka=True,
                kafka_bootstrap_servers='localhost:9092',
                kafka_topic='test_topic',
            )
            runner.run()

            mock_kafka.assert_called_once()
            mock_kafka.return_value.send_result.assert_called_once()

    def test_run_with_kafka_error(self, single_molecule_file):
        """Test handling of Kafka errors with a real file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set'),
            patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver,
            patch('quantum_pipeline.runners.vqe_runner.JordanWignerMapper') as mock_mapper,
            patch('quantum_pipeline.runners.vqe_runner.VQESolver') as mock_solver,
            patch('quantum_pipeline.runners.vqe_runner.VQEKafkaProducer') as mock_kafka,
            patch('quantum_pipeline.runners.vqe_runner.Timer') as mock_timer,
        ):
            # set up mock molecule
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            # set up mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # set up mock mapper
            mock_qubit_op = Mock()
            mock_mapper.return_value.map.return_value = mock_qubit_op

            # set up mock solver results
            mock_result = Mock()
            mock_result.minimum = -1.0
            mock_result.iteration_list = [0.5, 0.0, -0.5, -1.0]
            mock_result.initial_data = Mock()
            mock_result.initial_data.optimizer = 'COBYLA'
            mock_result.initial_data.ansatz_reps = 3
            mock_result.initial_data.hamiltonian = Mock()
            mock_result.initial_data.ansatz = Mock()
            mock_solver.return_value.solve.return_value = mock_result

            # set up mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            # simluating Kafka error
            mock_kafka.side_effect = KafkaError('Kafka connection error')

            # run the test
            runner = VQERunner(filepath=str(single_molecule_file), basis_set='sto3g', kafka=True)

            # should continue
            runner.run()

            # check if processing is continued, despite kafka failure
            assert len(runner.run_results) == 1

    def test_run_with_report_generation(self, single_molecule_file):
        """Test running with report generation enabled using a real file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set'),
            patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver,
            patch('quantum_pipeline.runners.vqe_runner.JordanWignerMapper') as mock_mapper,
            patch('quantum_pipeline.runners.vqe_runner.VQESolver') as mock_solver,
            patch('quantum_pipeline.runners.vqe_runner.ReportGenerator') as mock_report,
            patch('quantum_pipeline.runners.vqe_runner.Timer') as mock_timer,
        ):
            # mock molecule
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            # mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # mock mapper
            mock_qubit_op = Mock()
            mock_mapper.return_value.map.return_value = mock_qubit_op

            # mock solver results
            mock_result = Mock()
            mock_result.minimum = -1.0
            mock_result.iteration_list = [0.5, 0.0, -0.5, -1.0]
            mock_result.initial_data = Mock()
            mock_result.initial_data.optimizer = 'COBYLA'
            mock_result.initial_data.ansatz_reps = 3
            mock_result.initial_data.hamiltonian = Mock()
            mock_result.initial_data.ansatz = Mock()
            mock_solver.return_value.solve.return_value = mock_result

            # mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            # start the runner
            runner = VQERunner(filepath=str(single_molecule_file), basis_set='sto3g', report=True)
            runner.run()

            # check if report classes were called
            mock_report.return_value.add_header.assert_called()
            mock_report.return_value.add_molecule_plot.assert_called_once_with(mock_molecule)
            mock_report.return_value.generate_report.assert_called_once()

    def test_vqe_algorithm_error(self, single_molecule_file):
        """Test handling of VQE algorithm errors with a real file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set'),
            patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver,
            patch('quantum_pipeline.runners.vqe_runner.JordanWignerMapper') as mock_mapper,
            patch('quantum_pipeline.runners.vqe_runner.VQESolver') as mock_solver,
            patch('quantum_pipeline.runners.vqe_runner.Timer') as mock_timer,
        ):
            # mock molecule
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            # mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # mock mapper
            mock_qubit_op = Mock()
            mock_mapper.return_value.map.return_value = mock_qubit_op

            # mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            # simulate runtime error
            mock_solver.return_value.solve.side_effect = RuntimeError('Convergence error')

            # run the test
            runner = VQERunner(filepath=str(single_molecule_file), basis_set='sto3g')

            with pytest.raises(RuntimeError):
                runner.run()

    def test_multiple_molecules(self, multiple_molecules_file):
        """Test processing multiple molecules from a real file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set'),
            patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver,
            patch('quantum_pipeline.runners.vqe_runner.JordanWignerMapper') as mock_mapper,
            patch('quantum_pipeline.runners.vqe_runner.VQESolver') as mock_solver,
            patch('quantum_pipeline.runners.vqe_runner.Timer') as mock_timer,
        ):
            # two mock molecules
            molecule1 = Mock()
            molecule1.symbols = ['H', 'H']
            molecule2 = Mock()
            molecule2.symbols = ['C', 'O']
            mock_load.return_value = [molecule1, molecule2]

            # mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # mock mapper
            mock_qubit_op = Mock()
            mock_mapper.return_value.map.return_value = mock_qubit_op

            # mock solver results
            mock_result = Mock()
            mock_result.minimum = -1.0
            mock_result.iteration_list = [0.5, 0.0, -0.5, -1.0]
            mock_result.initial_data = Mock()
            mock_result.initial_data.optimizer = 'COBYLA'
            mock_result.initial_data.ansatz_reps = 3
            mock_result.initial_data.hamiltonian = Mock()
            mock_result.initial_data.ansatz = Mock()
            mock_solver.return_value.solve.return_value = mock_result

            # mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            # start the runner
            runner = VQERunner(filepath=str(multiple_molecules_file), basis_set='sto3g')
            runner.run()

            # and see if solver was called twice
            assert mock_solver.call_count == 2
            assert len(runner.run_results) == 2

            # check if molecules were processed
            mock_driver.assert_any_call(molecule1, basis='sto3g')
            mock_driver.assert_any_call(molecule2, basis='sto3g')

    def test_with_convergence_threshold(self, single_molecule_file):
        """Test with custom convergence threshold using a real file."""
        with (
            patch('quantum_pipeline.runners.vqe_runner.load_molecule') as mock_load,
            patch('quantum_pipeline.runners.vqe_runner.validate_basis_set'),
            patch('quantum_pipeline.runners.vqe_runner.PySCFDriver.from_molecule') as mock_driver,
            patch('quantum_pipeline.runners.vqe_runner.JordanWignerMapper') as mock_mapper,
            patch('quantum_pipeline.runners.vqe_runner.VQESolver') as mock_solver,
            patch('quantum_pipeline.runners.vqe_runner.Timer') as mock_timer,
        ):
            # mock molecule
            mock_molecule = Mock()
            mock_molecule.symbols = ['H', 'H']
            mock_load.return_value = [mock_molecule]

            # mock driver
            mock_problem = Mock()
            mock_second_q_op = Mock()
            mock_problem.second_q_ops.return_value = [mock_second_q_op]
            mock_driver.return_value.run.return_value = mock_problem

            # mock mapper
            mock_qubit_op = Mock()
            mock_mapper.return_value.map.return_value = mock_qubit_op

            # mock solver results
            mock_result = Mock()
            mock_result.minimum = -1.0
            mock_result.iteration_list = [0.5, 0.0, -0.5, -1.0]
            mock_result.initial_data = Mock()
            mock_result.initial_data.optimizer = 'COBYLA'
            mock_result.initial_data.ansatz_reps = 3
            mock_result.initial_data.hamiltonian = Mock()
            mock_result.initial_data.ansatz = Mock()
            mock_solver.return_value.solve.return_value = mock_result

            # mock timer
            mock_timer_context = MagicMock()
            mock_timer_context.__enter__.return_value = mock_timer_context
            mock_timer_context.elapsed = 0.5
            mock_timer.return_value = mock_timer_context

            # start the runner
            runner = VQERunner(
                filepath=str(single_molecule_file), basis_set='sto3g', convergence_threshold=1e-6
            )
            runner.run()

            # check if convergence threshold was passed correctly
            mock_solver.assert_called_once_with(
                qubit_op=mock_qubit_op,
                backend_config=runner.backend_config,
                max_iterations=100,
                optimization_level=3,
                optimizer='COBYLA',
                ansatz_reps=3,
                default_shots=1024,
                convergence_threshold=1e-6,
            )

    def test_static_methods(self):
        """Test the static methods of VQERunner."""
        with patch(
            'quantum_pipeline.runners.vqe_runner.BackendConfig.default_backend_config'
        ) as mock_default:
            # mock return value
            mock_backend = Mock()
            mock_default.return_value = mock_backend

            # call for defualt backend
            default_backend = VQERunner.default_backend()

            # see if it was called and if the return value is correct
            mock_default.assert_called_once()
            assert default_backend == mock_backend
