"""Integration tests for the quantum-pipeline processing flow.

Tests the wiring between config parsing, molecule loading, mapping,
and solving stages.  External services (IBM Quantum, Kafka, PySCF driver)
are mocked, but the real module interactions are exercised.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.parsing.argparser import QuantumPipelineArgParser
from quantum_pipeline.configs.parsing.configuration_manager import ConfigurationManager
from quantum_pipeline.drivers.basis_sets import validate_basis_set
from quantum_pipeline.drivers.molecule_loader import load_molecule, validate_molecule_data
from quantum_pipeline.mappers.jordan_winger_mapper import JordanWignerMapper
from quantum_pipeline.solvers.vqe_solver import VQESolver
from quantum_pipeline.structures.vqe_observation import (
    VQEDecoratedResult,
    VQEInitialData,
    VQEProcess,
    VQEResult,
)
from quantum_pipeline.utils.timer import Timer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_MOLECULES_JSON = [
    {
        'symbols': ['H', 'H'],
        'coords': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]],
        'multiplicity': 1,
        'charge': 0,
    }
]


@pytest.fixture
def molecule_file(tmp_path):
    """Write a temporary molecule JSON file and return its path."""
    fp = tmp_path / 'molecules.json'
    fp.write_text(json.dumps(SAMPLE_MOLECULES_JSON))
    return str(fp)


@pytest.fixture
def backend_config():
    return BackendConfig(
        local=True,
        gpu=False,
        optimization_level=2,
        min_num_qubits=4,
        filters=None,
        simulation_method='statevector',
        gpu_opts=None,
        noise=None,
    )


@pytest.fixture
def sample_hamiltonian():
    """A small 2-qubit Hamiltonian for integration tests."""
    return SparsePauliOp.from_list([('II', 1.0), ('IZ', 0.5), ('ZI', -0.3), ('ZZ', 0.2)])


# ---------------------------------------------------------------------------
# Stage 1: Config parsing → dict
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConfigParsingIntegration:
    """Config parsing produces a dict consumed by downstream stages."""

    def test_parsed_config_contains_required_keys(self, molecule_file):
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        test_args = ['--file', molecule_file, '--max-iterations', '10']
        with patch('sys.argv', ['quantum_pipeline.py', *test_args]):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        assert 'max_iterations' in config
        assert 'optimizer' in config
        assert 'backend_config' in config
        assert isinstance(config['backend_config'], BackendConfig)

    def test_config_feeds_into_solver_creation(self, molecule_file, sample_hamiltonian):
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()

        test_args = [
            '--file',
            molecule_file,
            '--max-iterations',
            '5',
            '--optimizer',
            'COBYLA',
        ]
        with patch('sys.argv', ['quantum_pipeline.py', *test_args]):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=config['backend_config'],
            max_iterations=config['max_iterations'],
            optimizer=config['optimizer'],
            ansatz_reps=2,
        )
        assert solver.max_iterations == 5
        assert solver.optimizer == 'COBYLA'


# ---------------------------------------------------------------------------
# Stage 2: Molecule loading + basis set validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMoleculeLoadingIntegration:
    """Molecule JSON → MoleculeInfo objects with basis-set validation."""

    def test_load_molecule_returns_molecule_info(self, molecule_file):
        molecules = load_molecule(molecule_file)
        assert len(molecules) == 1
        mol = molecules[0]
        assert list(mol.symbols) == ['H', 'H']

    def test_validate_molecule_data_accepts_valid(self):
        validate_molecule_data(SAMPLE_MOLECULES_JSON)

    def test_validate_molecule_data_rejects_missing_fields(self):
        with pytest.raises(ValueError):
            validate_molecule_data([{'symbols': ['H']}])

    def test_basis_set_validation_accepts_known(self):
        validate_basis_set('sto3g')

    def test_basis_set_validation_rejects_unknown(self):
        with pytest.raises(ValueError):
            validate_basis_set('fantasy-basis')

    def test_load_then_validate_basis_flow(self, molecule_file):
        """Full load → validate flow that VQERunner.load_molecules performs."""
        molecules = load_molecule(molecule_file)
        validate_basis_set('sto3g')
        assert len(molecules) >= 1


# ---------------------------------------------------------------------------
# Stage 3: Mapping (Jordan-Wigner)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMappingIntegration:
    """JordanWignerMapper wraps qiskit-nature mapper correctly."""

    def test_mapper_rejects_none(self):
        mapper = JordanWignerMapper()
        with pytest.raises(ValueError):
            mapper.map(None)

    def test_mapper_produces_qubit_op(self):
        """Use a mock fermionic op to verify the mapper delegates to qiskit."""
        mock_fermionic = MagicMock()
        mock_qubit_op = SparsePauliOp.from_list([('IZ', 0.5)])

        with patch(
            'quantum_pipeline.mappers.jordan_winger_mapper.JordanWignerMapperQiskit'
        ) as mock_jwm:
            mock_jwm.return_value.map.return_value = mock_qubit_op
            mapper = JordanWignerMapper()
            result = mapper.map(mock_fermionic)

        assert isinstance(result, SparsePauliOp)


# ---------------------------------------------------------------------------
# Stage 4: Solver creation and wiring
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSolverWiringIntegration:
    """VQESolver correctly wires config values and exposes them."""

    @pytest.mark.parametrize(
        'optimizer',
        ['COBYLA', 'L-BFGS-B', 'SLSQP'],
    )
    def test_solver_accepts_various_optimizers(
        self, sample_hamiltonian, backend_config, optimizer
    ):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=backend_config,
            max_iterations=10,
            optimizer=optimizer,
            ansatz_reps=2,
        )
        assert solver.optimizer == optimizer

    def test_solver_convergence_threshold_passthrough(self, sample_hamiltonian, backend_config):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=backend_config,
            max_iterations=20,
            convergence_threshold=1e-8,
            ansatz_reps=2,
        )
        assert solver.convergence_threshold == 1e-8
        assert solver.max_iterations == 20


# ---------------------------------------------------------------------------
# Stage 5: Result assembly (VQE result → DecoratedResult)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestResultAssemblyIntegration:
    """Assembling VQEResult into VQEDecoratedResult mirrors VQERunner.run logic."""

    def test_decorated_result_assembled_from_components(self):
        init_data = VQEInitialData(
            backend='aer_simulator',
            num_qubits=2,
            hamiltonian=np.array([[1, 0], [0, -1]]),
            num_parameters=4,
            initial_parameters=np.zeros(4),
            noise_backend='undef',
            optimizer='COBYLA',
            ansatz=MagicMock(),
            ansatz_reps=2,
            default_shots=1024,
        )
        process = VQEProcess(
            iteration=1,
            parameters=np.zeros(4),
            result=np.float64(-1.2),
            std=np.float64(0.01),
        )
        vqe_result = VQEResult(
            initial_data=init_data,
            iteration_list=[process],
            minimum=np.float64(-1.2),
            optimal_parameters=np.zeros(4),
            maxcv=None,
            minimization_time=np.float64(0.5),
        )

        molecule = MagicMock()
        molecule.symbols = ['H', 'H']

        decorated = VQEDecoratedResult(
            vqe_result=vqe_result,
            molecule=molecule,
            basis_set='sto3g',
            hamiltonian_time=np.float64(0.1),
            mapping_time=np.float64(0.05),
            vqe_time=np.float64(0.5),
            total_time=np.float64(0.65),
            molecule_id=0,
        )

        assert decorated.get_result_suffix() == '_it1'
        assert '_mol0' in decorated.get_schema_suffix()
        assert '_HH_' in decorated.get_schema_suffix()


# ---------------------------------------------------------------------------
# Stage 6: Timer used across pipeline stages
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTimerPipelineIntegration:
    """Timer context manager wires correctly into multi-stage timing."""

    def test_sequential_timers_accumulate(self):
        """Mirrors VQERunner.runVQE using three sequential timers."""
        with Timer() as t_hamiltonian:
            _ = 1 + 1  # placeholder work
        with Timer() as t_mapping:
            _ = 2 + 2
        with Timer() as t_vqe:
            _ = 3 + 3

        total = np.float64(t_hamiltonian.elapsed + t_mapping.elapsed + t_vqe.elapsed)
        assert total >= 0
        assert isinstance(total, np.float64)

    def test_timer_survives_exception_in_stage(self):
        """Ensure timing data is available even if a stage fails."""
        timer = Timer()
        with pytest.raises(ZeroDivisionError), timer:
            _ = 1 / 0
        assert timer.elapsed >= 0


# ---------------------------------------------------------------------------
# End-to-end: config → load → validate → solver creation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEndToEndWiring:
    """Config parsing through to solver construction (no quantum execution)."""

    def test_full_config_to_solver_pipeline(self, molecule_file, sample_hamiltonian):
        """Parse config → load molecules → validate basis → create solver."""
        # 1. Parse config
        parser = QuantumPipelineArgParser()
        config_manager = ConfigurationManager()
        test_args = [
            '--file',
            molecule_file,
            '--max-iterations',
            '3',
            '--optimizer',
            'COBYLA',
            '--basis',
            'sto3g',
        ]
        with patch('sys.argv', ['quantum_pipeline.py', *test_args]):
            args = parser.parser.parse_args(test_args)
            config = config_manager.get_config(args)

        # 2. Load molecules
        molecules = load_molecule(config['file'])
        assert len(molecules) == 1

        # 3. Validate basis set
        validate_basis_set(config['basis'])

        # 4. Create solver
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=config['backend_config'],
            max_iterations=config['max_iterations'],
            optimizer=config['optimizer'],
            ansatz_reps=2,
        )

        assert solver.max_iterations == 3
        assert solver.optimizer == 'COBYLA'
        assert solver.backend_config.local is True

    def test_invalid_basis_set_stops_pipeline(self, molecule_file):
        """Pipeline should fail at basis validation for unknown basis."""
        molecules = load_molecule(molecule_file)
        assert len(molecules) == 1

        with pytest.raises(ValueError):
            validate_basis_set('invalid-basis')

    def test_invalid_molecule_file_stops_pipeline(self, tmp_path):
        """Pipeline should fail at molecule loading for bad data."""
        bad_file = tmp_path / 'bad.json'
        bad_file.write_text(json.dumps([{'symbols': ['H']}]))  # missing coords

        with pytest.raises(ValueError):
            load_molecule(str(bad_file))

    def test_config_optimizer_choices_match_settings(self):
        """Argparser optimizer choices stay in sync with settings."""
        parser = QuantumPipelineArgParser()
        # Find the optimizer action
        optimizer_action = None
        for action in parser.parser._actions:
            if hasattr(action, 'dest') and action.dest == 'optimizer':
                optimizer_action = action
                break

        assert optimizer_action is not None
        from quantum_pipeline.configs.settings import SUPPORTED_OPTIMIZERS

        assert set(optimizer_action.choices) == set(SUPPORTED_OPTIMIZERS.keys())
