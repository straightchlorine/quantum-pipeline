import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.units import DistanceUnit

from quantum_pipeline.stream.serialization.interfaces.vqe import (
    AvroInterfaceBase,
    MoleculeInfoInterface,
    VQEDecoratedResultInterface,
    VQEInitialDataInterface,
    VQEProcessInterface,
    VQEResultInterface,
)
from quantum_pipeline.structures.vqe_observation import (
    VQEDecoratedResult,
    VQEInitialData,
    VQEProcess,
    VQEResult,
)


class MockSchemaRegistry:
    """Mock schema registry for testing."""

    def __init__(self):
        self.schemas = {}
        self.id_cache = {}

    def get_schema(self, schema_name):
        if schema_name not in self.schemas:
            raise FileNotFoundError(f'Schema {schema_name} not found')
        return self.schemas[schema_name]

    def save_schema(self, schema_name, schema):
        self.schemas[schema_name] = schema
        self.id_cache[schema_name] = 123  # Mock schema ID


class TestAvroInterfaceBase(unittest.TestCase):
    """Test the base class for Avro serializers."""

    def setUp(self):
        self.registry = MockSchemaRegistry()

        class ConcreteAvroInterface(AvroInterfaceBase):
            @property
            def schema(self):
                return {
                    'type': 'record',
                    'name': 'TestRecord',
                    'fields': [
                        {'name': 'value', 'type': 'int'},
                    ],
                }

            def serialize(self, obj):
                return {'value': obj.value}

            def deserialize(self, data):
                mock = Mock()
                mock.value = data['value']
                return mock

        self.interface = ConcreteAvroInterface(self.registry)

    def test_convert_to_primitives_numpy_int(self):
        """Test conversion of numpy integers to Python native types."""
        # test int32
        result = self.interface._convert_to_primitives(np.int32(42))
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

        # test int64
        result = self.interface._convert_to_primitives(np.int64(42))
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_convert_to_primitives_numpy_float(self):
        """Test conversion of numpy floats to Python native types."""
        # test float32
        result = self.interface._convert_to_primitives(np.float32(3.14))
        self.assertAlmostEqual(result, 3.14, places=6)
        self.assertIsInstance(result, float)

        # test float64
        result = self.interface._convert_to_primitives(np.float64(3.14))
        self.assertAlmostEqual(result, 3.14)
        self.assertIsInstance(result, float)

    def test_convert_to_primitives_ndarray(self):
        """Test conversion of numpy array to Python list."""
        arr = np.array([1, 2, 3])
        result = self.interface._convert_to_primitives(arr)
        self.assertEqual(result, [1, 2, 3])
        self.assertIsInstance(result, list)

    def test_convert_to_primitives_nested_structures(self):
        """Test conversion of nested structures with numpy types."""
        data = {
            'array': np.array([1, 2, 3]),
            'integer': np.int64(42),
            'float': np.float32(3.14),
            'list': [np.int32(1), np.int64(2), np.float32(3.0)],
            'nested': {'array': np.array([4, 5, 6])},
        }

        result = self.interface._convert_to_primitives(data)

        self.assertEqual(result['array'], [1, 2, 3])
        self.assertEqual(result['integer'], 42)
        self.assertAlmostEqual(result['float'], 3.14, places=6)
        self.assertEqual(result['list'], [1, 2, 3.0])
        self.assertEqual(result['nested']['array'], [4, 5, 6])

    def test_convert_to_numpy(self):
        """Test conversion of Python lists to numpy arrays."""
        data = [1, 2, 3]
        result = self.interface._convert_to_numpy(data)
        self.assertTrue(np.array_equal(result, np.array([1, 2, 3])))

    def test_to_avro_bytes(self):
        """Test conversion of object to Avro binary format."""
        test_obj = Mock()
        test_obj.value = 42

        result = self.interface.to_avro_bytes(test_obj)

        # verify if bytes were returned
        self.assertIsInstance(result, bytes)

    def test_from_avro_bytes(self):
        """Test conversion from Avro binary format to object."""
        # confluent schema registry header
        test_bytes = b'\x00\x00\x00\x00{'

        # mock result from the deserialize
        mock_result = Mock()
        mock_result.value = 42

        # patch the deserialize method
        with patch.object(self.interface, 'deserialize', return_value=mock_result):
            # patch the schema perse method to avoid the schema validation
            with patch('avro.schema.parse'):
                # patch the datum reader to avoid internal processing
                with patch('avro.io.DatumReader.read', return_value={'value': 42}):
                    # check the result
                    result = self.interface.from_avro_bytes(test_bytes)
                    self.assertEqual(result.value, 42)

                    # check if correct argument was passed
                    self.interface.deserialize.assert_called_once_with({'value': 42})

    def test_from_avro_bytes_invalid_magic(self):
        """Test error handling with invalid magic byte."""
        test_bytes = b'\x01\x00\x00\x00{'

        with self.assertRaises(ValueError):
            self.interface.from_avro_bytes(test_bytes)


class TestVQEProcessInterface(unittest.TestCase):
    """Test the VQE Process interface."""

    def setUp(self):
        self.registry = MockSchemaRegistry()
        self.interface = VQEProcessInterface(self.registry)

        # Create test fixture
        self.vqe_process = VQEProcess(
            iteration=5,
            parameters=np.array([0.1, 0.2, 0.3]),
            result=np.float64(-74.5),
            std=np.float64(0.01),
        )

    def test_schema(self):
        """Test schema generation."""
        schema = self.interface.schema

        self.assertIsInstance(schema, dict)
        self.assertEqual(schema['name'], 'VQEProcess')
        self.assertEqual(len(schema['fields']), 4)

    def test_serialize(self):
        """Test serialization of VQEProcess."""
        serialized = self.interface.serialize(self.vqe_process)

        self.assertEqual(serialized['iteration'], 5)
        self.assertEqual(serialized['parameters'], [0.1, 0.2, 0.3])
        self.assertEqual(serialized['result'], -74.5)
        self.assertEqual(serialized['std'], 0.01)

        self.assertIsInstance(serialized['parameters'], list)
        self.assertIsInstance(serialized['result'], float)
        self.assertIsInstance(serialized['std'], float)

    def test_deserialize(self):
        """Test deserialization of VQEProcess."""
        data = {'iteration': 5, 'parameters': [0.1, 0.2, 0.3], 'result': -74.5, 'std': 0.01}

        deserialized = self.interface.deserialize(data)

        self.assertEqual(deserialized.iteration, 5)
        self.assertTrue(np.array_equal(deserialized.parameters, np.array([0.1, 0.2, 0.3])))
        self.assertEqual(deserialized.result, np.float64(-74.5))
        self.assertEqual(deserialized.std, np.float64(0.01))

    def test_roundtrip(self):
        """Test round-trip serialization and deserialization."""
        serialized = self.interface.serialize(self.vqe_process)
        deserialized = self.interface.deserialize(serialized)

        self.assertEqual(deserialized.iteration, self.vqe_process.iteration)
        self.assertTrue(np.array_equal(deserialized.parameters, self.vqe_process.parameters))
        self.assertEqual(deserialized.result, self.vqe_process.result)
        self.assertEqual(deserialized.std, self.vqe_process.std)


class TestVQEInitialDataInterface(unittest.TestCase):
    """Test the VQE Initial Data interface."""

    def setUp(self):
        self.registry = MockSchemaRegistry()
        self.interface = VQEInitialDataInterface(self.registry)

        # Create test fixture with mock circuit
        mock_circuit = MagicMock(spec=QuantumCircuit)
        mock_circuit.__str__.return_value = 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];'

        self.vqe_initial_data = VQEInitialData(
            backend='fake_backend',
            num_qubits=2,
            hamiltonian=np.array([('ZZ', complex(1.0, 0.0)), ('X', complex(0.5, 0.0))]),
            num_parameters=3,
            initial_parameters=np.array([0.1, 0.2, 0.3]),
            optimizer='SPSA',
            ansatz=mock_circuit,
            ansatz_reps=1,
            noise_backend='fake_noise_backend',
            default_shots=1024,
        )

    def test_schema(self):
        """Test schema generation."""
        schema = self.interface.schema

        self.assertIsInstance(schema, dict)
        self.assertEqual(schema['name'], 'VQEInitialData')
        self.assertEqual(len(schema['fields']), 10)

    def test_serialize_hamiltonian(self):
        """Test serialization of Hamiltonian terms."""
        hamiltonian = np.array([('ZZ', complex(1.0, 0.0)), ('X', complex(0.5, 0.5))])
        serialized = self.interface._serialize_hamiltonian(hamiltonian)

        self.assertEqual(len(serialized), 2)
        self.assertEqual(serialized[0]['label'], 'ZZ')
        self.assertEqual(serialized[0]['coefficients']['real'], 1.0)
        self.assertEqual(serialized[0]['coefficients']['imaginary'], 0.0)
        self.assertEqual(serialized[1]['label'], 'X')
        self.assertEqual(serialized[1]['coefficients']['real'], 0.5)
        self.assertEqual(serialized[1]['coefficients']['imaginary'], 0.5)

    def test_deserialize_hamiltonian(self):
        """Test deserialization of Hamiltonian terms."""
        data = [
            {'label': 'ZZ', 'coefficients': {'real': 1.0, 'imaginary': 0.0}},
            {'label': 'X', 'coefficients': {'real': 0.5, 'imaginary': 0.5}},
        ]

        deserialized = self.interface._deserialize_hamiltonian(data)

        self.assertEqual(deserialized[0][0], 'ZZ')
        self.assertEqual(deserialized[0][1], complex(1.0, 0.0))
        self.assertEqual(deserialized[1][0], 'X')
        self.assertEqual(deserialized[1][1], complex(0.5, 0.5))

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.dumps')
    def test_serialize(self, mock_dumps):
        """Test serialization of VQEInitialData."""
        mock_dumps.return_value = 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];'

        serialized = self.interface.serialize(self.vqe_initial_data)

        self.assertEqual(serialized['backend'], 'fake_backend')
        self.assertEqual(serialized['num_qubits'], 2)
        self.assertEqual(len(serialized['hamiltonian']), 2)
        self.assertEqual(serialized['num_parameters'], 3)
        self.assertEqual(serialized['initial_parameters'], [0.1, 0.2, 0.3])
        self.assertEqual(serialized['optimizer'], 'SPSA')
        self.assertEqual(serialized['ansatz_reps'], 1)
        self.assertEqual(serialized['noise_backend'], 'fake_noise_backend')
        self.assertEqual(serialized['default_shots'], 1024)

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_deserialize(self, mock_loads):
        """Test deserialization of VQEInitialData."""
        mock_circuit = MagicMock(spec=QuantumCircuit)
        mock_loads.return_value = mock_circuit

        data = {
            'backend': 'fake_backend',
            'num_qubits': 2,
            'hamiltonian': [
                {'label': 'ZZ', 'coefficients': {'real': 1.0, 'imaginary': 0.0}},
                {'label': 'X', 'coefficients': {'real': 0.5, 'imaginary': 0.0}},
            ],
            'num_parameters': 3,
            'initial_parameters': [0.1, 0.2, 0.3],
            'optimizer': 'SPSA',
            'ansatz': 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];',
            'ansatz_reps': 1,
            'noise_backend': 'fake_noise_backend',
            'default_shots': 1024,
        }

        deserialized = self.interface.deserialize(data)

        self.assertEqual(deserialized.backend, 'fake_backend')
        self.assertEqual(deserialized.num_qubits, 2)
        self.assertEqual(len(deserialized.hamiltonian), 2)
        self.assertEqual(deserialized.num_parameters, 3)
        self.assertTrue(np.array_equal(deserialized.initial_parameters, np.array([0.1, 0.2, 0.3])))
        self.assertEqual(deserialized.optimizer, 'SPSA')
        self.assertEqual(deserialized.ansatz, mock_circuit)
        self.assertEqual(deserialized.ansatz_reps, 1)
        self.assertEqual(deserialized.noise_backend, 'fake_noise_backend')
        self.assertEqual(deserialized.default_shots, 1024)


class TestMoleculeInfoInterface(unittest.TestCase):
    """Test the MoleculeInfo interface."""

    def setUp(self):
        self.registry = MockSchemaRegistry()
        self.interface = MoleculeInfoInterface(self.registry)

        self.molecule = MoleculeInfo(
            symbols=['H', 'H'],
            coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)],
            multiplicity=1,
            charge=0,
            units=DistanceUnit.ANGSTROM,
            masses=[1.0, 1.0],
        )

    def test_schema(self):
        """Test schema generation."""
        schema = self.interface.schema

        self.assertIsInstance(schema, dict)
        self.assertEqual(schema['name'], 'MoleculeInfo')
        self.assertEqual(len(schema['fields']), 1)

    def test_serialize(self):
        """Test serialization of MoleculeInfo."""
        serialized = self.interface.serialize(self.molecule)

        self.assertEqual(serialized['molecule_data']['symbols'], ['H', 'H'])
        self.assertEqual(
            serialized['molecule_data']['coords'], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)]
        )
        self.assertEqual(serialized['molecule_data']['multiplicity'], 1)
        self.assertEqual(serialized['molecule_data']['charge'], 0)
        self.assertEqual(serialized['molecule_data']['units'], 'angstrom')
        self.assertEqual(serialized['molecule_data']['masses'], [1.0, 1.0])

    def test_deserialize(self):
        """Test deserialization of MoleculeInfo."""
        data = {
            'molecule_data': {
                'symbols': ['H', 'H'],
                'coords': [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)],
                'multiplicity': 1,
                'charge': 0,
                'units': 'angstrom',
                'masses': [1.0, 1.0],
            }
        }

        deserialized = self.interface.deserialize(data)

        self.assertEqual(deserialized.symbols, ['H', 'H'])
        self.assertEqual(deserialized.coords, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        self.assertEqual(deserialized.multiplicity, 1)
        self.assertEqual(deserialized.charge, 0)
        self.assertEqual(deserialized.units, DistanceUnit.ANGSTROM)
        self.assertTrue(np.array_equal(deserialized.masses, [1.0, 1.0]))

    def test_deserialize_flattened_coords(self):
        """Test deserialization of MoleculeInfo with flattened coordinates."""
        data = {
            'molecule_data': {
                'symbols': ['H', 'H'],
                'coords': [0.0, 0.0, 0.0, 0.0, 0.0, 0.74],  # flattened cords
                'multiplicity': 1,
                'charge': 0,
                'units': 'angstrom',
                'masses': None,
            }
        }

        deserialized = self.interface.deserialize(data)

        self.assertEqual(deserialized.symbols, ['H', 'H'])
        self.assertEqual(len(deserialized.coords), 2)  # expectes reshaped coords
        self.assertEqual(deserialized.multiplicity, 1)
        self.assertEqual(deserialized.charge, 0)
        self.assertEqual(deserialized.units, DistanceUnit.ANGSTROM)
        self.assertIsNone(deserialized.masses)


class TestVQEResultInterface(unittest.TestCase):
    """Test the VQE Result interface."""

    def setUp(self):
        self.registry = MockSchemaRegistry()
        self.interface = VQEResultInterface(self.registry)

        mock_circuit = MagicMock(spec=QuantumCircuit)

        initial_data = VQEInitialData(
            backend='fake_backend',
            num_qubits=2,
            hamiltonian=np.array([('ZZ', complex(1.0, 0.0)), ('X', complex(0.5, 0.0))]),
            num_parameters=3,
            initial_parameters=np.array([0.1, 0.2, 0.3]),
            optimizer='SPSA',
            ansatz=mock_circuit,
            ansatz_reps=1,
            noise_backend='fake_noise_backend',
            default_shots=1024,
        )

        process1 = VQEProcess(
            iteration=0,
            parameters=np.array([0.1, 0.2, 0.3]),
            result=np.float64(-74.0),
            std=np.float64(0.02),
        )

        process2 = VQEProcess(
            iteration=1,
            parameters=np.array([0.15, 0.25, 0.35]),
            result=np.float64(-74.5),
            std=np.float64(0.01),
        )

        self.vqe_result = VQEResult(
            initial_data=initial_data,
            iteration_list=[process1, process2],
            minimum=np.float64(-74.5),
            optimal_parameters=np.array([0.15, 0.25, 0.35]),
            maxcv=np.float64(0.001),
            minimization_time=np.float64(10.5),
        )

        self.patch_initial = patch.object(self.interface.initial_data_interface, 'serialize')
        self.patch_process = patch.object(self.interface.process_interface, 'serialize')
        self.mock_initial = self.patch_initial.start()
        self.mock_process = self.patch_process.start()

        self.mock_initial.return_value = {'mock': 'initial_data'}
        self.mock_process.return_value = {'mock': 'process'}

    def tearDown(self):
        self.patch_initial.stop()
        self.patch_process.stop()

    def test_schema(self):
        """Test schema generation."""
        schema = self.interface.schema

        self.assertIsInstance(schema, dict)
        self.assertEqual(schema['name'], 'VQEResult')
        self.assertEqual(len(schema['fields']), 6)

    def test_serialize(self):
        """Test serialization of VQEResult."""
        serialized = self.interface.serialize(self.vqe_result)

        self.assertEqual(serialized['initial_data'], {'mock': 'initial_data'})
        self.assertEqual(serialized['iteration_list'], [{'mock': 'process'}, {'mock': 'process'}])
        self.assertEqual(serialized['minimum'], -74.5)
        self.assertEqual(serialized['optimal_parameters'], [0.15, 0.25, 0.35])
        self.assertEqual(serialized['maxcv'], 0.001)
        self.assertEqual(serialized['minimization_time'], 10.5)

        self.mock_initial.assert_called_once_with(self.vqe_result.initial_data)
        self.assertEqual(self.mock_process.call_count, 2)

    def test_deserialize(self):
        """Test deserialization of VQEResult."""
        with patch.object(
            self.interface.initial_data_interface, 'deserialize'
        ) as mock_initial_deserialize:
            with patch.object(
                self.interface.process_interface, 'deserialize'
            ) as mock_process_deserialize:
                mock_initial_deserialize.return_value = self.vqe_result.initial_data
                mock_process_deserialize.return_value = self.vqe_result.iteration_list[0]

                data = {
                    'initial_data': {'mock': 'initial_data'},
                    'iteration_list': [{'mock': 'process'}, {'mock': 'process'}],
                    'minimum': -74.5,
                    'optimal_parameters': [0.15, 0.25, 0.35],
                    'maxcv': 0.001,
                    'minimization_time': 10.5,
                }

                deserialized = self.interface.deserialize(data)

                self.assertEqual(deserialized.initial_data, self.vqe_result.initial_data)
                self.assertEqual(len(deserialized.iteration_list), 2)
                self.assertEqual(deserialized.minimum, np.float64(-74.5))
                self.assertTrue(
                    np.array_equal(deserialized.optimal_parameters, np.array([0.15, 0.25, 0.35]))
                )
                self.assertEqual(deserialized.maxcv, np.float64(0.001))
                self.assertEqual(deserialized.minimization_time, np.float64(10.5))

                mock_initial_deserialize.assert_called_once_with({'mock': 'initial_data'})
                self.assertEqual(mock_process_deserialize.call_count, 2)


class TestVQEDecoratedResultInterface(unittest.TestCase):
    """Test the VQE Decorated Result interface."""

    def setUp(self):
        self.registry = MockSchemaRegistry()
        self.interface = VQEDecoratedResultInterface(self.registry)

        mock_vqe_result = MagicMock(spec=VQEResult)
        mock_molecule = MagicMock(spec=MoleculeInfo)
        mock_molecule.symbols = ['H', 'H']

        self.vqe_decorated_result = VQEDecoratedResult(
            vqe_result=mock_vqe_result,
            molecule=mock_molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(1.5),
            mapping_time=np.float64(0.5),
            vqe_time=np.float64(10.5),
            total_time=np.float64(12.5),
            molecule_id=1,
        )

        self.patch_result = patch.object(self.interface.result_interface, 'serialize')
        self.patch_molecule = patch.object(self.interface.molecule_interface, 'serialize')
        self.mock_result = self.patch_result.start()
        self.mock_molecule = self.patch_molecule.start()

        self.mock_result.return_value = {'mock': 'vqe_result'}
        self.mock_molecule.return_value = {'mock': 'molecule'}

    def tearDown(self):
        self.patch_result.stop()
        self.patch_molecule.stop()

    def test_schema(self):
        """Test schema generation."""
        schema = self.interface.schema

        self.assertIsInstance(schema, dict)
        self.assertEqual(schema['name'], 'VQEDecoratedResult')
        self.assertEqual(len(schema['fields']), 8)

    def test_serialize(self):
        """Test serialization of VQEDecoratedResult."""
        serialized = self.interface.serialize(self.vqe_decorated_result)

        self.assertEqual(serialized['vqe_result'], {'mock': 'vqe_result'})
        self.assertEqual(serialized['molecule'], {'mock': 'molecule'})
        self.assertEqual(serialized['basis_set'], 'sto-3g')
        self.assertEqual(serialized['hamiltonian_time'], 1.5)
        self.assertEqual(serialized['mapping_time'], 0.5)
        self.assertEqual(serialized['vqe_time'], 10.5)
        self.assertEqual(serialized['total_time'], 12.5)
        self.assertEqual(serialized['molecule_id'], 1)

        self.mock_result.assert_called_once_with(self.vqe_decorated_result.vqe_result)
        self.mock_molecule.assert_called_once_with(self.vqe_decorated_result.molecule)

    def test_deserialize(self):
        """Test deserialization of VQEDecoratedResult."""
        with patch.object(
            self.interface.result_interface, 'deserialize'
        ) as mock_result_deserialize:
            with patch.object(
                self.interface.molecule_interface, 'deserialize'
            ) as mock_molecule_deserialize:
                mock_result_deserialize.return_value = self.vqe_decorated_result.vqe_result
                mock_molecule_deserialize.return_value = self.vqe_decorated_result.molecule

                data = {
                    'vqe_result': {'mock': 'vqe_result'},
                    'molecule': {'mock': 'molecule'},
                    'basis_set': 'sto-3g',
                    'hamiltonian_time': 1.5,
                    'mapping_time': 0.5,
                    'vqe_time': 10.5,
                    'total_time': 12.5,
                    'molecule_id': 1,
                }

                deserialized = self.interface.deserialize(data)

                self.assertEqual(deserialized.vqe_result, self.vqe_decorated_result.vqe_result)
                self.assertEqual(deserialized.molecule, self.vqe_decorated_result.molecule)
                self.assertEqual(deserialized.basis_set, 'sto-3g')
                self.assertEqual(deserialized.hamiltonian_time, np.float64(1.5))
                self.assertEqual(deserialized.mapping_time, np.float64(0.5))
                self.assertEqual(deserialized.vqe_time, np.float64(10.5))
                self.assertEqual(deserialized.total_time, np.float64(12.5))
                self.assertEqual(deserialized.molecule_id, 1)

                mock_result_deserialize.assert_called_once_with({'mock': 'vqe_result'})
                mock_molecule_deserialize.assert_called_once_with({'mock': 'molecule'})

    def test_to_avro_bytes(self):
        """Test conversion to Avro binary format."""
        mock_data = {
            'vqe_result': {
                'initial_data': {
                    'backend': 'fake_backend',
                    'num_qubits': 2,
                    'hamiltonian': [
                        {'label': 'ZZ', 'coefficients': {'real': 1.0, 'imaginary': 0.0}},
                        {'label': 'X', 'coefficients': {'real': 0.5, 'imaginary': 0.0}},
                    ],
                    'num_parameters': 3,
                    'initial_parameters': [0.1, 0.2, 0.3],
                    'optimizer': 'SPSA',
                    'ansatz': 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];',
                    'noise_backend': 'fake_noise_backend',
                    'default_shots': 1024,
                    'ansatz_reps': 1,
                },
                'iteration_list': [
                    {'iteration': 0, 'parameters': [0.1, 0.2, 0.3], 'result': -74.0, 'std': 0.02},
                    {
                        'iteration': 1,
                        'parameters': [0.15, 0.25, 0.35],
                        'result': -74.5,
                        'std': 0.01,
                    },
                ],
                'minimum': -74.5,
                'optimal_parameters': [0.15, 0.25, 0.35],
                'maxcv': 0.001,
                'minimization_time': 10.5,
            },
            'molecule': {
                'molecule_data': {
                    'symbols': ['H', 'H'],
                    'coords': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                    'multiplicity': 1,
                    'charge': 0,
                    'units': 'angstrom',
                    'masses': [1.0, 1.0],
                }
            },
            'basis_set': 'sto-3g',
            'hamiltonian_time': 1.5,
            'mapping_time': 0.5,
            'vqe_time': 10.5,
            'total_time': 12.5,
            'molecule_id': 1,
        }

        with patch.object(self.interface, 'serialize') as mock_serialize:
            mock_serialize.return_value = mock_data
            result = self.interface.to_avro_bytes(self.vqe_decorated_result)
            mock_serialize.assert_called_once_with(self.vqe_decorated_result)
            self.assertIsInstance(result, bytes)

    def test_from_avro_bytes_with_valid_confluent_header(self):
        """Test conversion from Avro binary format with valid Confluent header."""
        # header (magic byte + schema ID)
        test_bytes = b'\x00\x00\x00\x00{data}'

        # structured schema
        dummy_data = {
            'vqe_result': {
                'initial_data': {
                    'backend': 'fake_backend',
                    'num_qubits': 2,
                    'hamiltonian': [
                        {'label': 'ZZ', 'coefficients': {'real': 1.0, 'imaginary': 0.0}},
                        {'label': 'X', 'coefficients': {'real': 0.5, 'imaginary': 0.0}},
                    ],
                    'num_parameters': 3,
                    'initial_parameters': [0.1, 0.2, 0.3],
                    'optimizer': 'SPSA',
                    'ansatz': 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];',
                    'noise_backend': 'fake_noise_backend',
                    'default_shots': 1024,
                    'ansatz_reps': 1,
                },
                'iteration_list': [
                    {'iteration': 0, 'parameters': [0.1, 0.2, 0.3], 'result': -74.0, 'std': 0.02},
                    {
                        'iteration': 1,
                        'parameters': [0.15, 0.25, 0.35],
                        'result': -74.5,
                        'std': 0.01,
                    },
                ],
                'minimum': -74.5,
                'optimal_parameters': [0.15, 0.25, 0.35],
                'maxcv': 0.001,
                'minimization_time': 10.5,
            },
            'molecule': {
                'molecule_data': {
                    'symbols': ['H', 'H'],
                    'coords': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                    'multiplicity': 1,
                    'charge': 0,
                    'units': 'angstrom',
                    'masses': [1.0, 1.0],
                }
            },
            'basis_set': 'sto-3g',
            'hamiltonian_time': 1.5,
            'mapping_time': 0.5,
            'vqe_time': 10.5,
            'total_time': 12.5,
            'molecule_id': 1,
        }

        # patched deserialize method
        with patch.object(self.interface, 'deserialize') as mock_deserialize:
            mock_deserialize.return_value = self.vqe_decorated_result

            # avoid schema validation
            with patch('avro.schema.parse'):
                # datum reader should return the dummy data
                with patch('avro.io.DatumReader.read', return_value=dummy_data):
                    result = self.interface.from_avro_bytes(test_bytes)

                    # verify whether the method was called with the dummy data
                    mock_deserialize.assert_called_once_with(dummy_data)

                    # verify if everything aligns
                    self.assertEqual(result, self.vqe_decorated_result)

    def test_get_result_suffix(self):
        """Test get_result_suffix method."""
        mock_iteration_list = MagicMock()
        mock_iteration_list.__len__.return_value = 10
        self.vqe_decorated_result.vqe_result.iteration_list = mock_iteration_list

        result = self.vqe_decorated_result.get_result_suffix()

        self.assertEqual(result, '_it10')

    def test_get_schema_suffix(self):
        """Test get_schema_suffix method."""

        # mock iteration list
        mock_iteration_list = MagicMock()
        mock_iteration_list.__len__.return_value = 10
        self.vqe_decorated_result.vqe_result.iteration_list = mock_iteration_list

        # mock initial data
        mock_initial_data = MagicMock()
        mock_initial_data.backend = 'fake-backend'
        self.vqe_decorated_result.vqe_result.initial_data = mock_initial_data

        self.vqe_decorated_result.molecule.symbols = ['H', 'H']
        self.vqe_decorated_result.molecule_id = 42
        self.vqe_decorated_result.basis_set = 'sto-3g'
        self.vqe_decorated_result.vqe_result.initial_data.backend = 'fake-backend'

        result = self.vqe_decorated_result.get_schema_suffix()

        expected = '_mol42_HH_it10_bs_sto_3g_bk_fake_backend'
        self.assertEqual(result, expected)


class TestEndToEndSerialization(unittest.TestCase):
    """End-to-end tests for serialization and deserialization."""

    def setUp(self):
        self.registry = MockSchemaRegistry()

        # Create test fixture for complete VQEDecoratedResult

        # 1. Create a quantum circuit for ansatz
        mock_circuit = MagicMock(spec=QuantumCircuit)
        mock_circuit.__str__.return_value = 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];'

        # 2. Create VQEInitialData
        initial_data = VQEInitialData(
            backend='fake_backend',
            num_qubits=2,
            hamiltonian=np.array([('ZZ', complex(1.0, 0.0)), ('X', complex(0.5, 0.0))]),
            num_parameters=3,
            initial_parameters=np.array([0.1, 0.2, 0.3]),
            optimizer='SPSA',
            ansatz=mock_circuit,
            ansatz_reps=1,
            noise_backend='fake_noise_backend',
            default_shots=1024,
        )

        # 3. Create VQEProcess objects for iteration list
        process1 = VQEProcess(
            iteration=0,
            parameters=np.array([0.1, 0.2, 0.3]),
            result=np.float64(-74.0),
            std=np.float64(0.02),
        )

        process2 = VQEProcess(
            iteration=1,
            parameters=np.array([0.15, 0.25, 0.35]),
            result=np.float64(-74.5),
            std=np.float64(0.01),
        )

        # 4. Create VQEResult
        vqe_result = VQEResult(
            initial_data=initial_data,
            iteration_list=[process1, process2],
            minimum=np.float64(-74.5),
            optimal_parameters=np.array([0.15, 0.25, 0.35]),
            maxcv=np.float64(0.001),
            minimization_time=np.float64(10.5),
        )

        # 5. Create MoleculeInfo
        molecule = MoleculeInfo(
            symbols=['H', 'H'],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            multiplicity=1,
            charge=0,
            units=DistanceUnit.ANGSTROM,
            masses=np.array([1.0, 1.0]),
        )

        # 6. Create VQEDecoratedResult
        self.vqe_decorated_result = VQEDecoratedResult(
            vqe_result=vqe_result,
            molecule=molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(1.5),
            mapping_time=np.float64(0.5),
            vqe_time=np.float64(10.5),
            total_time=np.float64(12.5),
            molecule_id=1,
        )

        # Create interfaces
        self.process_interface = VQEProcessInterface(self.registry)
        self.initial_data_interface = VQEInitialDataInterface(self.registry)
        self.result_interface = VQEResultInterface(self.registry)
        self.molecule_interface = MoleculeInfoInterface(self.registry)
        self.decorated_result_interface = VQEDecoratedResultInterface(self.registry)

    @patch('qiskit.qasm3.dumps')
    @patch('qiskit.qasm3.loads')
    def test_end_to_end_vqe_process(self, mock_loads, mock_dumps):
        """Test serialization/deserialization of VQEProcess."""

        # get vqe process object
        process = self.vqe_decorated_result.vqe_result.iteration_list[0]

        # serialize
        serialized_dict = self.process_interface.serialize(process)

        # deserialize back to the object
        deserialized_obj = self.process_interface.deserialize(serialized_dict)

        # ensure everything is the same
        self.assertEqual(deserialized_obj.iteration, process.iteration)
        self.assertTrue(np.array_equal(deserialized_obj.parameters, process.parameters))
        self.assertEqual(deserialized_obj.result, process.result)
        self.assertEqual(deserialized_obj.std, process.std)

    @patch('qiskit.qasm3.dumps')
    @patch('qiskit.qasm3.loads')
    def test_end_to_end_molecule_info(self, mock_loads, mock_dumps):
        """Test serialization/deserialization of MoleculeInfo."""

        # get molecule object
        molecule = self.vqe_decorated_result.molecule

        # serialize -> deserialize
        serialized_dict = self.molecule_interface.serialize(molecule)
        deserialized_obj = self.molecule_interface.deserialize(serialized_dict)

        # ensure reconstructed correctly
        self.assertEqual(deserialized_obj.symbols, molecule.symbols)
        self.assertEqual(deserialized_obj.coords, molecule.coords)
        self.assertEqual(deserialized_obj.multiplicity, molecule.multiplicity)
        self.assertEqual(deserialized_obj.charge, molecule.charge)
        self.assertEqual(deserialized_obj.units, molecule.units)
        self.assertTrue(np.array_equal(deserialized_obj.masses, molecule.masses))

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.dumps')
    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    @patch('avro.io.BinaryEncoder')
    @patch('avro.io.BinaryDecoder')
    @patch('avro.io.DatumWriter.write')
    @patch('avro.io.DatumReader.read')
    def test_end_to_end_decorated_result_avro_bytes(
        self, mock_read, mock_write, mock_decoder, mock_encoder, mock_loads, mock_dumps
    ):
        """Test round-trip serialization of VQEDecoratedResult to and from Avro bytes."""

        # setup mocks
        mock_dumps.return_value = 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];'
        mock_loads.return_value = self.vqe_decorated_result.vqe_result.initial_data.ansatz

        # mock avro serialization/deserialization (just return the object)
        def side_effect_write(obj, encoder):
            return obj

        # reader should return the original, serialized form
        def side_effect_read(decoder):
            return self.decorated_result_interface.serialize(self.vqe_decorated_result)

        # assign write/read side effects to mocks
        mock_write.side_effect = side_effect_write
        mock_read.side_effect = side_effect_read

        # serialize and ensure bytes were produced
        avro_bytes = self.decorated_result_interface.to_avro_bytes(self.vqe_decorated_result)
        self.assertIsInstance(avro_bytes, bytes)

        with patch('avro.schema.parse'):
            # deserialize and mock schema parsing to avoid validation
            # and check if we get the original object back
            deserialized_obj = self.decorated_result_interface.from_avro_bytes(avro_bytes)
            self.assertIsInstance(deserialized_obj, VQEDecoratedResult)

            # confirm each mock was called appropriately
            mock_dumps.assert_called()
            mock_write.assert_called()
            mock_read.assert_called()
