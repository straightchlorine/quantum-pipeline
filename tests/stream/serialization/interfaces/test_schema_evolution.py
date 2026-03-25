"""Schema evolution tests for QUA-19.

Verifies Avro backward compatibility:
- Old Avro records (without QUA-15 fields) can be read with the new schema.
- New fields receive their declared defaults (null / 'random' / 'EfficientSU2').
- New records round-trip correctly with all fields populated.

Tests operate at the binary Avro level (writer schema != reader schema) to
mimic what happens when the existing ~38k thesis-1 records are processed by
the updated pipeline.
"""

import io
import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import avro.schema
import numpy as np
from avro.io import BinaryDecoder, BinaryEncoder, DatumReader, DatumWriter

from quantum_pipeline.stream.serialization.interfaces.vqe import (
    VQEInitialDataInterface,
    VQEResultInterface,
)
from quantum_pipeline.structures.vqe_observation import (
    VQEInitialData,
    VQEProcess,
    VQEResult,
)


class MockSchemaRegistry:
    def __init__(self):
        self.schemas = {}
        self.id_cache = {}

    def get_schema(self, name):
        if name not in self.schemas:
            raise FileNotFoundError(f'Schema {name} not found')
        return self.schemas[name]

    def save_schema(self, name, schema):
        self.schemas[name] = schema
        self.id_cache[name] = 1


# ---------------------------------------------------------------------------
# Old schemas (pre-QUA-15) — no ML fields
# ---------------------------------------------------------------------------

OLD_VQE_INITIAL_SCHEMA = {
    'type': 'record',
    'name': 'VQEInitialData',
    'fields': [
        {'name': 'backend', 'type': 'string'},
        {'name': 'num_qubits', 'type': 'int'},
        {
            'name': 'hamiltonian',
            'type': {
                'type': 'array',
                'items': {
                    'type': 'record',
                    'name': 'HamiltonianTerm',
                    'fields': [
                        {'name': 'label', 'type': 'string'},
                        {
                            'name': 'coefficients',
                            'type': {
                                'type': 'record',
                                'name': 'ComplexNumber',
                                'fields': [
                                    {'name': 'real', 'type': 'double'},
                                    {'name': 'imaginary', 'type': 'double'},
                                ],
                            },
                        },
                    ],
                },
            },
        },
        {'name': 'num_parameters', 'type': 'int'},
        {'name': 'initial_parameters', 'type': {'type': 'array', 'items': 'double'}},
        {'name': 'optimizer', 'type': 'string'},
        {'name': 'ansatz', 'type': 'string'},
        {'name': 'noise_backend', 'type': 'string'},
        {'name': 'default_shots', 'type': 'int'},
        {'name': 'ansatz_reps', 'type': 'int'},
        # init_strategy, seed, ansatz_name intentionally absent (old schema)
    ],
}

_VQE_PROCESS_SCHEMA = {
    'type': 'record',
    'name': 'VQEProcess',
    'fields': [
        {'name': 'iteration', 'type': 'int'},
        {'name': 'parameters', 'type': {'type': 'array', 'items': 'double'}},
        {'name': 'result', 'type': 'double'},
        {'name': 'std', 'type': 'double'},
        {'name': 'energy_delta', 'type': ['null', 'double'], 'default': None},
        {'name': 'parameter_delta_norm', 'type': ['null', 'double'], 'default': None},
        {'name': 'cumulative_min_energy', 'type': ['null', 'double'], 'default': None},
    ],
}

OLD_VQE_RESULT_SCHEMA = {
    'type': 'record',
    'name': 'VQEResult',
    'fields': [
        {'name': 'initial_data', 'type': OLD_VQE_INITIAL_SCHEMA},
        {'name': 'iteration_list', 'type': {'type': 'array', 'items': _VQE_PROCESS_SCHEMA}},
        {'name': 'minimum', 'type': 'double'},
        {'name': 'optimal_parameters', 'type': {'type': 'array', 'items': 'double'}},
        {'name': 'maxcv', 'type': ['null', 'double'], 'default': None},
        {'name': 'minimization_time', 'type': 'double'},
        # nuclear_repulsion_energy, success, nfev, nit intentionally absent
    ],
}


def _avro_bytes(schema_dict: dict, record: dict) -> bytes:
    """Write a single Avro record to bytes using the given schema."""
    parsed = avro.schema.parse(json.dumps(schema_dict))
    buf = io.BytesIO()
    encoder = BinaryEncoder(buf)
    DatumWriter(parsed).write(record, encoder)
    return buf.getvalue()


def _read_avro_bytes(writer_schema_dict: dict, reader_schema_dict: dict, data: bytes) -> dict:
    """Read Avro bytes with schema evolution (writer != reader)."""
    writer_schema = avro.schema.parse(json.dumps(writer_schema_dict))
    reader_schema = avro.schema.parse(json.dumps(reader_schema_dict))
    buf = io.BytesIO(data)
    decoder = BinaryDecoder(buf)
    return DatumReader(writer_schema, reader_schema).read(decoder)


class TestVQEInitialDataSchemaEvolution(unittest.TestCase):
    """Schema evolution for VQEInitialData — new fields: init_strategy, seed, ansatz_name."""

    def setUp(self):
        self.registry = MockSchemaRegistry()
        self.interface = VQEInitialDataInterface(self.registry)
        self.new_schema = self.interface.schema

    def _old_record(self):
        return {
            'backend': 'qasm_simulator',
            'num_qubits': 4,
            'hamiltonian': [
                {'label': 'ZZ', 'coefficients': {'real': -1.0, 'imaginary': 0.0}}
            ],
            'num_parameters': 4,
            'initial_parameters': [0.1, 0.2, 0.3, 0.4],
            'optimizer': 'L-BFGS-B',
            'ansatz': 'OPENQASM 3.0;',
            'noise_backend': 'none',
            'default_shots': 1024,
            'ansatz_reps': 1,
        }

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_old_record_reads_with_new_schema(self, mock_loads):
        """Old Avro bytes (no ML fields) must be readable with the new schema."""
        mock_loads.return_value = object()
        old_bytes = _avro_bytes(OLD_VQE_INITIAL_SCHEMA, self._old_record())
        record = _read_avro_bytes(OLD_VQE_INITIAL_SCHEMA, self.new_schema, old_bytes)

        # New fields should have their schema defaults
        self.assertIsNone(record.get('seed'))
        # init_strategy default is 'random'
        self.assertEqual(record.get('init_strategy'), 'random')
        # ansatz_name default is None (nullable)
        self.assertIsNone(record.get('ansatz_name'))

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_old_record_deserializes_with_safe_defaults(self, mock_loads):
        """Deserialize old record; init_strategy defaults to 'random', ansatz_name to 'EfficientSU2'."""
        mock_loads.return_value = object()
        old_bytes = _avro_bytes(OLD_VQE_INITIAL_SCHEMA, self._old_record())
        record = _read_avro_bytes(OLD_VQE_INITIAL_SCHEMA, self.new_schema, old_bytes)

        deserialized = self.interface.deserialize(record)
        self.assertEqual(deserialized.init_strategy, 'random')
        self.assertEqual(deserialized.ansatz_name, 'EfficientSU2')
        self.assertIsNone(deserialized.seed)

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.dumps')
    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_new_record_round_trips(self, mock_loads, mock_dumps):
        """New records with all ML fields round-trip without data loss."""
        mock_circuit = object()
        mock_loads.return_value = mock_circuit
        mock_dumps.return_value = 'OPENQASM 3.0;'

        obj = VQEInitialData(
            backend='qasm_simulator',
            num_qubits=4,
            hamiltonian=np.array([('ZZ', complex(-1.0, 0.0))], dtype=object),
            num_parameters=4,
            initial_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            optimizer='L-BFGS-B',
            ansatz=mock_circuit,
            ansatz_reps=1,
            noise_backend='none',
            default_shots=1024,
            init_strategy='hf',
            seed=42,
            ansatz_name='RealAmplitudes',
        )
        serialized = self.interface.serialize(obj)
        deserialized = self.interface.deserialize(serialized)

        self.assertEqual(deserialized.init_strategy, 'hf')
        self.assertEqual(deserialized.seed, 42)
        self.assertEqual(deserialized.ansatz_name, 'RealAmplitudes')


class TestVQEResultSchemaEvolution(unittest.TestCase):
    """Schema evolution for VQEResult — new fields: nuclear_repulsion_energy, success, nfev, nit."""

    def setUp(self):
        self.registry = MockSchemaRegistry()
        self.interface = VQEResultInterface(self.registry)
        self.new_schema = self.interface.schema

    def _old_record(self):
        return {
            'initial_data': {
                'backend': 'qasm_simulator',
                'num_qubits': 4,
                'hamiltonian': [
                    {'label': 'ZZ', 'coefficients': {'real': -1.0, 'imaginary': 0.0}}
                ],
                'num_parameters': 4,
                'initial_parameters': [0.1, 0.2, 0.3, 0.4],
                'optimizer': 'L-BFGS-B',
                'ansatz': 'OPENQASM 3.0;',
                'noise_backend': 'none',
                'default_shots': 1024,
                'ansatz_reps': 1,
            },
            'iteration_list': [
                {
                    'iteration': 0,
                    'parameters': [0.1, 0.2, 0.3, 0.4],
                    'result': -1.1,
                    'std': 0.01,
                    'energy_delta': None,
                    'parameter_delta_norm': None,
                    'cumulative_min_energy': -1.1,
                }
            ],
            'minimum': -1.1,
            'optimal_parameters': [0.1, 0.2, 0.3, 0.4],
            'maxcv': None,
            'minimization_time': 5.0,
            # nuclear_repulsion_energy, success, nfev, nit absent
        }

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_old_record_reads_with_new_schema(self, mock_loads):
        """Old Avro bytes (no nuclear_repulsion_energy / success / nfev / nit) must be readable."""
        mock_loads.return_value = object()
        old_bytes = _avro_bytes(OLD_VQE_RESULT_SCHEMA, self._old_record())
        record = _read_avro_bytes(OLD_VQE_RESULT_SCHEMA, self.new_schema, old_bytes)

        self.assertIsNone(record.get('nuclear_repulsion_energy'))
        self.assertIsNone(record.get('success'))
        self.assertIsNone(record.get('nfev'))
        self.assertIsNone(record.get('nit'))

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_old_record_deserializes_without_crash(self, mock_loads):
        """Deserializing old record must not raise and must produce None for new fields."""
        mock_loads.return_value = object()
        old_bytes = _avro_bytes(OLD_VQE_RESULT_SCHEMA, self._old_record())
        record = _read_avro_bytes(OLD_VQE_RESULT_SCHEMA, self.new_schema, old_bytes)

        result = self.interface.deserialize(record)
        self.assertIsNone(result.nuclear_repulsion_energy)
        self.assertIsNone(result.success)
        self.assertIsNone(result.nfev)
        self.assertIsNone(result.nit)
        self.assertAlmostEqual(float(result.minimum), -1.1)

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.dumps')
    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_new_record_round_trips_ml_fields(self, mock_loads, mock_dumps):
        """New records with all ML fields round-trip correctly."""
        mock_circuit = object()
        mock_loads.return_value = mock_circuit
        mock_dumps.return_value = 'OPENQASM 3.0;'

        initial_data = VQEInitialData(
            backend='qasm_simulator',
            num_qubits=4,
            hamiltonian=np.array([('ZZ', complex(-1.0, 0.0))], dtype=object),
            num_parameters=4,
            initial_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            optimizer='L-BFGS-B',
            ansatz=mock_circuit,
            ansatz_reps=1,
            noise_backend='none',
            default_shots=1024,
            seed=42,
        )
        process = VQEProcess(
            iteration=0,
            parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            result=np.float64(-1.1),
            std=np.float64(0.01),
        )
        obj = VQEResult(
            initial_data=initial_data,
            iteration_list=[process],
            minimum=np.float64(-1.1),
            optimal_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            maxcv=None,
            minimization_time=np.float64(5.0),
            nuclear_repulsion_energy=np.float64(0.715),
            success=True,
            nfev=20,
            nit=2,
        )
        serialized = self.interface.serialize(obj)
        deserialized = self.interface.deserialize(serialized)

        self.assertAlmostEqual(float(deserialized.nuclear_repulsion_energy), 0.715)
        self.assertTrue(deserialized.success)
        self.assertEqual(deserialized.nfev, 20)
        self.assertEqual(deserialized.nit, 2)

    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.dumps')
    @patch('quantum_pipeline.stream.serialization.interfaces.vqe.loads')
    def test_null_ml_fields_round_trip(self, mock_loads, mock_dumps):
        """New records with None ML fields (e.g. gradient-free optimizers) round-trip correctly."""
        mock_circuit = object()
        mock_loads.return_value = mock_circuit
        mock_dumps.return_value = 'OPENQASM 3.0;'

        initial_data = VQEInitialData(
            backend='qasm_simulator',
            num_qubits=4,
            hamiltonian=np.array([('ZZ', complex(-1.0, 0.0))], dtype=object),
            num_parameters=4,
            initial_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            optimizer='Nelder-Mead',
            ansatz=mock_circuit,
            ansatz_reps=1,
            noise_backend='none',
            default_shots=1024,
        )
        process = VQEProcess(
            iteration=0,
            parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            result=np.float64(-1.1),
            std=np.float64(0.01),
        )
        obj = VQEResult(
            initial_data=initial_data,
            iteration_list=[process],
            minimum=np.float64(-1.1),
            optimal_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            maxcv=None,
            minimization_time=np.float64(5.0),
            nuclear_repulsion_energy=None,
            success=None,
            nfev=None,
            nit=None,
        )
        serialized = self.interface.serialize(obj)
        deserialized = self.interface.deserialize(serialized)

        self.assertIsNone(deserialized.nuclear_repulsion_energy)
        self.assertIsNone(deserialized.success)
        self.assertIsNone(deserialized.nfev)
        self.assertIsNone(deserialized.nit)


if __name__ == '__main__':
    unittest.main()
