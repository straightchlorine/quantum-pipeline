from abc import ABC, abstractmethod
from copy import deepcopy
import io
import json
from typing import Any, Generic, TypeVar

import numpy as np
from numpy import float32, float64, int32, int64, ndarray

from avro.io import BinaryDecoder, BinaryEncoder, DatumReader, DatumWriter
import avro.schema
from qiskit.qasm3 import dumps, loads
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.units import DistanceUnit
from quantum_pipeline.structures.vqe_observation import (
    VQEDecoratedResult,
    VQEInitialData,
    VQEProcess,
    VQEResult,
)
from quantum_pipeline.utils.logger import get_logger

T = TypeVar('T')


class AvroInterfaceBase(ABC, Generic[T]):
    """Base class for Avro serializers."""

    def __init__(self, registry):
        self.registry = registry
        self.logger = get_logger(self.__class__.__name__)

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return Avro schema for the type."""

    @abstractmethod
    def serialize(self, obj: T) -> dict[str, Any]:
        """Convert object to Avro-compatible dictionary."""

    @abstractmethod
    def deserialize(self, data: dict[str, Any]) -> T:
        """Convert Avro-compatible dictionary to object."""

    def _is_numpy_int(self, obj: Any) -> bool:
        """Check if object is a numpy integer type."""
        return isinstance(obj, int32 | int64)  # type: ignore

    def _is_numpy_float(self, obj: Any) -> bool:
        """Check if object is a numpy float type."""
        return isinstance(obj, float32 | float64)  # type: ignore

    def _convert_to_primitives(self, obj: Any) -> Any:
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif self._is_numpy_int(obj):
            return int(obj)
        elif self._is_numpy_float(obj):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_primitives(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_primitives(item) for item in obj]
        return obj

    def _convert_to_numpy(self, obj: Any) -> Any:
        """Convert Python native types to numpy types."""
        if isinstance(obj, list):
            return np.array([self._convert_to_primitives(item) for item in obj])
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj

    def to_avro_bytes(self, obj: T, schema_name: str = 'vqe_decorated_result') -> bytes:
        """Convert object to Avro binary format."""
        self.logger.debug(f'Serializing object with schema {schema_name}.')
        schema = self.schema

        parsed_schema = ''
        if isinstance(schema, dict):
            parsed_schema = avro.schema.parse(json.dumps(schema))
            self.logger.debug(f'Parsed dict schema: {parsed_schema}')
        elif isinstance(schema, str):
            parsed_schema = avro.schema.parse(schema)
            self.logger.debug(f'Parsed string schema: {parsed_schema}')

        writer = DatumWriter(parsed_schema)
        bytes_writer = io.BytesIO()

        if self.registry.id_cache.get(schema_name, False):
            self.logger.debug(
                f'Writing Confluent Schema Registry header for schema {schema_name}.'
            )
            bytes_writer.write(bytes([0]))
            bytes_writer.write(self.registry.id_cache[schema_name].to_bytes(4, byteorder='big'))
        else:
            self.logger.warning(
                'Unable to find id of the schema in the id cache. '
                'Serializing without Confluent Schema Registry header.'
            )

        encoder = BinaryEncoder(bytes_writer)
        writer.write(self.serialize(obj), encoder)
        return bytes_writer.getvalue()

    def from_avro_bytes(self, avro_bytes: bytes) -> T:
        """Convert Avro binary format to object."""
        bytes_reader = io.BytesIO(avro_bytes)

        # read the magic byte
        magic_byte = bytes_reader.read(1)
        if magic_byte != bytes([0]):
            raise ValueError(f'Invalid magic byte: {magic_byte}. Expected: {bytes([0])}')

        # read the schema id
        bytes_reader.read(4)

        # parse the schema and deserialize the object
        parsed_schema = avro.schema.parse(self.schema)
        reader = DatumReader(parsed_schema)
        decoder = BinaryDecoder(bytes_reader)
        return self.deserialize(reader.read(decoder))


class VQEProcessInterface(AvroInterfaceBase[VQEProcess]):
    @property
    def schema(self) -> dict[str, Any]:
        try:
            return self.registry.get_schema('vqe_process')
        except FileNotFoundError:
            schema = {
                'type': 'record',
                'name': 'VQEProcess',
                'fields': [
                    {'name': 'iteration', 'type': 'int'},
                    {'name': 'parameters', 'type': {'type': 'array', 'items': 'double'}},
                    {'name': 'result', 'type': 'double'},
                    {'name': 'std', 'type': 'double'},
                ],
            }
            dict_schema = deepcopy(schema)
            self.registry.save_schema('vqe_process', schema)
            return dict_schema

    def serialize(self, obj: VQEProcess) -> dict[str, Any]:
        return {
            'iteration': obj.iteration,
            'parameters': self._convert_to_primitives(obj.parameters),
            'result': float(obj.result),
            'std': float(obj.std),
        }

    def deserialize(self, data: dict[str, Any]) -> VQEProcess:
        return VQEProcess(
            iteration=data['iteration'],
            parameters=self._convert_to_numpy(data['parameters']),
            result=float64(data['result']),
            std=float64(data['std']),
        )


class VQEInitialDataInterface(AvroInterfaceBase[VQEInitialData]):
    @property
    def schema(self) -> dict[str, Any]:
        try:
            return self.registry.get_schema('vqe_initial')
        except FileNotFoundError:
            schema = {
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
                ],
            }
            dict_schema = deepcopy(schema)
            self.registry.save_schema('vqe_initial', schema)
            return dict_schema

    def _serialize_hamiltonian(self, data: ndarray):
        serialized_data = []

        for label, complex_number in data:
            if isinstance(complex_number, (str, np.str_)):
                complex_number = complex(
                    complex_number.replace('(', '').replace(')', ''),
                )

            real_part = np.float64(complex_number.real)
            imaginary_part = np.float64(complex_number.imag)

            serialized_data.append(
                {'label': label, 'coefficients': {'real': real_part, 'imaginary': imaginary_part}}
            )

        return serialized_data

    def _deserialize_hamiltonian(self, data: list):
        return np.array(
            [
                (
                    term['label'],
                    complex(term['coefficients']['real'], term['coefficients']['imaginary']),
                )
                for term in data
            ],
            dtype=object,  # maintain the orignal dtype
        )

    def serialize(self, obj: VQEInitialData) -> dict[str, Any]:
        return {
            'backend': obj.backend,
            'num_qubits': obj.num_qubits,
            'hamiltonian': self._serialize_hamiltonian(obj.hamiltonian),
            'num_parameters': obj.num_parameters,
            'initial_parameters': self._convert_to_primitives(obj.initial_parameters),
            'optimizer': obj.optimizer,
            'ansatz': dumps(obj.ansatz),
            'ansatz_reps': obj.ansatz_reps,
            'noise_backend': obj.noise_backend,
            'default_shots': obj.default_shots,
        }

    def deserialize(self, data: dict[str, Any]) -> VQEInitialData:
        return VQEInitialData(
            backend=data['backend'],
            num_qubits=data['num_qubits'],
            hamiltonian=self._deserialize_hamiltonian(data['hamiltonian']),
            num_parameters=data['num_parameters'],
            initial_parameters=self._convert_to_numpy(data['initial_parameters']),
            optimizer=data['optimizer'],
            ansatz=loads(data['ansatz']),
            ansatz_reps=data['ansatz_reps'],
            noise_backend=data['noise_backend'],
            default_shots=data['default_shots'],
        )


class VQEResultInterface(AvroInterfaceBase[VQEResult]):
    def __init__(self, registry):
        super().__init__(registry)
        self.initial_data_interface = VQEInitialDataInterface(self.registry)
        self.process_interface = VQEProcessInterface(self.registry)
        self.schema_name = 'vqe_result'

    @property
    def schema(self) -> dict[str, Any]:
        try:
            return self.registry.get_schema(self.schema_name)
        except FileNotFoundError:
            schema = {
                'type': 'record',
                'name': 'VQEResult',
                'fields': [
                    {'name': 'initial_data', 'type': self.initial_data_interface.schema},
                    {
                        'name': 'iteration_list',
                        'type': {'type': 'array', 'items': self.process_interface.schema},
                    },
                    {'name': 'minimum', 'type': 'double'},
                    {'name': 'optimal_parameters', 'type': {'type': 'array', 'items': 'double'}},
                    {'name': 'maxcv', 'type': 'double'},
                    {'name': 'minimization_time', 'type': 'double'},
                ],
            }
            dict_schema = deepcopy(schema)
            self.registry.save_schema(self.schema_name, schema)
            return dict_schema

    def serialize(self, obj: VQEResult) -> dict[str, Any]:
        return {
            'initial_data': self.initial_data_interface.serialize(obj.initial_data),
            'iteration_list': [self.process_interface.serialize(p) for p in obj.iteration_list],
            'minimum': float(obj.minimum),
            'optimal_parameters': self._convert_to_primitives(obj.optimal_parameters),
            'maxcv': float(obj.maxcv),
            'minimization_time': float(obj.minimization_time),
        }

    def deserialize(self, data: dict[str, Any]) -> VQEResult:
        return VQEResult(
            initial_data=self.initial_data_interface.deserialize(data['initial_data']),
            iteration_list=[self.process_interface.deserialize(p) for p in data['iteration_list']],
            minimum=float64(data['minimum']),
            optimal_parameters=self._convert_to_numpy(data['optimal_parameters']),
            maxcv=float64(data['maxcv']),
            minimization_time=float64(data['minimization_time']),
        )


class MoleculeInfoInterface(AvroInterfaceBase[MoleculeInfo]):
    @property
    def schema(self) -> dict[str, Any]:
        try:
            return self.registry.get_schema('vqe_molecule')
        except FileNotFoundError:
            schema = {
                'type': 'record',
                'name': 'MoleculeInfo',
                'namespace': 'quantum_pipeline',
                'fields': [
                    {
                        'name': 'molecule_data',
                        'type': {
                            'type': 'record',
                            'name': 'MoleculeData',
                            'fields': [
                                {'name': 'symbols', 'type': {'type': 'array', 'items': 'string'}},
                                {
                                    'name': 'coords',
                                    'type': {
                                        'type': 'array',
                                        'items': {'type': 'array', 'items': 'double'},
                                    },
                                },
                                {'name': 'multiplicity', 'type': 'int'},
                                {'name': 'charge', 'type': 'int'},
                                {'name': 'units', 'type': 'string'},
                                {
                                    'name': 'masses',
                                    'type': ['null', {'type': 'array', 'items': 'double'}],
                                    'default': None,
                                },
                            ],
                        },
                    }
                ],
            }
            dict_schema = deepcopy(schema)
            self.registry.save_schema('vqe_molecule', schema)
            return dict_schema

    def serialize(self, obj: MoleculeInfo) -> dict[str, Any]:
        return {
            'molecule_data': {
                'symbols': self._convert_to_primitives(obj.symbols),
                'coords': self._convert_to_primitives(obj.coords),
                'multiplicity': obj.multiplicity,
                'charge': obj.charge,
                'units': obj.units.value.lower(),
                'masses': self._convert_to_primitives(obj.masses),
            }
        }

    def deserialize(self, data: dict[str, Any]) -> MoleculeInfo:
        mol = data['molecule_data']

        # coordinates to nested list if they're flattened
        coords = mol['coords']
        if not isinstance(coords[0], list):
            coords = np.array(coords).reshape(-1, 3).tolist()

        # handle masses which might be None
        masses = mol.get('masses')
        if masses is not None:
            masses = self._convert_to_numpy(masses)

        return MoleculeInfo(
            symbols=list(mol['symbols']),
            coords=coords,
            multiplicity=mol['multiplicity'],
            charge=mol['charge'],
            units=getattr(DistanceUnit, mol['units'].upper()),
            masses=masses,
        )


class VQEDecoratedResultInterface(AvroInterfaceBase[VQEDecoratedResult]):
    def __init__(self, registry):
        super().__init__(registry)
        self.result_interface = VQEResultInterface(self.registry)
        self.molecule_interface = MoleculeInfoInterface(self.registry)
        self.schema_name = 'vqe_decorated_result'

    @property
    def schema(self) -> dict[str, Any]:
        try:
            return self.registry.get_schema(self.schema_name)
        except FileNotFoundError:
            schema = {
                'type': 'record',
                'name': 'VQEDecoratedResult',
                'fields': [
                    {'name': 'vqe_result', 'type': self.result_interface.schema},
                    {'name': 'molecule', 'type': self.molecule_interface.schema},
                    {'name': 'basis_set', 'type': 'string'},
                    {'name': 'hamiltonian_time', 'type': 'double'},
                    {'name': 'mapping_time', 'type': 'double'},
                    {'name': 'vqe_time', 'type': 'double'},
                    {'name': 'total_time', 'type': 'double'},
                    {'name': 'molecule_id', 'type': 'int'},
                ],
            }
            dict_schema = deepcopy(schema)
            self.registry.save_schema(self.schema_name, schema)
            return dict_schema

    def serialize(self, obj: VQEDecoratedResult) -> dict[str, Any]:
        return {
            'vqe_result': self.result_interface.serialize(obj.vqe_result),
            'molecule': self.molecule_interface.serialize(obj.molecule),
            'basis_set': obj.basis_set,
            'hamiltonian_time': float(obj.hamiltonian_time),
            'mapping_time': float(obj.mapping_time),
            'vqe_time': float(obj.vqe_time),
            'total_time': float(obj.total_time),
            'molecule_id': obj.molecule_id,
        }

    def deserialize(self, data: dict[str, Any]) -> VQEDecoratedResult:
        return VQEDecoratedResult(
            vqe_result=self.result_interface.deserialize(data['vqe_result']),
            molecule=self.molecule_interface.deserialize(data['molecule']),
            basis_set=data['basis_set'],
            hamiltonian_time=float64(data['hamiltonian_time']),
            mapping_time=float64(data['mapping_time']),
            vqe_time=float64(data['vqe_time']),
            total_time=float64(data['total_time']),
            molecule_id=int(data['molecule_id']),
        )
