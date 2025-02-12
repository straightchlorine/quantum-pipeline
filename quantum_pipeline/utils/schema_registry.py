import json
from typing import Any

from avro import schema

from quantum_pipeline.configs.settings import SCHEMA_DIR
from quantum_pipeline.utils.logger import get_logger


class SchemaRegistry:
    def __init__(self):
        self.schema_dir = SCHEMA_DIR
        self.schema_cache: dict[str, dict[str, Any]] = {}
        self.logger = get_logger(self.__class__.__name__)

    def get_schema(self, schema_name: str) -> dict[str, Any]:
        """
        Get Avro schema from file system.

        Args:
            schema_name: Name of the schema without extension

        Returns:
            Dict containing the Avro schema

        Raises:
            FileNotFoundError: If schema file doesn't exist
            ValueError: If schema is invalid
        """
        self.logger.debug(f'Checking the {schema_name} schema registry cache...')
        if schema_name in self.schema_cache:
            self.logger.info(f'Found cached {schema_name} schema.')
            return self.schema_cache[schema_name]

        self.logger.debug(f'Checking schema directory for {schema_name}...')
        schema_file = self.schema_dir / f'{schema_name}.avsc'
        if not schema_file.exists():
            self.logger.error(f'Unable to find {schema_name} locally')
            raise FileNotFoundError(f'Schema file not found: {schema_file}')

        self.logger.info(f'Found schema {schema_name} locally.')
        try:
            with open(schema_file) as f:
                schema_dict = json.load(f)

            self.logger.debug(f'Validating the {schema_name}...')
            schema.parse(json.dumps(schema_dict))

            self.logger.debug(f'Validation passed, caching the {schema_name} schema.')
            self.schema_cache[schema_name] = schema_dict
            return schema_dict

        except json.JSONDecodeError as e:
            self.logger.error(f'Validation of the {schema_name} schema failed.')
            raise ValueError(f'Invalid JSON in schema file {schema_file}: {str(e)}')
        except Exception as e:
            self.logger.error(f'Unknown error during loading the {schema_name} schema.')
            raise ValueError(f'Invalid Avro schema in {schema_file}: {str(e)}')

    def save_schema(self, schema_name: str, schema_dict: dict[str, Any]) -> None:
        """
        Save the given schema to the file system if it is different from the existing one.

        Args:
            schema_name: Name of the schema (without extension).
            schema_dict: The Avro schema dictionary to save.

        Raises:
            ValueError: If the provided schema is invalid.
            IOError: If there is an issue writing to the file.
        """
        schema_file = self.schema_dir / f'{schema_name}.avsc'

        self.logger.info('Validating the schema dict...')
        self.logger.debug(f'\n\n{schema_dict}\n\n')
        try:
            schema.parse(json.dumps(schema_dict))
        except Exception as e:
            self.logger.error('Invalid Avro schema.')
            raise ValueError(f'Invalid Avro schema: {e}')
        self.schema_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Checking if {schema_name} exists...')
        if schema_file.exists():
            try:
                with open(schema_file, encoding='utf-8') as file:
                    existing_schema = json.load(file)
            except json.JSONDecodeError:
                existing_schema = None
        else:
            existing_schema = None

        self.logger.info(f'Schema {schema_name} not found locally, saving generated dict...')
        if existing_schema != schema_dict:
            try:
                with open(schema_file, 'w', encoding='utf-8') as file:
                    json.dump(schema_dict, file, indent=4)
                self.schema_cache[schema_name] = schema_dict
                self.logger.info('Schema successfully written and cached.')
            except IOError as e:
                self.logger.error('Failed to write schema.')
                raise IOError(f'Failed to write schema to {schema_file}: {e}')
