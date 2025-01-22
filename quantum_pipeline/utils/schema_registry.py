import json
from typing import Any, Dict

from avro import schema

from quantum_pipeline.configs.settings import SCHEMA_DIR


class SchemaRegistry:
    def __init__(self):
        self.schema_dir = SCHEMA_DIR
        self.schema_cache: Dict[str, Dict[str, Any]] = {}

    def get_schema(self, schema_name: str) -> Dict[str, Any]:
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
        # check cache
        if schema_name in self.schema_cache:
            return self.schema_cache[schema_name]

        # build the path to the file
        schema_file = self.schema_dir / f'{schema_name}.avsc'

        # check if file exists
        if not schema_file.exists():
            raise FileNotFoundError(f'Schema file not found: {schema_file}')

        # read the schema
        try:
            with open(schema_file, 'r') as f:
                schema_dict = json.load(f)

            # validate
            schema.parse(json.dumps(schema_dict))

            # cache the schema
            self.schema_cache[schema_name] = schema_dict
            return schema_dict

        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in schema file {schema_file}: {str(e)}')
        except Exception as e:
            raise ValueError(f'Invalid Avro schema in {schema_file}: {str(e)}')
