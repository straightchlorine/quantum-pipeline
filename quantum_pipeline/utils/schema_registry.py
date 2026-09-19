from dataclasses import dataclass
import json
from typing import Any

import avro.schema
import requests

from quantum_pipeline.configs.settings import SCHEMA_REGISTRY_URL
from quantum_pipeline.utils.logger import get_logger


@dataclass(slots=True)
class SchemaRecord:
    id: int | None
    schema: str


class SchemaRegistry:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.url = SCHEMA_REGISTRY_URL
        self.cache: dict[str, SchemaRecord] = {}

    def serialize_schema(self, schema: dict[str, Any] | str) -> avro.schema.Schema:
        """Parse a schema (dict or JSON string) into an Avro schema object."""
        if isinstance(schema, dict):
            parsed_schema = avro.schema.parse(json.dumps(schema))
            self.logger.debug(f'Parsed dict schema: {parsed_schema}')
            return parsed_schema

        if isinstance(schema, str):
            parsed_schema = avro.schema.parse(schema)
            self.logger.debug(f'Parsed string schema: {parsed_schema}')
            return parsed_schema

        raise TypeError(f'Unsupported schema type: {type(schema).__name__}')

    def is_upstream_up(self) -> bool:
        test_url = f'{self.url}/subjects'
        try:
            response = requests.get(test_url, timeout=5)
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False

    def get_schema(self, schema_name: str) -> str:
        """Get Avro schema by name. First try the cache then the Schema Registry.

        Args:
            schema_name: Name of the schema without extension

        Returns:
            Dict containing the Avro schema

        Raises:
            KeyError: If schema is not found in cache or registry
        """
        # try to get the schema from the cache
        schema = self.fetch_schema_from_cache(schema_name)
        if schema:
            return schema

        # try the schema registry
        if self.is_upstream_up():
            schema = self.fetch_schema_from_upstream(schema_name)
            if schema:
                return schema
        else:
            raise ConnectionError('Schema registry is not available!')

        raise KeyError(f'Schema "{schema_name}" not found in cache or registry.')

    def fetch_schema_from_cache(self, schema_name: str) -> str | None:
        """Fetch the schema from local cache."""
        self.logger.debug(f'Checking if "{schema_name}" schema exists in cache...')
        if schema_name in self.cache:
            self.logger.info(f'Found "{schema_name}" cached.')
            return self.cache[schema_name].schema
        self.logger.info(f'No entry of "{schema_name}" in the cache.')
        return None

    def fetch_schema_from_upstream(self, schema_name: str) -> str | None:
        """Fetch a schema from the upstream schema registry."""
        self.logger.debug(f'Checking the schema registry at {self.url}...')
        try:
            response = requests.get(
                f'{self.url}/subjects/{schema_name}-value/versions/latest',
                timeout=5,
            )
            if response.status_code == 200:
                self.logger.debug('Schema found in the registry!')

                response_json = response.json()
                schema = self.validate_schema(schema_name, response_json['schema'])

                # cache the schema and its id
                self.cache[schema_name] = SchemaRecord(id=response_json['id'], schema=schema)

                return schema
            self.logger.warning(f'Unable to find "{schema_name}" at the registry.')
        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch "{schema_name}" from registry: {e}')

        return None

    def register_schema(self, schema_name: str, schema_dict: dict[str, Any]) -> None:
        """Validate, cache, and publish the given schema to the Schema Registry.

        Args:
            schema_name: Name of the schema (without extension).
            schema_dict: The Avro schema dictionary to register.

        Raises:
            ValueError: If the provided schema is invalid.
        """
        # already published: skip re-validating and re-POSTing on every .schema access
        cached = self.cache.get(schema_name)
        if cached is not None and cached.id is not None:
            return

        schema = self.validate_schema(schema_name, schema_dict)
        self.cache[schema_name] = SchemaRecord(id=None, schema=schema)
        self.publish_schema(schema_name)

    def validate_schema(self, schema_name: str, schema_dict: dict[str, Any] | str) -> str:
        """Validate the schema and return it as a string."""
        self.logger.info(f'Validating the "{schema_name}" schema...')
        self.logger.debug(f'{schema_name} structure:\n\n{schema_dict}\n\n')

        try:
            # dict schemas need dumping to a string first; strings are already Avro JSON
            schema = json.dumps(schema_dict) if isinstance(schema_dict, dict) else schema_dict
            avro.schema.parse(schema)
        except Exception as e:
            self.logger.error('Invalid Avro schema.')
            raise ValueError(f'Invalid Avro schema: {e}') from e

        return schema

    def publish_schema(self, schema_name: str) -> bool:
        """Attempt to publish schema to the registry from cache.

        Assumes schema is already registered in the SchemaRegistry's cache.

        Adding to this method a default schema: dict[str, Any] | string param
        was considered for future use. This was declined - project doesn't need
        such method, as all created schemas are registered to cache either way.

        This approach is for now sufficient.

        Returns:
            bool: True if successfully saved to registry, False otherwise
        """

        self.logger.debug('Attempting to publish the schema to the registry...')

        if schema_name not in self.cache:
            self.logger.warning(f'Schema "{schema_name}" not found in cache.')
            return False

        if not self.is_upstream_up():
            self.logger.warning('Schema registry is not available!')
            return False

        try:
            response = requests.post(
                f'{self.url}/subjects/{schema_name}-value/versions',
                headers={'Content-Type': 'application/vnd.schemaregistry.v1+json'},
                json={'schema': self.cache[schema_name].schema},
                timeout=5,
            )

            if response.status_code not in [200, 201]:
                self.logger.warning(f'Failed to publish the schema: {response.text}')
                return False

            self.logger.info('Schema published successfully.')
            self.cache[schema_name].id = response.json()['id']

            return True

        except requests.RequestException as e:
            self.logger.warning(f'Error registering schema in registry: {e}')
            return False
