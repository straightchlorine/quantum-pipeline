import json
from typing import Any

import avro.schema
import requests

from quantum_pipeline.configs.settings import SCHEMA_REGISTRY_URL
from quantum_pipeline.utils.logger import get_logger


class SchemaRegistry:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        self.schema_cache: dict[str, dict[str, Any]] = {}
        self.id_cache: dict[str, int] = {}

        self.schema_registry_url = SCHEMA_REGISTRY_URL

        # dictionary, with schema_name:bool pairs
        self.registry_schema_existence = {}

    def _normalize_schema(self, schema: dict[str, Any] | str) -> dict[str, Any]:
        """Normalize schema to ensure consistent formatting."""
        if isinstance(schema, dict):
            schema_str = json.dumps(schema)
            return json.loads(schema_str)
        if isinstance(schema, str):
            return json.loads(schema)
        raise ValueError('Invalid schema format.')

    def is_schema_registry_available(self) -> bool:
        """Check if the schema registry is accessible."""
        test_url = f'{self.schema_registry_url}/subjects'
        try:
            response = requests.get(test_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def is_schema_in_registry(self, schema_name: str) -> bool:
        """Check if a schema exists in the registry.

        Determine if a schema exists in the registry by:
        1. Checking the existence cache first
        2. Checking registry connectivity
        3. Making an API call only when necessary

        Args:
            schema_name: Name of the schema without extension

        Returns:
            bool: True if schema exists in registry, False otherwise
        """
        # check the cache
        if schema_name in self.registry_schema_existence:
            self.logger.debug(f'Using cached registry status for {schema_name}.')
            return self.registry_schema_existence[schema_name]

        # check if registry is available
        self.logger.info('Checking the availability of the schema registry')
        if not self.is_schema_registry_available():
            self.logger.warning('Schema registry is not available.')
            self.registry_schema_existence[schema_name] = False
            return False

        # check if schema already exists in the registry
        try:
            url = f'{self.schema_registry_url}/subjects/{schema_name}-value/versions'
            response = requests.get(url, timeout=5)

            exists = response.status_code == 200

            # cache the result
            self.registry_schema_existence[schema_name] = exists

            if exists:
                self.logger.info(f'Schema {schema_name} exists in registry.')
            else:
                self.logger.info(f'Schema {schema_name} does not exist in registry.')

            return exists

        except requests.RequestException as e:
            self.logger.warning(f'Error checking schema existence in registry:\n\n{e}\n')
            self.registry_schema_existence[schema_name] = False
            return False

    def get_schema(self, schema_name: str) -> dict[str, Any]:
        """Get Avro schema by name, trying memory cache then Schema Registry.

        Args:
            schema_name: Name of the schema without extension

        Returns:
            Dict containing the Avro schema

        Raises:
            KeyError: If schema is not found in cache or registry
        """
        # try to get the schema from the cache
        schema = self._get_schema_from_cache(schema_name)
        if schema:
            self.logger.debug(f'Retrieved schema {schema_name} from cache.')
            return self._normalize_schema(schema)

        # try the schema registry
        if self.is_schema_registry_available():
            schema = self._get_schema_from_registry(schema_name)
            if schema:
                self.logger.debug(f'Retrieved schema {schema_name} from registry.')
                return self._normalize_schema(schema)

        raise KeyError(f'Schema {schema_name} not found in cache or registry.')

    def _get_schema_from_cache(self, schema_name: str) -> dict[str, Any] | None:
        """Get schema from in-memory cache if available."""
        self.logger.debug(f'Checking if {schema_name} schema exists in cache...')
        if schema_name in self.schema_cache:
            self.logger.info(f'Found cached {schema_name} schema.')
            return self._normalize_schema(self.schema_cache[schema_name])
        self.logger.info(f'No entry of {schema_name} in the cache.')
        return None

    def _get_schema_from_registry(self, schema_name: str) -> dict[str, Any] | None:
        """Get schema from remote schema registry if available."""
        self.logger.debug(f'Checking schema registry at {self.schema_registry_url}...')
        try:
            response = requests.get(
                f'{self.schema_registry_url}/subjects/{schema_name}-value/versions/latest',
                timeout=5,
            )
            if response.status_code == 200:
                self.logger.debug('Found schema in the schema-registry service.')

                response_json = response.json()
                schema = response_json['schema']
                schema_id = response_json['id']

                # Cache the schema and its ID
                self.schema_cache[schema_name] = schema
                if not self.id_cache.get(schema_name, False):
                    self.id_cache[schema_name] = schema_id

                return self._normalize_schema(schema)
            self.logger.warning('Unable to find schema at the schema registry.')
        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch schema from registry: {e}')

        return None

    def register_schema(self, schema_name: str, schema_dict: dict[str, Any]) -> None:
        """Validate, cache, and publish the given schema to the Schema Registry.

        Args:
            schema_name: Name of the schema (without extension).
            schema_dict: The Avro schema dictionary to register.

        Raises:
            ValueError: If the provided schema is invalid.
        """
        self._validate_schema(schema_name, schema_dict)

        normalized = self._normalize_schema(schema_dict)
        self.schema_cache[schema_name] = normalized

        self._save_schema_to_registry(schema_name, normalized)

    def _validate_schema(self, schema_name: str, schema_dict: dict[str, Any]) -> None:
        """Validate the Avro schema format."""
        self.logger.info('Validating the schema dict...')
        self.logger.debug(f'{schema_name} structure:\n\n{schema_dict}\n\n')
        try:
            avro.schema.parse(json.dumps(schema_dict))
        except Exception as e:
            self.logger.error('Invalid Avro schema.')
            raise ValueError(f'Invalid Avro schema: {e}') from e

    def _save_schema_to_registry(self, schema_name: str, schema_dict: dict[str, Any]) -> bool:
        """Attempt to save schema to registry.

        Returns:
            bool: True if successfully saved to registry, False otherwise
        """
        self.logger.debug('Attempting to save schema at schema registry...')
        try:
            response = requests.post(
                f'{self.schema_registry_url}/subjects/{schema_name}-value/versions',
                headers={'Content-Type': 'application/vnd.schemaregistry.v1+json'},
                json={'schema': json.dumps(schema_dict)},
                timeout=5,
            )

            if response.status_code not in [200, 201]:
                self.logger.warning(f'Failed to register schema: {response.text}')
                return False

            self.logger.info('Schema registered successfully.')
            if not self.id_cache.get(schema_name, False):
                self.id_cache[schema_name] = response.json()['id']

            return True

        except requests.RequestException as e:
            self.logger.warning(f'Error registering schema in registry: {e}')
            return False
