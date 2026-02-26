import json
from pathlib import Path
from typing import Any

import avro.schema
import requests

from quantum_pipeline.configs.settings import SCHEMA_DIR, SCHEMA_REGISTRY_URL
from quantum_pipeline.utils.logger import get_logger


class SchemaRegistry:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        self.schema_cache: dict[str, dict[str, Any]] = {}
        self.id_cache: dict[str, int] = {}

        self.schema_dir = SCHEMA_DIR
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
            response = requests.get(url)

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
        """Get Avro schema by name, trying cache, registry, and local file in sequence.

        Args:
            schema_name: Name of the schema without extension

        Returns:
            Dict containing the Avro schema

        Raises:
            FileNotFoundError: If schema file doesn't exist anywhere
            ValueError: If schema is invalid
        """
        schema = None
        from_cache = False

        # try to get the schema from the cache
        schema = self._get_schema_from_cache(schema_name)
        if schema:
            from_cache = True
            self.logger.debug(f'Retrieved schema {schema_name} from cache.')

        registry_available = self.is_schema_registry_available()
        if registry_available:
            self.logger.info('Schema registry is available.')
        else:
            self.logger.warning('Schema registry is not available.')

        # if unable to get from cache, try the schema registry service
        if not from_cache and registry_available:
            # is schema in the registry
            schema_in_registry = (
                schema_name in self.registry_schema_existence
                and self.registry_schema_existence[schema_name]
            )

            # if found in the registry or if the id_cache is empty, get it from the registry
            if schema_in_registry or not self.id_cache.get(schema_name, False):
                schema = self._get_schema_from_registry(schema_name)
                if schema:
                    self.logger.debug(f'Retrieved schema {schema_name} from registry.')
                    return self._normalize_schema(schema)
                # update tracking if failed to get it
                self.registry_schema_existence[schema_name] = False

        # if unable to get from cache and unable to get form the registry, try
        # the local file
        if not schema:
            try:
                schema = self._get_schema_from_file(schema_name)
                self.logger.debug(f'Retrieved schema {schema_name} from local file.')
            except FileNotFoundError:
                # reraise if we couldn't find the schema anywhere
                raise FileNotFoundError(
                    f'Schema {schema_name} not found in cache, registry, or local filesystem.'
                )

        # here schema is available, check if we need to publish it to the registry
        if registry_available:
            schema_in_registry = False

            # check schema's availability in cache and the registry
            if schema_name not in self.registry_schema_existence:
                schema_in_registry = self.is_schema_in_registry(schema_name)
            else:
                schema_in_registry = self.registry_schema_existence[schema_name]

            # if not available in the registry, try to publish it
            if not schema_in_registry:
                self.logger.info(
                    f"Schema {schema_name} found "
                    f"{'in cache' if from_cache else 'locally'} "
                    "but not in registry. Synchronizing..."
                )
                success = self._save_schema_to_registry(schema_name, schema)
                if success:
                    # update the cache
                    self.logger.info(f'Schema {schema_name} synchronized to registry.')
                    self.registry_schema_existence[schema_name] = True

        return self._normalize_schema(schema)

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

    def _get_schema_from_file(self, schema_name: str) -> dict[str, Any]:
        """Get schema from local file system.

        Raises:
            FileNotFoundError: If schema file doesn't exist
            ValueError: If schema is invalid
        """
        self.logger.debug(f'Checking schema directory for {schema_name}...')
        schema_file = self.schema_dir / f'{schema_name}.avsc'

        if not schema_file.exists():
            self.logger.error(f'Unable to find {schema_name} locally.')
            raise FileNotFoundError(f'Schema file not found: {schema_file}')

        self.logger.info(f'Found schema {schema_name} locally.')

        try:
            with open(schema_file) as f:
                schema_dict = json.load(f)

            self.logger.debug(f'Validating {schema_name} schema...')
            avro.schema.parse(json.dumps(schema_dict))

            self.logger.debug(f'Validation passed, caching {schema_name} schema.')

            schema_dict = self._normalize_schema(schema_dict)
            self.schema_cache[schema_name] = schema_dict
            return schema_dict

        except json.JSONDecodeError as e:
            self.logger.error(f'Validation of {schema_name} schema failed: Invalid JSON')
            raise ValueError(f'Invalid JSON in schema file {schema_file}: {e!s}')
        except Exception as e:
            self.logger.error(f'Error loading {schema_name} schema: {e}')
            raise ValueError(f'Invalid Avro schema in {schema_file}: {e!s}')

    def save_schema(self, schema_name: str, schema_dict: dict[str, Any]) -> None:
        """Save the given schema to registry and file system if different from existing.

        Args:
            schema_name: Name of the schema (without extension).
            schema_dict: The Avro schema dictionary to save.

        Raises:
            ValueError: If the provided schema is invalid.
            IOError: If there is an issue writing to the file.
        """
        self._validate_schema(schema_name, schema_dict)

        self._ensure_schema_directory_exists()

        self._save_schema_to_registry(schema_name, schema_dict)

        self._save_schema_to_file_if_changed(schema_name, schema_dict)

    def _validate_schema(self, schema_name: str, schema_dict: dict[str, Any]) -> None:
        """Validate the Avro schema format."""
        self.logger.info('Validating the schema dict...')
        self.logger.debug(f'{schema_name} structure:\n\n{schema_dict}\n\n')
        try:
            avro.schema.parse(json.dumps(schema_dict))
        except Exception as e:
            self.logger.error('Invalid Avro schema.')
            raise ValueError(f'Invalid Avro schema: {e}')

    def _ensure_schema_directory_exists(self) -> None:
        """Create schema directory if it doesn't exist."""
        try:
            self.schema_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f'Failed to create schema directory: {e}')
            raise OSError(f'Cannot create schema directory {self.schema_dir}: {e}')

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

    def _save_schema_to_file_if_changed(
        self, schema_name: str, schema_dict: dict[str, Any]
    ) -> None:
        """Save schema to file if it differs from existing version."""
        schema_file = self.schema_dir / f'{schema_name}.avsc'

        # check if exists and read
        existing_schema = self._read_existing_schema(schema_file)

        # write only if differs
        if existing_schema != schema_dict:
            self.logger.info(
                f'Saving schema {schema_name} as it differs from existing or is new...'
            )
            try:
                with open(schema_file, 'w', encoding='utf-8') as file:
                    json.dump(schema_dict, file, indent=4)
                # Update cache
                self.schema_cache[schema_name] = schema_dict
                self.logger.info('Schema successfully written and cached.')
            except OSError as e:
                self.logger.error('Failed to write schema.')
                raise OSError(f'Failed to write schema to {schema_file}: {e}')
        else:
            self.logger.info(f'Schema {schema_name} unchanged, no need to save.')

    def _read_existing_schema(self, schema_file: Path) -> dict[str, Any] | None:
        """Read existing schema from file if it exists."""
        self.logger.info(f'Checking if schema exists at {schema_file}...')
        if schema_file.exists():
            try:
                with open(schema_file, encoding='utf-8') as file:
                    return json.load(file)
            except json.JSONDecodeError as e:
                self.logger.warning(f'Existing schema file contains invalid JSON: {e}')
                return None
        else:
            self.logger.info('Schema file does not exist yet.')
            return None
