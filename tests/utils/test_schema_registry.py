"""Tests for Schema Registry utilities."""

import json
from unittest.mock import MagicMock, patch

import pytest
from avro.errors import SchemaParseException

from quantum_pipeline.utils.schema_registry import SchemaRegistry


@pytest.fixture
def schema_registry():
    """Create a SchemaRegistry instance for testing."""
    return SchemaRegistry()


@pytest.fixture
def sample_schema():
    """Sample Avro schema for testing."""
    return {
        'type': 'record',
        'name': 'TestRecord',
        'fields': [
            {'name': 'id', 'type': 'int'},
            {'name': 'name', 'type': 'string'},
            {'name': 'value', 'type': 'float'},
        ],
    }


@pytest.fixture
def sample_schema_json(sample_schema):
    """Sample schema as JSON string."""
    return json.dumps(sample_schema)


class TestSchemaRegistryInitialization:
    """Test SchemaRegistry initialization."""

    def test_registry_initialization(self, schema_registry):
        """Test that SchemaRegistry initializes correctly."""
        assert schema_registry is not None
        assert schema_registry.cache == {}
        assert schema_registry.url is not None

    def test_logger_creation(self, schema_registry):
        """Test that logger is created."""
        assert schema_registry.logger is not None


class TestSerializeSchema:
    """Test parsing schemas into Avro schema objects."""

    def test_serialize_dict_schema(self, schema_registry, sample_schema):
        """Test parsing a dict schema."""
        import avro.schema

        parsed = schema_registry.serialize_schema(sample_schema)
        assert isinstance(parsed, avro.schema.Schema)

    def test_serialize_json_string_schema(self, schema_registry, sample_schema_json):
        """Test parsing a JSON string schema."""
        import avro.schema

        parsed = schema_registry.serialize_schema(sample_schema_json)
        assert isinstance(parsed, avro.schema.Schema)

    def test_serialize_invalid_type_raises_error(self, schema_registry):
        """Test that an unsupported schema type raises TypeError."""
        with pytest.raises(TypeError):
            schema_registry.serialize_schema(123)  # type: ignore[arg-type]

    def test_serialize_invalid_json_raises_error(self, schema_registry):
        """Test that an invalid JSON string raises an error."""
        with pytest.raises(SchemaParseException):
            schema_registry.serialize_schema('not valid json {')


class TestUpstreamAvailability:
    """Test schema registry upstream availability checking."""

    def test_upstream_available(self, schema_registry):
        """Test when the registry is available."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert schema_registry.is_upstream_up() is True

    def test_upstream_unavailable(self, schema_registry):
        """Test when the registry is unavailable."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            assert schema_registry.is_upstream_up() is False

    def test_upstream_connection_error(self, schema_registry):
        """Test handling of connection errors."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            import requests

            mock_get.side_effect = requests.RequestException('Connection refused')

            assert schema_registry.is_upstream_up() is False

    def test_upstream_timeout(self, schema_registry):
        """Test handling of timeout."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            import requests

            mock_get.side_effect = requests.Timeout('Request timeout')

            assert schema_registry.is_upstream_up() is False

    def test_upstream_check_uses_correct_url(self, schema_registry):
        """Test that the availability check hits the /subjects endpoint."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            schema_registry.is_upstream_up()
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert '/subjects' in call_args[0][0]


class TestGetSchema:
    """Test fetching schemas through the cache/registry."""

    def test_get_schema_from_cache(self, schema_registry, sample_schema_json):
        """Test that a cached schema is returned without hitting the registry."""
        from quantum_pipeline.utils.schema_registry import SchemaRecord

        schema_registry.cache['test-schema'] = SchemaRecord(id=1, schema=sample_schema_json)

        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            result = schema_registry.get_schema('test-schema')
            assert result == sample_schema_json
            mock_get.assert_not_called()

    def test_get_schema_from_upstream(self, schema_registry, sample_schema, sample_schema_json):
        """Test fetching an uncached schema from the registry."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            availability_response = MagicMock(status_code=200)
            fetch_response = MagicMock(status_code=200)
            fetch_response.json.return_value = {'id': 7, 'schema': sample_schema_json}
            mock_get.side_effect = [availability_response, fetch_response]

            result = schema_registry.get_schema('test-schema')
            assert json.loads(result) == sample_schema
            assert schema_registry.cache['test-schema'].id == 7

    def test_get_schema_registry_down_raises(self, schema_registry):
        """Test that an unreachable registry raises ConnectionError."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_get.return_value = MagicMock(status_code=500)

            with pytest.raises(ConnectionError):
                schema_registry.get_schema('test-schema')

    def test_get_schema_not_found_raises(self, schema_registry):
        """Test that a schema missing from cache and registry raises KeyError."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            availability_response = MagicMock(status_code=200)
            fetch_response = MagicMock(status_code=404)
            mock_get.side_effect = [availability_response, fetch_response]

            with pytest.raises(KeyError):
                schema_registry.get_schema('missing-schema')


class TestRegisterSchema:
    """Test validating, caching and publishing schemas."""

    def test_register_schema_caches_it(self, schema_registry, sample_schema):
        """Test that registering a schema caches it, keyed by name."""
        with patch.object(schema_registry, 'publish_schema') as mock_publish:
            schema_registry.register_schema('test-schema', sample_schema)

            assert 'test-schema' in schema_registry.cache
            assert schema_registry.cache['test-schema'].id is None
            mock_publish.assert_called_once_with('test-schema')

    def test_register_invalid_schema_raises(self, schema_registry):
        """Test that an invalid Avro schema raises ValueError."""
        with pytest.raises(ValueError):
            schema_registry.register_schema('bad-schema', {'type': 'not-a-real-type'})


class TestPublishSchema:
    """Test publishing cached schemas to the registry."""

    def test_publish_schema_not_cached_returns_false(self, schema_registry):
        """Test that publishing an uncached schema fails cleanly."""
        assert schema_registry.publish_schema('missing-schema') is False

    def test_publish_schema_registry_down_returns_false(self, schema_registry, sample_schema):
        """Test that publishing while the registry is down fails cleanly."""
        from quantum_pipeline.utils.schema_registry import SchemaRecord

        schema_registry.cache['test-schema'] = SchemaRecord(id=None, schema=json.dumps(sample_schema))

        with patch.object(schema_registry, 'is_upstream_up', return_value=False):
            assert schema_registry.publish_schema('test-schema') is False

    def test_publish_schema_success(self, schema_registry, sample_schema):
        """Test a successful publish sets the schema id from the response."""
        from quantum_pipeline.utils.schema_registry import SchemaRecord

        schema_registry.cache['test-schema'] = SchemaRecord(id=None, schema=json.dumps(sample_schema))

        with (
            patch.object(schema_registry, 'is_upstream_up', return_value=True),
            patch('quantum_pipeline.utils.schema_registry.requests.post') as mock_post,
        ):
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.json.return_value = {'id': 42}

            assert schema_registry.publish_schema('test-schema') is True
            assert schema_registry.cache['test-schema'].id == 42

    def test_publish_schema_rejected_returns_false(self, schema_registry, sample_schema):
        """Test that a non-2xx response from the registry fails cleanly."""
        from quantum_pipeline.utils.schema_registry import SchemaRecord

        schema_registry.cache['test-schema'] = SchemaRecord(id=None, schema=json.dumps(sample_schema))

        with (
            patch.object(schema_registry, 'is_upstream_up', return_value=True),
            patch('quantum_pipeline.utils.schema_registry.requests.post') as mock_post,
        ):
            mock_post.return_value = MagicMock(status_code=422, text='invalid schema')

            assert schema_registry.publish_schema('test-schema') is False
