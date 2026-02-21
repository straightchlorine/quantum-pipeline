"""Tests for Schema Registry utilities."""

import json
import pytest
from unittest.mock import MagicMock, patch
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
        assert schema_registry.schema_cache == {}
        assert schema_registry.id_cache == {}
        assert schema_registry.registry_schema_existence == {}
        assert schema_registry.schema_dir is not None
        assert schema_registry.schema_registry_url is not None

    def test_logger_creation(self, schema_registry):
        """Test that logger is created."""
        assert schema_registry.logger is not None


class TestSchemaNormalization:
    """Test schema normalization functionality."""

    def test_normalize_dict_schema(self, schema_registry, sample_schema):
        """Test normalizing dictionary schema."""
        normalized = schema_registry._normalize_schema(sample_schema)
        assert isinstance(normalized, dict)
        assert 'type' in normalized
        assert normalized['type'] == 'record'

    def test_normalize_json_string_schema(self, schema_registry, sample_schema_json):
        """Test normalizing JSON string schema."""
        normalized = schema_registry._normalize_schema(sample_schema_json)
        assert isinstance(normalized, dict)
        assert 'type' in normalized

    def test_normalize_preserves_content(self, schema_registry, sample_schema):
        """Test that normalization preserves schema content."""
        normalized = schema_registry._normalize_schema(sample_schema)
        assert normalized['name'] == sample_schema['name']
        assert len(normalized['fields']) == len(sample_schema['fields'])

    def test_normalize_invalid_schema_raises_error(self, schema_registry):
        """Test that invalid schema raises ValueError."""
        with pytest.raises(ValueError):
            schema_registry._normalize_schema(123)

    def test_normalize_invalid_json_raises_error(self, schema_registry):
        """Test that invalid JSON string raises error."""
        with pytest.raises(json.JSONDecodeError):
            schema_registry._normalize_schema('not valid json {')

    def test_normalize_empty_dict(self, schema_registry):
        """Test normalizing empty dictionary."""
        normalized = schema_registry._normalize_schema({})
        assert normalized == {}

    def test_normalize_complex_schema(self, schema_registry):
        """Test normalizing complex schema with nested fields."""
        complex_schema = {
            'type': 'record',
            'name': 'Complex',
            'fields': [
                {'name': 'id', 'type': 'int'},
                {
                    'name': 'nested',
                    'type': {
                        'type': 'record',
                        'name': 'Nested',
                        'fields': [
                            {'name': 'field1', 'type': 'string'},
                        ],
                    },
                },
            ],
        }
        normalized = schema_registry._normalize_schema(complex_schema)
        assert 'nested' in normalized['fields'][1]['name']


class TestRegistryAvailability:
    """Test schema registry availability checking."""

    def test_registry_available(self, schema_registry):
        """Test when registry is available."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert schema_registry.is_schema_registry_available() is True

    def test_registry_unavailable(self, schema_registry):
        """Test when registry is unavailable."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            assert schema_registry.is_schema_registry_available() is False

    def test_registry_connection_error(self, schema_registry):
        """Test handling of connection errors."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            import requests
            mock_get.side_effect = requests.RequestException('Connection refused')

            assert schema_registry.is_schema_registry_available() is False

    def test_registry_timeout(self, schema_registry):
        """Test handling of timeout."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            import requests
            mock_get.side_effect = requests.Timeout('Request timeout')

            assert schema_registry.is_schema_registry_available() is False

    def test_registry_check_with_correct_url(self, schema_registry):
        """Test that registry availability check uses correct URL."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            schema_registry.is_schema_registry_available()
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert '/subjects' in call_args[0][0]


class TestSchemaExistenceCheck:
    """Test checking if schema exists in registry."""

    def test_schema_exists_in_registry(self, schema_registry):
        """Test when schema exists in registry."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            # Mock registry availability
            availability_response = MagicMock()
            availability_response.status_code = 200

            # Mock schema existence check
            schema_response = MagicMock()
            schema_response.status_code = 200

            mock_get.side_effect = [availability_response, schema_response]

            assert schema_registry.is_schema_in_registry('test-schema') is True

    def test_schema_not_exists_in_registry(self, schema_registry):
        """Test when schema doesn't exist in registry."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            # Mock registry availability
            availability_response = MagicMock()
            availability_response.status_code = 200

            # Mock schema not found
            schema_response = MagicMock()
            schema_response.status_code = 404

            mock_get.side_effect = [availability_response, schema_response]

            assert schema_registry.is_schema_in_registry('nonexistent-schema') is False

    def test_schema_existence_caching(self, schema_registry):
        """Test that schema existence is cached."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            # Mock responses
            availability_response = MagicMock()
            availability_response.status_code = 200
            schema_response = MagicMock()
            schema_response.status_code = 200

            mock_get.side_effect = [availability_response, schema_response]

            # First call
            result1 = schema_registry.is_schema_in_registry('test-schema')
            # Second call (should use cache)
            result2 = schema_registry.is_schema_in_registry('test-schema')

            assert result1 is True
            assert result2 is True
            # Should only call requests.get twice (availability + first existence check)
            assert mock_get.call_count == 2

    def test_registry_unavailable_cached(self, schema_registry):
        """Test that unavailable registry status is cached."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            # Mock unavailable registry
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            # First call
            result1 = schema_registry.is_schema_in_registry('test-schema')
            # Second call
            result2 = schema_registry.is_schema_in_registry('test-schema')

            assert result1 is False
            assert result2 is False

    def test_different_schemas_different_cache_entries(self, schema_registry):
        """Test that different schemas have separate cache entries."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            # Mock responses
            availability_response = MagicMock()
            availability_response.status_code = 200

            schema1_response = MagicMock()
            schema1_response.status_code = 200

            schema2_response = MagicMock()
            schema2_response.status_code = 404

            mock_get.side_effect = [
                availability_response,
                schema1_response,
                availability_response,
                schema2_response,
            ]

            result1 = schema_registry.is_schema_in_registry('schema-a')
            result2 = schema_registry.is_schema_in_registry('schema-b')

            assert result1 is True
            assert result2 is False
            assert len(schema_registry.registry_schema_existence) == 2


class TestErrorHandling:
    """Test error handling in schema registry."""

    def test_json_decode_error_handling(self, schema_registry):
        """Test handling of JSON decode errors."""
        with pytest.raises(json.JSONDecodeError):
            schema_registry._normalize_schema('{invalid json}')

    def test_request_exception_handling(self, schema_registry):
        """Test handling of request exceptions."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            import requests
            mock_get.side_effect = requests.RequestException('Network error')

            # Should not raise, should return False
            result = schema_registry.is_schema_registry_available()
            assert result is False

    def test_http_error_status_codes(self, schema_registry):
        """Test handling of various HTTP error status codes."""
        for status_code in [400, 401, 403, 404, 500, 502, 503]:
            with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
                availability_response = MagicMock()
                availability_response.status_code = 200
                error_response = MagicMock()
                error_response.status_code = status_code

                mock_get.side_effect = [availability_response, error_response]

                result = schema_registry.is_schema_in_registry('test-schema')
                assert result is (status_code == 200)
