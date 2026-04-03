"""Additional coverage tests for schema_registry.py.

Targets the paths not reached by the existing test_schema_registry.py:
  - get_schema() two-tier fallback logic (cache → registry → KeyError)
  - _get_schema_from_registry() caching and error paths
  - register_schema() and its sub-methods
  - _save_schema_to_registry() HTTP error / network failure
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from quantum_pipeline.utils.schema_registry import SchemaRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """Fresh SchemaRegistry instance."""
    return SchemaRegistry()


@pytest.fixture
def sample_schema():
    return {
        'type': 'record',
        'name': 'TestRecord',
        'fields': [
            {'name': 'id', 'type': 'int'},
            {'name': 'value', 'type': 'float'},
        ],
    }


# ---------------------------------------------------------------------------
# _get_schema_from_registry
# ---------------------------------------------------------------------------


class TestGetSchemaFromRegistry:
    """Coverage for _get_schema_from_registry."""

    def test_success_caches_schema_and_id(self, registry, sample_schema):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            'schema': json.dumps(sample_schema),
            'id': 42,
        }
        with patch('quantum_pipeline.utils.schema_registry.requests.get', return_value=resp):
            result = registry._get_schema_from_registry('test_schema')

        assert result is not None
        assert result['name'] == 'TestRecord'
        assert 'test_schema' in registry.schema_cache
        assert registry.id_cache['test_schema'] == 42

    def test_non_200_returns_none(self, registry):
        resp = MagicMock()
        resp.status_code = 404
        with patch('quantum_pipeline.utils.schema_registry.requests.get', return_value=resp):
            assert registry._get_schema_from_registry('missing') is None

    def test_request_exception_returns_none(self, registry):
        import requests as _req

        with patch(
            'quantum_pipeline.utils.schema_registry.requests.get',
            side_effect=_req.RequestException('timeout'),
        ):
            assert registry._get_schema_from_registry('boom') is None

    def test_id_cache_not_overwritten(self, registry, sample_schema):
        """If id_cache already has an entry it should not be overwritten."""
        registry.id_cache['test_schema'] = 99
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            'schema': json.dumps(sample_schema),
            'id': 42,
        }
        with patch('quantum_pipeline.utils.schema_registry.requests.get', return_value=resp):
            registry._get_schema_from_registry('test_schema')
        assert registry.id_cache['test_schema'] == 99


# ---------------------------------------------------------------------------
# get_schema (orchestrator)
# ---------------------------------------------------------------------------


class TestGetSchema:
    """Coverage for the main get_schema() method."""

    def test_returns_from_cache(self, registry, sample_schema):
        """If schema is in cache, should return it without network calls."""
        registry.schema_cache['cached'] = sample_schema
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_get.return_value = MagicMock(status_code=500)  # registry unavailable
            result = registry.get_schema('cached')
        assert result['name'] == 'TestRecord'

    def test_raises_key_error_when_not_found(self, registry):
        """When schema is absent from cache and registry, raise KeyError."""
        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            with pytest.raises(KeyError):
                registry.get_schema('totally_missing')

    def test_returns_from_registry_when_not_cached(self, registry, sample_schema):
        """Schema absent from cache but present in registry should be returned."""
        avail_resp = MagicMock(status_code=200)
        registry_resp = MagicMock(status_code=200)
        registry_resp.json.return_value = {
            'schema': json.dumps(sample_schema),
            'id': 7,
        }

        with patch('quantum_pipeline.utils.schema_registry.requests.get') as mock_get:
            mock_get.side_effect = [avail_resp, registry_resp]
            result = registry.get_schema('from_registry')

        assert result['name'] == 'TestRecord'


# ---------------------------------------------------------------------------
# _save_schema_to_registry
# ---------------------------------------------------------------------------


class TestSaveSchemaToRegistry:
    """Coverage for _save_schema_to_registry."""

    def test_success_returns_true(self, registry, sample_schema):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {'id': 1}
        with patch('quantum_pipeline.utils.schema_registry.requests.post', return_value=resp):
            assert registry._save_schema_to_registry('s', sample_schema) is True
        assert registry.id_cache['s'] == 1

    @pytest.mark.parametrize('status', [400, 409, 500])
    def test_non_success_status_returns_false(self, registry, sample_schema, status):
        resp = MagicMock(status_code=status, text='error')
        with patch('quantum_pipeline.utils.schema_registry.requests.post', return_value=resp):
            assert registry._save_schema_to_registry('s', sample_schema) is False

    def test_request_exception_returns_false(self, registry, sample_schema):
        import requests as _req

        with patch(
            'quantum_pipeline.utils.schema_registry.requests.post',
            side_effect=_req.RequestException('network'),
        ):
            assert registry._save_schema_to_registry('s', sample_schema) is False


# ---------------------------------------------------------------------------
# register_schema (public)
# ---------------------------------------------------------------------------


class TestRegisterSchema:
    """Coverage for the public register_schema() method."""

    def test_invalid_avro_raises(self, registry):
        with pytest.raises(ValueError):
            registry.register_schema('bad', {'not': 'avro'})

    def test_valid_schema_is_cached_and_published(self, registry, sample_schema):
        """register_schema should cache the schema and post it to the registry."""
        post_resp = MagicMock(status_code=200)
        post_resp.json.return_value = {'id': 5}
        with patch('quantum_pipeline.utils.schema_registry.requests.post', return_value=post_resp):
            registry.register_schema('my_schema', sample_schema)

        assert 'my_schema' in registry.schema_cache
        assert registry.schema_cache['my_schema']['name'] == 'TestRecord'

    def test_registry_failure_does_not_raise(self, registry, sample_schema):
        """A failed registry POST should not raise — schema is still cached."""
        with patch(
            'quantum_pipeline.utils.schema_registry.requests.post',
            return_value=MagicMock(status_code=500, text='err'),
        ):
            registry.register_schema('cached_only', sample_schema)

        assert 'cached_only' in registry.schema_cache


# ---------------------------------------------------------------------------
# _validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_schema_passes(self, registry, sample_schema):
        registry._validate_schema('ok', sample_schema)  # no error

    def test_invalid_schema_raises(self, registry):
        with pytest.raises(ValueError):
            registry._validate_schema('bad', {'not': 'valid avro'})
