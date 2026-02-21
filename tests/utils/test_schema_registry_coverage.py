"""Additional coverage tests for schema_registry.py.

Targets the paths not reached by the existing test_schema_registry.py:
  - get_schema() multi-source fallback logic
  - _get_schema_from_registry() caching and error paths
  - _get_schema_from_file() validation and error paths
  - save_schema() and its sub-methods
  - _read_existing_schema() edge cases
  - _save_schema_to_registry() HTTP error / network failure
  - _save_schema_to_file_if_changed() write and skip paths
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from quantum_pipeline.utils.schema_registry import SchemaRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    """Fresh SchemaRegistry with mocked-out directory settings."""
    with patch(
        "quantum_pipeline.utils.schema_registry.SCHEMA_DIR",
        Path("/tmp/fake_schemas"),
    ):
        r = SchemaRegistry()
        r.schema_dir = Path("/tmp/fake_schemas")
        yield r


@pytest.fixture
def sample_schema():
    return {
        "type": "record",
        "name": "TestRecord",
        "fields": [
            {"name": "id", "type": "int"},
            {"name": "value", "type": "float"},
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
            "schema": json.dumps(sample_schema),
            "id": 42,
        }
        with patch("quantum_pipeline.utils.schema_registry.requests.get", return_value=resp):
            result = registry._get_schema_from_registry("test_schema")

        assert result is not None
        assert result["name"] == "TestRecord"
        assert "test_schema" in registry.schema_cache
        assert registry.id_cache["test_schema"] == 42

    def test_non_200_returns_none(self, registry):
        resp = MagicMock()
        resp.status_code = 404
        with patch("quantum_pipeline.utils.schema_registry.requests.get", return_value=resp):
            assert registry._get_schema_from_registry("missing") is None

    def test_request_exception_returns_none(self, registry):
        import requests as _req
        with patch(
            "quantum_pipeline.utils.schema_registry.requests.get",
            side_effect=_req.RequestException("timeout"),
        ):
            assert registry._get_schema_from_registry("boom") is None

    def test_id_cache_not_overwritten(self, registry, sample_schema):
        """If id_cache already has an entry it should not be overwritten."""
        registry.id_cache["test_schema"] = 99
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "schema": json.dumps(sample_schema),
            "id": 42,
        }
        with patch("quantum_pipeline.utils.schema_registry.requests.get", return_value=resp):
            registry._get_schema_from_registry("test_schema")
        assert registry.id_cache["test_schema"] == 99


# ---------------------------------------------------------------------------
# _get_schema_from_file
# ---------------------------------------------------------------------------

class TestGetSchemaFromFile:
    """Coverage for _get_schema_from_file."""

    def test_file_not_found_raises(self, registry):
        with pytest.raises(FileNotFoundError):
            registry._get_schema_from_file("nonexistent")

    def test_valid_file_caches_schema(self, registry, sample_schema, tmp_path):
        registry.schema_dir = tmp_path
        schema_file = tmp_path / "good_schema.avsc"
        schema_file.write_text(json.dumps(sample_schema))

        result = registry._get_schema_from_file("good_schema")
        assert result["name"] == "TestRecord"
        assert "good_schema" in registry.schema_cache

    def test_invalid_json_raises_value_error(self, registry, tmp_path):
        registry.schema_dir = tmp_path
        bad_file = tmp_path / "bad.avsc"
        bad_file.write_text("{not json!!}")

        with pytest.raises(ValueError):
            registry._get_schema_from_file("bad")

    def test_invalid_avro_raises_value_error(self, registry, tmp_path):
        """Valid JSON but not a valid Avro schema."""
        registry.schema_dir = tmp_path
        schema_file = tmp_path / "not_avro.avsc"
        schema_file.write_text(json.dumps({"foo": "bar"}))

        with pytest.raises(ValueError):
            registry._get_schema_from_file("not_avro")


# ---------------------------------------------------------------------------
# get_schema (orchestrator)
# ---------------------------------------------------------------------------

class TestGetSchema:
    """Coverage for the main get_schema() method."""

    def test_returns_from_cache(self, registry, sample_schema):
        """If schema is in cache, should return it without network calls."""
        registry.schema_cache["cached"] = sample_schema
        with patch("quantum_pipeline.utils.schema_registry.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)  # registry unavailable
            result = registry.get_schema("cached")
        assert result["name"] == "TestRecord"

    def test_falls_back_to_file_when_registry_down(self, registry, sample_schema, tmp_path):
        registry.schema_dir = tmp_path
        schema_file = tmp_path / "local.avsc"
        schema_file.write_text(json.dumps(sample_schema))

        with patch("quantum_pipeline.utils.schema_registry.requests.get") as mock_get:
            # registry unavailable
            mock_get.return_value = MagicMock(status_code=500)
            result = registry.get_schema("local")

        assert result["name"] == "TestRecord"

    def test_file_not_found_anywhere_raises(self, registry):
        with patch("quantum_pipeline.utils.schema_registry.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            with pytest.raises(FileNotFoundError):
                registry.get_schema("totally_missing")

    def test_syncs_local_schema_to_registry(self, registry, sample_schema, tmp_path):
        """If schema found locally but not in registry, it should be published."""
        registry.schema_dir = tmp_path
        schema_file = tmp_path / "sync_me.avsc"
        schema_file.write_text(json.dumps(sample_schema))

        avail_resp = MagicMock(status_code=200)
        # is_schema_in_registry check — 404 means not in registry
        not_found_resp = MagicMock(status_code=404)
        # _get_schema_from_registry returns a non-200 so it falls through to file
        registry_schema_resp = MagicMock(status_code=404)

        with (
            patch("quantum_pipeline.utils.schema_registry.requests.get") as mock_get,
            patch("quantum_pipeline.utils.schema_registry.requests.post") as mock_post,
        ):
            mock_get.side_effect = [
                avail_resp,            # is_schema_registry_available (in get_schema)
                registry_schema_resp,  # _get_schema_from_registry (GET versions/latest)
                avail_resp,            # is_schema_registry_available (in is_schema_in_registry)
                not_found_resp,        # schema versions check in is_schema_in_registry
            ]
            post_resp = MagicMock(status_code=200)
            post_resp.json.return_value = {"id": 10}
            mock_post.return_value = post_resp

            result = registry.get_schema("sync_me")
            assert result is not None
            mock_post.assert_called_once()
            assert registry.registry_schema_existence.get("sync_me") is True


# ---------------------------------------------------------------------------
# _save_schema_to_registry
# ---------------------------------------------------------------------------

class TestSaveSchemaToRegistry:
    """Coverage for _save_schema_to_registry."""

    def test_success_returns_true(self, registry, sample_schema):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"id": 1}
        with patch("quantum_pipeline.utils.schema_registry.requests.post", return_value=resp):
            assert registry._save_schema_to_registry("s", sample_schema) is True
        assert registry.id_cache["s"] == 1

    @pytest.mark.parametrize("status", [400, 409, 500])
    def test_non_success_status_returns_false(self, registry, sample_schema, status):
        resp = MagicMock(status_code=status, text="error")
        with patch("quantum_pipeline.utils.schema_registry.requests.post", return_value=resp):
            assert registry._save_schema_to_registry("s", sample_schema) is False

    def test_request_exception_returns_false(self, registry, sample_schema):
        import requests as _req
        with patch(
            "quantum_pipeline.utils.schema_registry.requests.post",
            side_effect=_req.RequestException("network"),
        ):
            assert registry._save_schema_to_registry("s", sample_schema) is False


# ---------------------------------------------------------------------------
# save_schema (public)
# ---------------------------------------------------------------------------

class TestSaveSchema:
    """Coverage for the public save_schema() method."""

    def test_invalid_avro_raises(self, registry):
        with pytest.raises(ValueError):
            registry.save_schema("bad", {"not": "avro"})

    def test_full_save_flow(self, registry, sample_schema, tmp_path):
        registry.schema_dir = tmp_path
        with patch(
            "quantum_pipeline.utils.schema_registry.requests.post",
            return_value=MagicMock(status_code=200, json=lambda: {"id": 5}),
        ):
            registry.save_schema("full_flow", sample_schema)
        saved_file = tmp_path / "full_flow.avsc"
        assert saved_file.exists()
        assert json.loads(saved_file.read_text())["name"] == "TestRecord"


# ---------------------------------------------------------------------------
# _save_schema_to_file_if_changed
# ---------------------------------------------------------------------------

class TestSaveSchemaToFileIfChanged:
    """Coverage for _save_schema_to_file_if_changed."""

    def test_writes_new_file(self, registry, sample_schema, tmp_path):
        registry.schema_dir = tmp_path
        registry._save_schema_to_file_if_changed("new_one", sample_schema)
        assert (tmp_path / "new_one.avsc").exists()
        assert "new_one" in registry.schema_cache

    def test_skips_write_when_unchanged(self, registry, sample_schema, tmp_path):
        registry.schema_dir = tmp_path
        schema_file = tmp_path / "same.avsc"
        schema_file.write_text(json.dumps(sample_schema, indent=4))

        registry._save_schema_to_file_if_changed("same", sample_schema)
        # Should not update cache since nothing changed
        assert "same" not in registry.schema_cache

    def test_write_failure_raises_os_error(self, registry, sample_schema, tmp_path):
        registry.schema_dir = tmp_path
        with patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                registry._save_schema_to_file_if_changed("fail", sample_schema)


# ---------------------------------------------------------------------------
# _read_existing_schema
# ---------------------------------------------------------------------------

class TestReadExistingSchema:
    """Coverage for _read_existing_schema."""

    def test_reads_valid_file(self, registry, sample_schema, tmp_path):
        f = tmp_path / "read_me.avsc"
        f.write_text(json.dumps(sample_schema))
        result = registry._read_existing_schema(f)
        assert result["name"] == "TestRecord"

    def test_returns_none_for_missing_file(self, registry, tmp_path):
        assert registry._read_existing_schema(tmp_path / "nope.avsc") is None

    def test_returns_none_for_invalid_json(self, registry, tmp_path):
        bad = tmp_path / "bad.avsc"
        bad.write_text("{broken json!!}")
        assert registry._read_existing_schema(bad) is None


# ---------------------------------------------------------------------------
# _ensure_schema_directory_exists
# ---------------------------------------------------------------------------

class TestEnsureSchemaDirectoryExists:

    def test_creates_directory(self, registry, tmp_path):
        registry.schema_dir = tmp_path / "new_dir" / "nested"
        registry._ensure_schema_directory_exists()
        assert registry.schema_dir.exists()

    def test_existing_directory_no_error(self, registry, tmp_path):
        registry.schema_dir = tmp_path
        registry._ensure_schema_directory_exists()  # should not raise


# ---------------------------------------------------------------------------
# _validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:

    def test_valid_schema_passes(self, registry, sample_schema):
        registry._validate_schema("ok", sample_schema)  # no error

    def test_invalid_schema_raises(self, registry):
        with pytest.raises(ValueError):
            registry._validate_schema("bad", {"not": "valid avro"})
