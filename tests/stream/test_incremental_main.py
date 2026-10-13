"""Tests for the control flow of quantum_incremental_processing.py (no Spark session needed)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_AIRFLOW_DIR = Path(__file__).parent.parent.parent / 'docker' / 'airflow'
sys.path.insert(0, str(_AIRFLOW_DIR))
sys.path.insert(0, str(_AIRFLOW_DIR / 'scripts'))

import quantum_incremental_processing as qip  # noqa: E402


def test_main_continues_past_a_topic_without_data(monkeypatch):
    spark = MagicMock()
    processed = []
    monkeypatch.setattr(qip, 'create_spark_session', lambda _name: spark)
    monkeypatch.setattr(qip, 'list_available_topics', lambda _spark, _path: ['a', 'b', 'c'])
    monkeypatch.setattr(
        qip,
        'check_for_new_data',
        lambda _spark, topic, _path: None if topic == 'a' else f'df-{topic}',
    )
    monkeypatch.setattr(
        qip,
        'process_experiments_incrementally',
        lambda _spark, df, topic: processed.append((df, topic)) or {},
    )

    qip.main()

    assert processed == [('df-b', 'b'), ('df-c', 'c')]
    spark.stop.assert_called_once()


def test_list_available_topics_propagates_filesystem_errors():
    spark = MagicMock()
    spark._jvm.org.apache.hadoop.fs.FileSystem.get.side_effect = OSError('bad credentials')

    with pytest.raises(OSError, match='bad credentials'):
        qip.list_available_topics(spark, 's3a://bucket/experiments/')


def test_list_available_topics_returns_empty_for_missing_path():
    spark = MagicMock()
    spark._jvm.org.apache.hadoop.fs.FileSystem.get.return_value.exists.return_value = False

    assert qip.list_available_topics(spark, 's3a://bucket/experiments/') == []


def test_append_merges_schema_through_the_v2_writer(monkeypatch):
    spark = MagicMock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.collect.return_value = [[7]]
    new_rows = MagicMock()
    new_rows.count.return_value = 3
    monkeypatch.setattr(qip, 'identify_new_records', lambda *_args: new_rows)

    frame = MagicMock()
    frame.isEmpty.return_value = False

    tag, count = qip.process_incremental_data(spark, frame, 'molecules', ['experiment_id'], 'a-b')

    assert (tag, count) == ('v_incr_ab', 3)
    new_rows.writeTo.assert_called_once_with(f'{qip.CATALOG_FQN}.molecules')
    new_rows.writeTo.return_value.option.assert_called_once_with('mergeSchema', 'true')
    new_rows.writeTo.return_value.option.return_value.append.assert_called_once_with()


def test_created_table_accepts_any_schema():
    spark = MagicMock()
    spark.catalog.tableExists.return_value = False
    spark.sql.return_value.collect.return_value = [[7]]
    frame = MagicMock()
    frame.isEmpty.return_value = False

    qip.process_incremental_data(spark, frame, 'molecules', ['experiment_id'], 'a-b')

    statements = ' '.join(call.args[0] for call in spark.sql.call_args_list)
    assert "'write.spark.accept-any-schema'='true'" in statements


def test_created_table_is_tagged_with_the_creating_snapshot():
    spark = MagicMock()
    spark.catalog.tableExists.return_value = False
    spark.sql.return_value.collect.return_value = [[7]]
    frame = MagicMock()
    frame.isEmpty.return_value = False
    frame.count.return_value = 5

    tag, count = qip.process_incremental_data(spark, frame, 'molecules', ['experiment_id'], 'a-b')

    assert (tag, count) == ('v_ab', 5)
    statements = ' '.join(call.args[0] for call in spark.sql.call_args_list)
    assert 'CREATE TAG v_ab AS OF VERSION 7' in statements


def test_empty_frame_creates_no_table_and_no_tag():
    spark = MagicMock()
    spark.catalog.tableExists.return_value = False
    frame = MagicMock()
    frame.isEmpty.return_value = True

    assert qip.process_incremental_data(spark, frame, 'hamiltonian_terms', ['term_id'], 'a-b') == (
        None,
        0,
    )

    frame.write.format.assert_not_called()
    spark.sql.assert_not_called()


def test_append_with_no_new_keys_writes_nothing(monkeypatch):
    spark = MagicMock()
    spark.catalog.tableExists.return_value = True
    new_rows = MagicMock()
    new_rows.count.return_value = 0
    monkeypatch.setattr(qip, 'identify_new_records', lambda *_args: new_rows)
    frame = MagicMock()
    frame.isEmpty.return_value = False

    assert qip.process_incremental_data(spark, frame, 'molecules', ['experiment_id'], 'a-b') == (
        None,
        0,
    )

    new_rows.writeTo.assert_not_called()
    spark.sql.assert_not_called()
