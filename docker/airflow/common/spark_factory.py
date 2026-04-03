"""Shared Spark session factory for quantum pipeline scripts."""

from pyspark.sql import SparkSession


def create_spark_session(app_name):
    """Create a Spark session with Iceberg catalog configuration.

    Catalog configuration is read from spark-defaults.conf mounted
    into the Spark container. This factory only sets the app name.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
