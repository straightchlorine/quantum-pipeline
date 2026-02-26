import re
from time import sleep

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.stream.kafka_security import KafkaSecurity
from quantum_pipeline.stream.serialization.interfaces.vqe import (
    VQEDecoratedResultInterface,
)
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult
from quantum_pipeline.utils.logger import get_logger
from quantum_pipeline.utils.schema_registry import SchemaRegistry


class KafkaProducerError(Exception):
    """Custom exception for Kafka producer errors."""


class VQEKafkaProducer:
    """Kafka producer for sending VQE (Variational Quantum Eigensolver) results.

    Provides:
        - secure Kafka connections;
        - serialization of VQE results;
        - sending of results with retry mechanisms.
    """

    def __init__(self, config: ProducerConfig):
        """Initialize the Kafka producer with given configuration.

        Args:
            config: Configuration for the Kafka producer.
        """

        self.logger = get_logger(self.__class__.__name__)

        self.config = config
        self.producer: KafkaProducer | None = None

        self.security = KafkaSecurity(self.config)
        self.registry = SchemaRegistry()
        self.serializer = VQEDecoratedResultInterface(self.registry)

        self._initialize_producer()

    def _initialize_producer(self) -> None:
        """Initialize the Kafka producer with error handling."""

        try:
            security_config = self.security.build_security_config()
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.servers,
                value_serializer=lambda v: v,
                retries=self.config.kafka_retries,
                acks=self.config.acks,
                **security_config,
            )
        except NoBrokersAvailable as e:
            self.logger.error('No brokers available. Check the Kafka broker configuration.')
            raise KafkaProducerError('No brokers available.') from e
        except Exception as e:
            self.logger.error(f'Failed to initialize KafkaProducer: {e!s}')
            raise KafkaProducerError(f'Failed to initialize producer: {e!s}') from e

    def _serialize_result(self, result: VQEDecoratedResult) -> bytes:
        """Serialize the result.

        Args:
            result: VQEDecoratedResult to be serialized.

        Returns:
            Serialized result in Avro bytes.

        Raises:
            KafkaProducerError: If serialization fails
        """
        try:
            self.logger.info(f'Serializing the result {self.serializer.schema_name}...')
            return self.serializer.to_avro_bytes(
                result,
                schema_name=self.serializer.schema_name,
            )
        except Exception as e:
            self.logger.error('Object serialization failed!')
            raise KafkaProducerError(f'Serialization failed: {e!s}') from e

    def _send_with_retry(self, avro_bytes: bytes) -> bool:
        """Send message with retry logic.

        Args:
            avro_bytes: Avro bytes to send.

        Returns:
            True if message sent successfully, False otherwise.

        Raises:
            KafkaProducerError: If send fails after all retries
        """
        if self.producer is None:
            raise KafkaProducerError('Producer not initialized')

        last_error: KafkaError | None = None
        for attempt in range(1, self.config.retries + 1):
            last_error = self._attempt_send(avro_bytes, attempt)
            if last_error is None:
                return True
            if attempt < self.config.retries:
                sleep(self.config.retry_delay)

        raise KafkaProducerError(
            f'Failed after {self.config.retries} retries: {last_error!s}'
        ) from last_error

    def _attempt_send(self, avro_bytes: bytes, attempt: int) -> KafkaError | None:
        """Attempt a single send, returning the error if one occurred."""
        try:
            record_metadata = self.producer.send(
                self.config.topic,
                avro_bytes,
            ).get(timeout=self.config.timeout)

            self.logger.info(
                f'Message sent successfully to {record_metadata.topic}-'
                f'{record_metadata.partition} at '
                f'offset {record_metadata.offset}.'
            )
            return None

        except KafkaError as ke:
            self.logger.warning(f'Attempt {attempt}/{self.config.retries}: Kafka error: {ke!s}')
            return ke

        except Exception as e:
            self.logger.error(f'Unexpected error during send: {e!s}')
            raise KafkaProducerError(f'Send failed: {e!s}') from e

    def _send_and_flush(self, result):
        """Serialize and sent the result and after that - flush.

        Args:
            result: VQEDecoratedResult to be sent.
        """
        if self.producer is None:
            raise KafkaProducerError('Producer not initialized')

        avro_bytes = self._serialize_result(result)
        self._send_with_retry(avro_bytes)
        self.producer.flush()

    def _update_topic(self, result: VQEDecoratedResult) -> None:
        """Update topic and schema names with simulation suffix.

        Args:
            result: VQEDecoratedResult with simulation details

        Returns:
            Updated topic and schema names
        """
        # get the suffix with simulation details
        suffix = result.get_schema_suffix()
        self.logger.info(f'Updating the topic with the new simulation suffix {suffix}...')

        # remove existing suffixes from the topic and schema names
        suffix_pattern = r'_mol.*$'
        for attr in ['topic', 'schema_name', 'result_interface.schema_name']:
            if attr.startswith('topic'):
                base_name = getattr(self.config, attr)
            elif attr.startswith('schema_name'):
                base_name = getattr(self.serializer, attr)
            else:
                base_name = self.serializer.result_interface.schema_name

            updated_name = re.sub(suffix_pattern, '', base_name) + suffix

            if self.config.topic == base_name:
                self.config.topic = updated_name
            if self.serializer.schema_name == base_name:
                self.serializer.schema_name = updated_name
            if self.serializer.result_interface.schema_name == base_name:
                self.serializer.result_interface.schema_name = updated_name

    def send_result(self, result: VQEDecoratedResult) -> None:
        """Send VQE results to Kafka.

        Args:
            result: VQEDecoratedResult to be sent.

        Raises:
            KafkaProducerError: If sending fails
        """
        self._update_topic(result)

        self.logger.info(f'Sending the result to the Kafka topic {self.config.topic}...')
        if not self.producer:
            raise KafkaProducerError('Producer not initialized')

        try:
            self._send_and_flush(result)
        except KafkaProducerError as e:
            self.logger.error(f'Failed to send result: {e!s}')
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error: {e!s}')
            raise KafkaProducerError(f'Unexpected error during send: {e!s}') from e

    def close(self) -> None:
        """Close the producer safely."""
        if self.producer:
            try:
                self.producer.close()
                self.logger.info('Kafka producer closed successfully.')
            except Exception as e:
                self.logger.error(f'Error closing producer: {e!s}')
                raise KafkaProducerError(f'Failed to close producer: {e!s}') from e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
