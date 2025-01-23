from time import sleep
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError

from quantum_pipeline.configs.parsing.producer_config import ProducerConfig
from quantum_pipeline.stream.serialization.interfaces.vqe import (
    VQEDecoratedResultInterface,
)
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult
from quantum_pipeline.utils.logger import get_logger


class KafkaProducerError(Exception):
    """Custom exception for Kafka producer errors."""


class VQEKafkaProducer:
    def __init__(self, config: ProducerConfig):
        self.config = config
        self.serializer = VQEDecoratedResultInterface()
        self.logger = get_logger(self.__class__.__name__)
        self.producer: Optional[KafkaProducer] = None
        self._initialize_producer()

    def _initialize_producer(self) -> None:
        """Initialize the Kafka producer with error handling."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.servers,
                value_serializer=lambda v: v,
                retries=self.config.kafka_retries,
                acks=self.config.acks,
            )
        except Exception as e:
            self.logger.error(f'Failed to initialize KafkaProducer: {str(e)}')
            raise KafkaProducerError(f'Producer initialization failed: {str(e)}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _serialize_result(self, result: VQEDecoratedResult) -> bytes:
        """Serialize the result with error handling."""
        try:
            return self.serializer.to_avro_bytes(result)
        except Exception as e:
            self.logger.error('Object serialization failed!')
            raise KafkaProducerError(f'Serialization failed: {str(e)}')

    def _send_with_retry(self, avro_bytes: bytes) -> bool:
        """Send message with retry logic, returns success status."""
        assert isinstance(self.producer, KafkaProducer)

        for attempt in range(1, self.config.retries + 1):
            try:
                record_metadata = self.producer.send(self.config.topic, avro_bytes).get(
                    timeout=self.config.timeout
                )

                self.logger.info(
                    f'Message sent successfully to {record_metadata.topic}-'
                    f'{record_metadata.partition} at offset {record_metadata.offset}'
                )
                return True

            except KafkaError as ke:
                self.logger.warning(
                    f'Attempt {attempt}/{self.config.retries}: Kafka error: {str(ke)}'
                )
                # restart if attempts left
                if attempt < self.config.retries:
                    sleep(self.config.retry_delay)
                    continue
                raise KafkaProducerError(f'Failed after {self.config.retries} retries: {str(ke)}')

            except Exception as e:
                self.logger.error(f'Unexpected error during send: {str(e)}')
                raise KafkaProducerError(f'Send failed: {str(e)}')

        return False

    def _send_and_flush(self, result):
        assert isinstance(self.producer, KafkaProducer)
        avro_bytes = self._serialize_result(result)
        self._send_with_retry(avro_bytes)
        self.producer.flush()

    def send_result(self, result: VQEDecoratedResult) -> None:
        """Send VQEDecoratedResult to Kafka topic with proper error handling."""
        self.logger.info(f'Sending the result to the Kafka topic {self.config.topic}...')
        if not self.producer:
            raise KafkaProducerError('Producer not initialized')

        try:
            self._send_and_flush(result)
        except KafkaProducerError as e:
            self.logger.error(f'Failed to send result: {str(e)}')
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error: {str(e)}')
            raise KafkaProducerError(f'Unexpected error during send: {str(e)}')

    def close(self) -> None:
        """Close the Kafka producer with proper error handling."""
        if self.producer:
            try:
                self.producer.close()
                self.logger.info('Kafka producer closed successfully.')
            except Exception as e:
                self.logger.error(f'Error closing producer: {str(e)}')
                raise KafkaProducerError(f'Failed to close producer: {str(e)}')
