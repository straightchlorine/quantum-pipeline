from pathlib import Path
from time import sleep
from typing import Any, Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError

from quantum_pipeline.configs.parsing.producer_config import ProducerConfig
from quantum_pipeline.stream.serialization.interfaces.vqe import (
    VQEDecoratedResultInterface,
)
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult
from quantum_pipeline.utils.logger import get_logger
from quantum_pipeline.utils.schema_registry import SchemaRegistry


class KafkaProducerError(Exception):
    """Custom exception for Kafka producer errors."""


class VQEKafkaProducer:
    def __init__(self, config: ProducerConfig):
        self.config = config
        self.registry = SchemaRegistry()
        self.serializer = VQEDecoratedResultInterface(self.registry)
        self.logger = get_logger(self.__class__.__name__)
        self.producer: Optional[KafkaProducer] = None
        self._initialize_producer()

    def __security_configuration_handling(self) -> dict[str, Any]:
        security = {}
        if self.config.security.ssl or self.config.security.sasl_ssl:
            ssl_dir = self.config.security.cert_config.ssl_dir
            ssl_files = {
                'ssl_cafile': self.config.security.cert_config.ssl_cafile,
                'ssl_certfile': self.config.security.cert_config.ssl_certfile,
                'ssl_keyfile': self.config.security.cert_config.ssl_keyfile,
                'ssl_crlfile': self.config.security.cert_config.ssl_crlfile,
            }

            security = {
                'ssl_password': self.config.security.cert_config.ssl_password,
                'ssl_ciphers': self.config.security.cert_config.ssl_ciphers,
                'ssl_check_hostname': self.config.security.ssl_check_hostname,
            }

            # build the paths based on the specified
            for key, filename in ssl_files.items():
                if filename:
                    security[key] = Path(ssl_dir, filename).as_posix()

            # adjust the configuration based on the protocols used
            if self.config.security.ssl:
                security['security_protocol'] = 'SSL'
            if self.config.security.sasl_ssl:
                security['security_protocol'] = 'SASL_SSL'

                # determine the mechanisim of the connection
                if not self.config.security.sasl_opts.sasl_mechanism:
                    raise ValueError('SASL mechanism required for SASL_SSL')
                security['sasl_mechanism'] = self.config.security.sasl_opts.sasl_mechanism

                # adjust other parameters based on the mechanism
                if security['sasl_mechanism'] in ['PLAIN', 'SCRAM-SHA-256', 'SCRAM-SHA-512']:
                    if not (
                        self.config.security.sasl_opts.sasl_plain_username
                        and self.config.security.sasl_opts.sasl_plain_password
                    ):
                        raise ValueError(
                            f"Username and password required for {security['sasl_mechanism']}"
                        )
                    security.update(
                        {
                            'sasl_plain_username': self.config.security.sasl_opts.sasl_plain_username,
                            'sasl_plain_password': self.config.security.sasl_opts.sasl_plain_password,
                        }
                    )

                elif security['sasl_mechanism'] == 'GSSAPI':
                    security['sasl_kerberos_service_name'] = (
                        self.config.security.sasl_opts.sasl_kerberos_service_name
                    )
                    if self.config.security.sasl_opts.sasl_kerberos_domain_name:
                        security['sasl_kerberos_domain_name'] = (
                            self.config.security.sasl_opts.sasl_kerberos_domain_name
                        )

        print(30 * '=')
        __import__('pprint').pprint(security)
        print(30 * '=')

        return security

    def _initialize_producer(self) -> None:
        """Initialize the Kafka producer with error handling."""

        security = self.__security_configuration_handling()
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.servers,
                value_serializer=lambda v: v,
                retries=self.config.kafka_retries,
                acks=self.config.acks,
                **security,
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
