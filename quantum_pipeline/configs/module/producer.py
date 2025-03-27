from dataclasses import asdict, dataclass
from typing import Any

from quantum_pipeline.configs.defaults import DEFAULTS
from quantum_pipeline.configs.module.security import SecurityConfig


@dataclass
class ProducerConfig:
    """Configuration for Kafka Producer object"""

    servers: str
    topic: str
    security: SecurityConfig
    retries: int
    retry_delay: int
    kafka_retries: int
    acks: str = 'all'
    timeout: int = 10

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        data = asdict(self)
        data['security'] = self.security.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProducerConfig':
        """Create a ProducerConfig instance from a dictionary."""
        return cls(
            servers=data.get('servers', DEFAULTS['kafka']['servers']),
            topic=data.get('topic', DEFAULTS['kafka']['topic']),
            security=SecurityConfig.from_dict(data.get('security', {})),
            retries=data.get('retries', DEFAULTS['kafka']['retries']),
            retry_delay=data.get('retry_delay', DEFAULTS['kafka']['retry_delay']),
            kafka_retries=data.get('kafka_retries', DEFAULTS['kafka']['internal_retries']),
            acks=data.get('acks', DEFAULTS['kafka']['acks']),
            timeout=data.get('timeout', DEFAULTS['kafka']['timeout']),
        )
