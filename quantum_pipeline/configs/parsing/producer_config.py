from typing import Any, Dict
from dataclasses import asdict, dataclass
from quantum_pipeline.configs.defaults import DEFAULTS


@dataclass
class ProducerConfig:
    """Configuration for Kafka Producer object"""

    servers: str
    topic: str
    retries: int = 3
    retry_delay: int = 2
    kafka_retries: int = 5
    acks: str = 'all'
    timeout: int = 10

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProducerConfig':
        """Create a ProducerConfig instance from a dictionary."""
        return cls(
            servers=data.get('servers', DEFAULTS['kafka']['servers']),
            topic=data.get('topic', DEFAULTS['kafka']['topic']),
            retries=data.get('retries', DEFAULTS['kafka']['retries']),
            retry_delay=data.get('retry_delay', DEFAULTS['kafka']['retry_delay']),
            kafka_retries=data.get('kafka_retries', DEFAULTS['kafka']['internal_retries']),
            acks=data.get('acks', DEFAULTS['kafka']['acks']),
            timeout=data.get('timeout', None),
        )
