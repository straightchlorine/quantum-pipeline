from dataclasses import asdict, dataclass
from typing import Any

from quantum_pipeline.configs.defaults import DEFAULTS
from quantum_pipeline.configs.module.security import SecurityConfig


@dataclass
class ProducerConfig:
    servers: str
    topic: str
    security: SecurityConfig
    retries: int = 3
    retry_delay: int = 2
    kafka_retries: int = 5
    acks: str = 'all'
    timeout: int = 10
    max_request_size: int = 134217728
    buffer_memory: int = 134217728

    def to_dict(self) -> dict:
        data = asdict(self)
        data['security'] = self.security.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProducerConfig':
        return cls(
            servers=data.get('servers', DEFAULTS['kafka']['servers']),
            topic=data.get('topic', DEFAULTS['kafka']['topic']),
            security=SecurityConfig.from_dict(data.get('security', {})),
            retries=data.get('retries', DEFAULTS['kafka']['retries']),
            retry_delay=data.get('retry_delay', DEFAULTS['kafka']['retry_delay']),
            kafka_retries=data.get('kafka_retries', DEFAULTS['kafka']['internal_retries']),
            acks=data.get('acks', DEFAULTS['kafka']['acks']),
            timeout=data.get('timeout', DEFAULTS['kafka']['timeout']),
            max_request_size=data.get('max_request_size', DEFAULTS['kafka']['max_request_size']),
            buffer_memory=data.get('buffer_memory', DEFAULTS['kafka']['buffer_memory']),
        )
