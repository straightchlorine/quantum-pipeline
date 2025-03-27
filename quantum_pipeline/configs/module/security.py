from dataclasses import asdict, dataclass
from typing import Any

from quantum_pipeline.configs.defaults import DEFAULTS


@dataclass
class CertConfig:
    """Configuration for SSL certificates."""

    ssl_dir: str
    ssl_cafile: str
    ssl_certfile: str
    ssl_keyfile: str
    ssl_password: str | None = None
    ssl_crlfile: str | None = None
    ssl_ciphers: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CertConfig':
        return cls(
            ssl_dir=data.get('ssl_dir', ''),
            ssl_cafile=data.get('cafile', DEFAULTS['kafka']['security']['certs']['cafile']),
            ssl_certfile=data.get('certfile', DEFAULTS['kafka']['security']['certs']['certfile']),
            ssl_keyfile=data.get('keyfile', DEFAULTS['kafka']['security']['certs']['keyfile']),
            ssl_password=data.get('password'),
            ssl_crlfile=data.get('crlfile'),
            ssl_ciphers=data.get('ciphers'),
        )


@dataclass
class SaslSslOpts:
    """Configuration for SASL over SSL."""

    sasl_mechanism: str | None
    sasl_plain_username: str | None
    sasl_plain_password: str | None
    sasl_kerberos_service_name: str | None
    sasl_kerberos_domain_name: str | None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SaslSslOpts':
        return cls(
            sasl_mechanism=data.get('sasl_mechanism', ''),
            sasl_plain_username=data.get('sasl_plain_username', ''),
            sasl_plain_password=data.get('sasl_plain_password', ''),
            sasl_kerberos_service_name=data.get('sasl_kerberos_service_name', ''),
            sasl_kerberos_domain_name=data.get('sasl_kerberos_domain_name', ''),
        )


@dataclass
class SecurityConfig:
    """Configuration for Kafka security (SSL & SASL)."""

    ssl: bool
    sasl_ssl: bool
    ssl_check_hostname: bool
    cert_config: CertConfig
    sasl_opts: SaslSslOpts

    @staticmethod
    def get_default() -> 'SecurityConfig':
        return SecurityConfig(
            ssl=DEFAULTS['kafka']['security']['ssl'],
            sasl_ssl=DEFAULTS['kafka']['security']['sasl_ssl'],
            ssl_check_hostname=DEFAULTS['kafka']['security']['ssl_check_hostname'],
            cert_config=CertConfig.from_dict(DEFAULTS['kafka']['security']['certs']),
            sasl_opts=SaslSslOpts.from_dict(DEFAULTS['kafka']['security']['sasl_ssl_opts']),
        )

    def to_dict(self) -> dict:
        return {
            'ssl': self.ssl,
            'sasl_ssl': self.sasl_ssl,
            'ssl_check_hostname': self.ssl_check_hostname,
            'cert_config': self.cert_config.to_dict()
            if isinstance(self.cert_config, CertConfig)
            else {},
            'sasl_opts': self.sasl_opts.to_dict()
            if isinstance(self.sasl_opts, SaslSslOpts)
            else {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SecurityConfig':
        return cls(
            ssl=data.get('ssl', False),
            sasl_ssl=data.get('sasl_ssl', False),
            ssl_check_hostname=data.get('ssl_check_hostname', False),
            cert_config=CertConfig.from_dict(data.get('cert_config', {})),
            sasl_opts=SaslSslOpts.from_dict(data.get('sasl_opts', {})),
        )
