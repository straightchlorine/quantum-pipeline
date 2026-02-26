from pathlib import Path
from typing import Any

from quantum_pipeline.configs.module.producer import ProducerConfig


class KafkaSecurity:
    def __init__(self, config: ProducerConfig):
        """Initialize the Kafka security configuration.

        Args:
            config: Configuration for the Kafka connection.
        """
        self.config = config

    def _get_ssl_config(self) -> dict[str, Any]:
        """Generate SSL configuration parameters.

        Returns:
            SSL configuration
        """
        ssl_dir = self.config.security.cert_config.ssl_dir
        ssl_files = {
            'ssl_cafile': self.config.security.cert_config.ssl_cafile,
            'ssl_certfile': self.config.security.cert_config.ssl_certfile,
            'ssl_keyfile': self.config.security.cert_config.ssl_keyfile,
            'ssl_crlfile': self.config.security.cert_config.ssl_crlfile,
        }

        ssl_config = {
            'ssl_password': self.config.security.cert_config.ssl_password,
            'ssl_ciphers': self.config.security.cert_config.ssl_ciphers,
            'ssl_check_hostname': self.config.security.ssl_check_hostname,
        }

        # build full paths for SSL files
        for key, filename in ssl_files.items():
            if filename:
                ssl_config[key] = Path(ssl_dir, filename).as_posix()

        # set security protocol
        if self.config.security.ssl:
            ssl_config['security_protocol'] = 'SSL'
        return ssl_config

    def _validate_plain_sasl(self, sasl_config: dict[str, Any]) -> None:
        """Validate and configure PLAIN/SCRAM SASL settings.

        Args:
            sasl_config: SASL configuration dictionary to update

        Raises:
            ValueError: If username or password is missing
        """
        opts = self.config.security.sasl_opts
        if not (opts.sasl_plain_username and opts.sasl_plain_password):
            raise ValueError(f'Username and password required for {sasl_config["sasl_mechanism"]}')

        sasl_config.update(
            {
                'sasl_plain_username': opts.sasl_plain_username,
                'sasl_plain_password': opts.sasl_plain_password,
            }
        )

    def _configure_gssapi_sasl(self, sasl_config: dict[str, Any]) -> None:
        """Configure GSSAPI (Kerberos) SASL settings.


        Args:
            sasl_config: SASL configuration dictionary to update
        """
        opts = self.config.security.sasl_opts
        sasl_config['sasl_kerberos_service_name'] = opts.sasl_kerberos_service_name

        if opts.sasl_kerberos_domain_name:
            sasl_config['sasl_kerberos_domain_name'] = opts.sasl_kerberos_domain_name

    def _get_sasl_config(self) -> dict[str, Any]:
        """Generate SASL configuration parameters.

        Returns:
            SASL configuration

        Raises:
            ValueError: If SASL configuration is invalid
        """
        if not self.config.security.sasl_ssl:
            return {}

        sasl_config = {}
        sasl_config['security_protocol'] = 'SASL_SSL'

        # validate SASL mechanism
        mechanism = self.config.security.sasl_opts.sasl_mechanism
        if not mechanism:
            raise ValueError('SASL mechanism is required for SASL_SSL')
        sasl_config['sasl_mechanism'] = mechanism

        # configure based on mechanism
        if mechanism in ['PLAIN', 'SCRAM-SHA-256', 'SCRAM-SHA-512']:
            self._validate_plain_sasl(sasl_config)
        elif mechanism == 'GSSAPI':
            self._configure_gssapi_sasl(sasl_config)

        return sasl_config

    def build_security_config(self) -> dict[str, Any]:
        """Build security configuration for Kafka connection.

        Returns
            Security configuration parameters

        Raises:
            ValueError: If security configuration is invalid
        """
        if not self.config.security.ssl and not self.config.security.sasl_ssl:
            return {}

        security_config = self._get_ssl_config()
        security_config.update(self._get_sasl_config())

        return security_config
