from typing import Any

import pytest

from quantum_pipeline.configs.module.security import (
    CertConfig,
    SaslSslOpts,
    SecurityConfig,
)


@pytest.fixture
def sample_defaults() -> dict[str, Any]:
    """Fixture to provide sample default configurations."""
    return {
        'kafka': {
            'security': {
                'ssl': False,
                'sasl_ssl': False,
                'ssl_check_hostname': False,
                'certs': {
                    'ssl_dir': '/path/to/ssl',
                    'cafile': '/path/to/default/cafile',
                    'certfile': '/path/to/default/certfile',
                    'keyfile': '/path/to/default/keyfile',
                },
                'sasl_opts': {
                    'sasl_mechanism': 'PLAIN',
                    'sasl_plain_username': '',
                    'sasl_plain_password': '',
                    'sasl_kerberos_service_name': '',
                    'sasl_kerberos_domain_name': '',
                },
            }
        }
    }


def test_cert_config_init():
    """Test initialization of CertConfig with all parameters."""
    cert_config = CertConfig(
        ssl_dir='/path/to/ssl',
        ssl_cafile='/path/to/cafile',
        ssl_certfile='/path/to/certfile',
        ssl_keyfile='/path/to/keyfile',
        ssl_password='secret',
        ssl_crlfile='/path/to/crlfile',
        ssl_ciphers='HIGH:!aNULL:!MD5',
    )

    assert cert_config.ssl_dir == '/path/to/ssl'
    assert cert_config.ssl_cafile == '/path/to/cafile'
    assert cert_config.ssl_certfile == '/path/to/certfile'
    assert cert_config.ssl_keyfile == '/path/to/keyfile'
    assert cert_config.ssl_password == 'secret'
    assert cert_config.ssl_crlfile == '/path/to/crlfile'
    assert cert_config.ssl_ciphers == 'HIGH:!aNULL:!MD5'


def test_cert_config_default_values(sample_defaults):
    """Test CertConfig initialization with default values."""
    cert_config = CertConfig.from_dict(
        sample_defaults['kafka']['security']['certs'],
    )

    assert cert_config.ssl_dir == '/path/to/ssl'
    assert cert_config.ssl_cafile == '/path/to/default/cafile'
    assert cert_config.ssl_certfile == '/path/to/default/certfile'
    assert cert_config.ssl_keyfile == '/path/to/default/keyfile'
    assert cert_config.ssl_password is None
    assert cert_config.ssl_crlfile is None
    assert cert_config.ssl_ciphers is None


def test_cert_config_to_dict():
    """Test conversion of CertConfig to dictionary."""
    cert_config = CertConfig(
        ssl_dir='/path/to/ssl',
        ssl_cafile='/path/to/cafile',
        ssl_certfile='/path/to/certfile',
        ssl_keyfile='/path/to/keyfile',
        ssl_password='secret',
        ssl_crlfile='/path/to/crlfile',
        ssl_ciphers='HIGH:!aNULL:!MD5',
    )

    cert_dict = cert_config.to_dict()

    assert cert_dict['ssl_dir'] == '/path/to/ssl'
    assert cert_dict['ssl_cafile'] == '/path/to/cafile'
    assert cert_dict['ssl_certfile'] == '/path/to/certfile'
    assert cert_dict['ssl_keyfile'] == '/path/to/keyfile'
    assert cert_dict['ssl_password'] == 'secret'
    assert cert_dict['ssl_crlfile'] == '/path/to/crlfile'
    assert cert_dict['ssl_ciphers'] == 'HIGH:!aNULL:!MD5'


def test_sasl_ssl_opts_init():
    """Test initialization of SaslSslOpts with all parameters."""
    sasl_opts = SaslSslOpts(
        sasl_mechanism='PLAIN',
        sasl_plain_username='user',
        sasl_plain_password='pass',
        sasl_kerberos_service_name='kafka',
        sasl_kerberos_domain_name='example.com',
    )

    assert sasl_opts.sasl_mechanism == 'PLAIN'
    assert sasl_opts.sasl_plain_username == 'user'
    assert sasl_opts.sasl_plain_password == 'pass'
    assert sasl_opts.sasl_kerberos_service_name == 'kafka'
    assert sasl_opts.sasl_kerberos_domain_name == 'example.com'


def test_sasl_ssl_opts_default_values():
    """Test SaslSslOpts initialization with default values."""
    sasl_opts = SaslSslOpts.from_dict({})

    assert sasl_opts.sasl_mechanism == ''
    assert sasl_opts.sasl_plain_username == ''
    assert sasl_opts.sasl_plain_password == ''
    assert sasl_opts.sasl_kerberos_service_name == ''
    assert sasl_opts.sasl_kerberos_domain_name == ''


def test_security_config_init(sample_cert_config=None, sample_sasl_opts=None):
    """Test initialization of SecurityConfig with all parameters."""
    sample_cert_config = sample_cert_config or CertConfig(
        ssl_dir='/path/to/ssl',
        ssl_cafile='/path/to/cafile',
        ssl_certfile='/path/to/certfile',
        ssl_keyfile='/path/to/keyfile',
    )

    sample_sasl_opts = sample_sasl_opts or SaslSslOpts(
        sasl_mechanism='PLAIN',
        sasl_plain_username='user',
        sasl_plain_password='pass',
        sasl_kerberos_service_name='',
        sasl_kerberos_domain_name='',
    )

    security_config = SecurityConfig(
        ssl=True,
        sasl_ssl=True,
        ssl_check_hostname=True,
        cert_config=sample_cert_config,
        sasl_opts=sample_sasl_opts,
    )

    assert security_config.ssl is True
    assert security_config.sasl_ssl is True
    assert security_config.ssl_check_hostname is True
    assert security_config.cert_config == sample_cert_config
    assert security_config.sasl_opts == sample_sasl_opts


def test_security_config_get_default():
    """Test the get_default method of SecurityConfig."""
    default_config = SecurityConfig.get_default()

    assert default_config.ssl is False
    assert default_config.sasl_ssl is False
    assert default_config.ssl_check_hostname is True
    assert isinstance(default_config.cert_config, CertConfig)
    assert isinstance(default_config.sasl_opts, SaslSslOpts)


def test_security_config_to_dict(sample_cert_config=None, sample_sasl_opts=None):
    """Test conversion of SecurityConfig to dictionary."""
    sample_cert_config = sample_cert_config or CertConfig(
        ssl_dir='/path/to/ssl',
        ssl_cafile='/path/to/cafile',
        ssl_certfile='/path/to/certfile',
        ssl_keyfile='/path/to/keyfile',
    )

    sample_sasl_opts = sample_sasl_opts or SaslSslOpts(
        sasl_mechanism='PLAIN',
        sasl_plain_username='user',
        sasl_plain_password='pass',
        sasl_kerberos_service_name='',
        sasl_kerberos_domain_name='',
    )

    security_config = SecurityConfig(
        ssl=True,
        sasl_ssl=True,
        ssl_check_hostname=True,
        cert_config=sample_cert_config,
        sasl_opts=sample_sasl_opts,
    )

    security_dict = security_config.to_dict()

    assert security_dict['ssl'] is True
    assert security_dict['sasl_ssl'] is True
    assert security_dict['ssl_check_hostname'] is True
    assert isinstance(security_dict['cert_config'], dict)
    assert isinstance(security_dict['sasl_opts'], dict)


def test_security_config_from_dict():
    """Test creating SecurityConfig from a dictionary."""

    input_dict = {
        'ssl': True,
        'sasl_ssl': True,
        'ssl_check_hostname': True,
        'cert_config': {
            'ssl_dir': '/path/to/ssl',
            'ssl_cafile': '/path/to/cafile',
            'ssl_certfile': '/path/to/certfile',
            'ssl_keyfile': '/path/to/keyfile',
        },
        'sasl_opts': {
            'sasl_mechanism': 'PLAIN',
            'sasl_plain_username': 'user',
            'sasl_plain_password': 'pass',
        },
    }

    security_config = SecurityConfig.from_dict(input_dict)

    assert security_config.ssl is True
    assert security_config.sasl_ssl is True
    assert security_config.ssl_check_hostname is True
    assert security_config.cert_config.ssl_dir == '/path/to/ssl'
    assert security_config.sasl_opts.sasl_mechanism == 'PLAIN'


def test_security_config_partial_dict():
    """Test creating SecurityConfig with a partial dictionary."""
    security_config = SecurityConfig.from_dict({})

    assert security_config.ssl is False
    assert security_config.sasl_ssl is False
    assert security_config.ssl_check_hostname is False
    assert isinstance(security_config.cert_config, CertConfig)
    assert isinstance(security_config.sasl_opts, SaslSslOpts)
