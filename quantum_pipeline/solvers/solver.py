import os

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_pipeline.configs.argparser import SUPPORTED_OPTIMIZERS, BackendConfig
from quantum_pipeline.utils.logger import get_logger


class Solver:
    """Base class for quantum solvers."""

    backend_config: BackendConfig

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def supported_optimizers_prompt(self):
        supported_optimizers = ''
        for opt, description in SUPPORTED_OPTIMIZERS.items():
            supported_optimizers += f'{opt}: {description}\n'

    def __validate_env(self):
        """Validates the environment variables required for IBM Quantum authentication.

        Returns:
            tuple: (channel, instance, token) if validation succeeds.

        Raises:
            RuntimeError: If any required environment variable is missing or invalid.
        """
        channel = os.getenv('IBM_RUNTIME_CHANNEL')
        instance = os.getenv('IBM_RUNTIME_INSTANCE')
        token = os.getenv('IBM_RUNTIME_TOKEN')

        if not channel or not instance or not token:
            raise RuntimeError('IBM Quantum credentials not found')

        if channel not in {'ibm_quantum', 'ibm_cloud', 'local'}:
            raise RuntimeError('Invalid IBM Quantum channel')

        return channel, instance, token

    def _get_service(self):
        """Authenticates and connects to the IBM Quantum platform.

        Returns:
            QiskitRuntimeService: Connected IBM Quantum service instance.

        Raises:
            RuntimeError: If authentication or connection fails.
        """
        self.logger.info('Authenticating with IBM Quantum platform...')
        try:
            channel, instance, token = self.__validate_env()

            self.logger.info('Connecting to IBM Quantum...')
            service = QiskitRuntimeService(
                channel=channel,
                instance=instance,
                token=token,
            )
            self.logger.info('Connected to IBM Quantum.')

            return service

        except Exception as e:
            self.logger.error(f'Failed to connect to IBM Quantum: {str(e)}')
            raise RuntimeError('IBM Quantum connection failed.') from e

    def get_backend(self):
        if not self.backend_config:
            raise RuntimeError('Backend configuration not set.')

        if self.backend_config.local:
            self.logger.info('Initializing Aer simulator backend...')
            backend = AerSimulator()
            self.logger.info('Aer simulator backend initialized.')
        else:
            service = self._get_service()
            config = self.backend_config.to_dict()
            try:
                if config:
                    self.logger.info(f'Waiting for backend fitting the requirements: {config}...')
                    backend = service.least_busy(operational=True, **config)
                else:
                    self.logger.info('Waiting for a least busy backend...')
                    backend = service.least_busy(operational=True)
            except Exception as e:
                self.logger.error(f'Failed to get backend: {str(e)}')
                raise RuntimeError('Backend retrieval failed.') from e

            self.logger.info(f'Backend {backend.name} acquired.')
        return backend
