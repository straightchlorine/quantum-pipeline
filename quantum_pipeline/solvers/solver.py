import logging
import os
import sys

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_pipeline.configs.parsing.backend_config import BackendConfig
from quantum_pipeline.configs.settings import LOG_LEVEL, SUPPORTED_OPTIMIZERS
from quantum_pipeline.utils.logger import get_logger


class Solver:
    """Base class for quantum solvers."""

    backend_config: BackendConfig

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def supported_optimizers_prompt(self):
        supported_optimizers = ''
        for opt, description in SUPPORTED_OPTIMIZERS.items():
            supported_optimizers += f'{opt}: {description}'

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
            self.logger.error('IBM Quantum credentials not found.')
            raise RuntimeError('IBM Quantum credentials not found')

        if channel not in {'ibm_quantum', 'ibm_cloud', 'local'}:
            self.logger.error('Invalid IBM Quantum channel.')
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

        except Exception:
            self.logger.error('IBM Quantum connection failed.')
            if LOG_LEVEL == logging.DEBUG:
                raise RuntimeError('IBM Quantum connection failed.')
            else:
                sys.exit(1)

    def _get_noise_model(self, backend):
        provider = self._get_service()

        try:
            self.logger.info(f'Initializing noise model based on {backend}...')
            backend = provider.get_backend(backend)
            noise_model = NoiseModel.from_backend(backend)
            self.logger.info('Initialized noise model.')
            self.logger.debug(f'Model:\n\n{noise_model}')
        except Exception:
            self.logger.error(f'Failed to get the noise model for backend {backend}.')
            if LOG_LEVEL == logging.DEBUG:
                raise RuntimeError(f'Failed to get the noise model for backend {backend}.')
            else:
                sys.exit(1)

        return noise_model

    def get_backend(self):
        if not self.backend_config:
            raise RuntimeError('Backend configuration not set.')

        noise_model = None
        if self.backend_config.noise:
            noise_model = self._get_noise_model(self.backend_config.noise)

        if self.backend_config.local:
            if self.backend_config.gpu:
                self.logger.info('Initializing Aer simulator backend with GPU acceleration...')

                backend = AerSimulator(
                    method=self.backend_config.simulation_method,
                    **self.backend_config.gpu_opts,
                    noise_model=noise_model if noise_model else None,
                )
            else:
                self.logger.info('Initializing Aer simulator backend...')
                backend = AerSimulator(
                    method=self.backend_config.simulation_method,
                    noise_model=noise_model if noise_model else None,
                )

            self.logger.info('Aer simulator backend initialized.')
        else:
            try:
                service = self._get_service()
                config = self.backend_config

                if config.min_num_qubits is not None and config.filters is not None:
                    self.logger.info(f'Waiting for backend fitting the requirements: {config}...')
                    backend = service.least_busy(operational=True, **config.to_dict())
                else:
                    self.logger.info('Waiting for a least busy backend...')
                    backend = service.least_busy(operational=True)
            except Exception:
                self.logger.error(f'Failed to get backend:\n\n{self.backend_config.to_dict()}\n')
                raise RuntimeError('Backend retrieval failed.')

            self.logger.info(f'Backend {backend.name} acquired.')
        return backend
