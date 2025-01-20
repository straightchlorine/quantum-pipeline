from quantum_simulation.report.report_generator import ReportGenerator
from qiskit_ibm_runtime import QiskitRuntimeService
import os

from quantum_simulation.utils.logger import get_logger


class Runner:
    def __init__(self, report: ReportGenerator, ibm=False):
        self.report = report
        self.logger = get_logger(self.__class__.__name__)
        self.ibm = ibm

    def validate_env(self):
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

    def get_provider(self):
        """Authenticates and connects to the IBM Quantum platform.

        Returns:
            QiskitRuntimeService: Connected IBM Quantum service instance.

        Raises:
            RuntimeError: If authentication or connection fails.
        """
        self.logger.info('Authenticating with IBM Quantum platform...')
        try:
            channel, instance, token = self.validate_env()

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


if __name__ == '__main__':
    runner = Runner(ReportGenerator())
    runner.get_provider()
