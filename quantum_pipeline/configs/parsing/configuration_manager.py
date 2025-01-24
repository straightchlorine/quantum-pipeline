import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict

from quantum_pipeline.configs import settings
from quantum_pipeline.configs.parsing.backend_config import BackendConfig
from quantum_pipeline.configs.parsing.producer_config import ProducerConfig
from quantum_pipeline.utils.logger import get_logger


class ConfigurationManager:
    """Class for managing application configurations."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def create_kafka_config(self, args: argparse.Namespace) -> ProducerConfig:
        """Create Kafka configuration from arguments."""
        return ProducerConfig.from_dict(
            {
                'servers': args.servers,
                'topic': args.topic,
                'retries': args.retries,
                'retry_delay': args.retry_delay,
                'kafka_retries': args.internal_retries,
                'acks': args.acks,
                'timeout': args.timeout,
            }
        )

    def create_backend_config(self, args: argparse.Namespace) -> BackendConfig:
        """Create Backend configuration from arguments."""
        return BackendConfig.from_dict(
            {
                'local': args.ibm,
                'min_num_qubits': args.min_qubits,
                'optimization_level': args.optimization_level,
                'filters': None,
            }
        )

    def dump(self, args: argparse.Namespace, config: Dict[str, Any]):
        """Dump the configurations for debugging or logging."""

        self.logger.debug('Dumping configuration into the file...')

        # create a copy thats JSON serializable
        config_dict = deepcopy(config)
        config_dict['backend_config'] = self.create_backend_config(args)
        config_dict['kafka_config'] = self.create_kafka_config(args)
        config_dict['backend_config'] = config_dict['backend_config'].to_dict()
        config_dict['kafka_config'] = config_dict['kafka_config'].to_dict()

        for k in list(config_dict.keys()):
            if k in config_dict.get('backend_config', {}):
                config_dict.pop(k)
            if k in config_dict.get('kafka_config', {}):
                config_dict.pop(k)

        #  create a filename
        file_name = Path(args.file).stem
        basis_set = args.basis
        optimizer = args.optimizer
        backend_local = 'api' if args.ibm else 'local'
        current_date = datetime.now().strftime('%Y%m%d')

        file_path = f'{file_name}-{basis_set}-{optimizer}-{backend_local}-{current_date}.json'
        self.config_path = Path(settings.RUN_CONFIGS, file_path)

        try:
            with open(self.config_path, 'w') as file:
                json.dump(config_dict, file, indent=4)
        except Exception as e:
            self.logger.error(f'Failed to save configuration: {e}')
            raise

        self.logger.info(f'Configuration saved to:\n\n{file_path}\n')
        self.logger.debug(f'Arguments:\n\n{vars(args)}\n')
        self.logger.debug(f'Configurations:\n\n{config_dict}\n')

    def load(self, file_path: str) -> Dict[str, Any]:
        """Load the configurations from a JSON file."""
        try:
            self.logger.debug(f'Loading configuration from:\n\n{file_path}\n')
            with open(file_path, 'r') as file:
                config_dict = json.load(file)

            # reconstruct config objects
            backend_config = BackendConfig.from_dict(config_dict['backend_config'])
            kafka_config = ProducerConfig.from_dict(config_dict['kafka_config'])

            # build full dictionary
            config_dict['backend_config'] = backend_config
            config_dict['kafka_config'] = kafka_config

            self.logger.info(f'Configuration loaded from:\n\n{file_path}\n')
            self.logger.debug(f'Loaded configurations:\n\n{config_dict}\n')
            return config_dict

        except FileNotFoundError:
            self.logger.error(f'Configuration file not found: {file_path}')
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse JSON configuration: {e}')
            raise

    def get_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert parsed arguments dynamically into a dictionary."""

        if args.load:
            self.load(args.load)

        # create a dictionary from the parsed arguments
        config_dict = {key: value for key, value in vars(args).items() if key != 'local'}
        config_dict['kafka_config'] = self.create_kafka_config(args)
        config_dict['backend_config'] = self.create_backend_config(args)

        if args.dump:
            self.dump(args, config_dict)

        return config_dict
