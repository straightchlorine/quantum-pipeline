from abc import ABC, abstractmethod

from quantum_pipeline.utils.logger import get_logger


class Runner(ABC):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self):
        """Execute the quantum algorithm. Subclasses must implement this."""
        ...
