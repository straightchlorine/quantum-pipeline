from abc import ABC, abstractmethod


class Mapper(ABC):
    """Base class for operator mappers."""

    @abstractmethod
    def map(self, operator):
        """Map a fermionic operator to qubit operator."""
        ...

    @abstractmethod
    def get_qiskit_mapper(self):
        """Return the underlying Qiskit mapper instance."""
        ...
