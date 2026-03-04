class Mapper:
    """Base class for operator mappers."""

    def map(self, operator):
        raise NotImplementedError('Subclasses must implement this method')

    def get_qiskit_mapper(self):
        raise NotImplementedError('Subclasses must implement this method')
