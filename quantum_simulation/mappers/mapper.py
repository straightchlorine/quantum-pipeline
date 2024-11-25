class Mapper:
    """Base class for operator mappers."""

    def map(self, operator):
        raise NotImplementedError('Subclasses must implement this method')
