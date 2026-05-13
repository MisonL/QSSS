"""Shared cache exception types."""


class CacheDeserializationError(RuntimeError):
    """Raised when a stored cache value cannot be decoded."""
