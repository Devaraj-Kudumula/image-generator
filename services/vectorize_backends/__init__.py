"""Pluggable vectorization backends (vtracer, paid APIs)."""

from services.vectorize_backends.paid_backends import (
    vectorize_via_paid_backend,
    supported_backends,
)

__all__ = [
    'vectorize_via_paid_backend',
    'supported_backends',
]
