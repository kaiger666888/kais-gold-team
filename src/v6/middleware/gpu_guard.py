"""GPU guard middleware — auto-evicts engines on OOM before API requests."""
from src.v6.middleware import GPUGuardMiddleware

__all__ = ["GPUGuardMiddleware"]
