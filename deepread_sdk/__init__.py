"""DeepRead SDK: progressive-access views over a local markdown corpus."""
from .build import build_store
from .reader import Reader

__all__ = ["Reader", "build_store"]
