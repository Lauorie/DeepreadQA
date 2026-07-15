"""DeepreadQA HTTP API: production-grade service surface over the QA engine."""
__version__ = "1.1.1"

from .config import ApiConfig  # noqa: E402 - __version__ must precede app imports

__all__ = ["ApiConfig", "__version__"]
