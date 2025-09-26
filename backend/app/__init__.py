# =============================
# backend/app/__init__.py
# =============================
from importlib.metadata import version, PackageNotFoundError
__all__ = ["__version__"]
try:
    __version__ = version("argo-rag")  # optional; safe if packaged
except PackageNotFoundError:
    __version__ = "0.1"