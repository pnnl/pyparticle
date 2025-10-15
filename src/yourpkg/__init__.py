# >>> RELEASE KIT START
"""yourpkg – short description."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("yourpkg")
except PackageNotFoundError:
    __version__ = "0.0.0"
# <<< RELEASE KIT END
