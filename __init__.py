import sys
from pathlib import Path


_NODE_ROOT = Path(__file__).resolve().parent
_NODE_ROOT_TEXT = str(_NODE_ROOT)
if _NODE_ROOT_TEXT not in sys.path:
    sys.path.insert(0, _NODE_ROOT_TEXT)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
