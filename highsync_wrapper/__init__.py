"""ComfyUI adapter utilities for HighSync."""

import sys
from pathlib import Path


def ensure_node_root_on_path():
    node_root = Path(__file__).resolve().parents[1]
    node_root_text = str(node_root)
    if node_root_text not in sys.path:
        sys.path.insert(0, node_root_text)
    return node_root


ensure_node_root_on_path()
