import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path


WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def is_windows_drive_path(path):
    return isinstance(path, str) and WINDOWS_DRIVE_RE.match(path.strip()) is not None


def normalize_path(path):
    if path is None:
        return path

    raw_path = str(path).strip().strip('"')
    if raw_path == "":
        return raw_path

    match = WINDOWS_DRIVE_RE.match(raw_path)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    return os.path.abspath(os.path.expanduser(raw_path))


def resolve_path(path):
    return Path(normalize_path(path)).expanduser().resolve()


def ensure_exists(path, label="path", is_dir=None):
    if path is None or str(path).strip() == "":
        raise FileNotFoundError(f"{label} is empty.")

    resolved = resolve_path(path)
    if is_dir is True and not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if is_dir is False and not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if is_dir is None and not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def safe_copy(src, dst):
    src_path = ensure_exists(src, "source file", is_dir=False)
    dst_path = resolve_path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() != dst_path.resolve():
        shutil.copy2(src_path, dst_path)
    return dst_path


def get_comfy_output_dir():
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()).resolve()
    except Exception:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "custom_nodes").exists() and (parent / "output").exists():
                return (parent / "output").resolve()
        return Path("/mnt/d/ComfyUI/output").resolve()


def create_task_dir(prefix="highsync"):
    output_root = get_comfy_output_dir() / "highsync"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    task_id = f"{prefix}_{timestamp}_{suffix}"
    task_dir = output_root / task_id

    paths = {
        "task_id": task_id,
        "task_dir": task_dir,
        "media_dir": task_dir / "media",
        "logs_dir": task_dir / "logs",
        "frames_dir": task_dir / "frames",
    }

    for key, value in paths.items():
        if key.endswith("_dir"):
            value.mkdir(parents=True, exist_ok=True)

    return paths
