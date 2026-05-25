from pathlib import Path

from .file_utils import normalize_path, resolve_path


CORE_FILES = (
    "denoising_unet-500.pth",
    "reference_unet-500.pth",
    "audio_processor/whisper_tiny.pt",
)

ALLOW_PATTERNS = (
    "denoising_unet-500.pth",
    "reference_unet-500.pth",
    "audio_processor/whisper_tiny.pt",
    "sd-vae-ft-mse/**",
    "sd-image-variations-diffusers/**",
)


def _directory_has_model_files(path):
    if not path.is_dir():
        return False
    return any(candidate.is_file() for candidate in path.rglob("*"))


def validate_required_models(models_dir):
    root = resolve_path(models_dir)
    missing = []

    for rel_path in CORE_FILES:
        if not (root / rel_path).is_file():
            missing.append(rel_path)

    vae_dir = root / "sd-vae-ft-mse"
    if not _directory_has_model_files(vae_dir) or not (vae_dir / "config.json").is_file():
        missing.append("sd-vae-ft-mse/config.json")

    base_dir = root / "sd-image-variations-diffusers"
    if not _directory_has_model_files(base_dir) or not (base_dir / "unet" / "config.json").is_file():
        missing.append("sd-image-variations-diffusers/unet/config.json")

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"HighSync model files are missing in {root}: {missing_text}. "
            "Run the HighSync Download Models node with download=True, or download "
            "https://huggingface.co/saeed-5959/high_sync into this folder."
        )

    return root


def download_or_check_models(models_dir, repo_id="saeed-5959/high_sync", download=False):
    root = Path(normalize_path(models_dir))

    if download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for HighSync Download Models. "
                "Install this wrapper's requirements.txt first."
            ) from exc

        root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(root),
            allow_patterns=list(ALLOW_PATTERNS),
            repo_type="model",
        )
    elif not root.exists():
        raise FileNotFoundError(
            f"models_dir does not exist: {root}. "
            "Run HighSync Download Models with download=True or download the models manually."
        )

    return str(validate_required_models(root))
