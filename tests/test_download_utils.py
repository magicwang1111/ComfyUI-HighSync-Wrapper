from pathlib import Path

import pytest

from highsync_wrapper.download_utils import validate_required_models


def test_validate_required_models_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_required_models(tmp_path)

    message = str(exc_info.value)
    assert "denoising_unet-500.pth" in message
    assert "reference_unet-500.pth" in message
    assert "audio_processor/whisper_tiny.pt" in message
    assert "sd-vae-ft-mse/config.json" in message
    assert "sd-image-variations-diffusers/unet/config.json" in message


def test_validate_required_models_accepts_expected_layout(tmp_path):
    (tmp_path / "denoising_unet-500.pth").write_bytes(b"")
    (tmp_path / "reference_unet-500.pth").write_bytes(b"")

    audio_dir = tmp_path / "audio_processor"
    audio_dir.mkdir()
    (audio_dir / "whisper_tiny.pt").write_bytes(b"")

    vae_dir = tmp_path / "sd-vae-ft-mse"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text("{}", encoding="utf-8")

    unet_dir = tmp_path / "sd-image-variations-diffusers" / "unet"
    unet_dir.mkdir(parents=True)
    (unet_dir / "config.json").write_text("{}", encoding="utf-8")

    assert validate_required_models(tmp_path) == Path(tmp_path).resolve()
