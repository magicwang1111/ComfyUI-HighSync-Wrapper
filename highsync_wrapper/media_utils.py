import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .ffmpeg_utils import check_ffmpeg
from .file_utils import resolve_path


TARGET_FPS = 25
TARGET_AUDIO_SR = 16000


def _as_image_tensor(images):
    if isinstance(images, list):
        images = torch.stack(images)
    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)
    if images.dim() != 4:
        raise ValueError(f"Expected IMAGE tensor with shape [frames, height, width, channels], got {tuple(images.shape)}")
    if images.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE tensor with at least 3 channels, got {tuple(images.shape)}")
    return images.detach().cpu()


def images_to_uint8_rgb(images):
    image_tensor = _as_image_tensor(images)[..., :3]
    if image_tensor.dtype != torch.uint8:
        image_tensor = image_tensor.float()
        if image_tensor.numel() > 0 and float(image_tensor.max()) <= 1.0:
            image_tensor = image_tensor * 255.0
        image_tensor = image_tensor.clamp(0, 255).byte()
    return np.ascontiguousarray(image_tensor.numpy())


def frames_uint8_rgb_to_image_tensor(frames):
    if len(frames) == 0:
        raise ValueError("No output frames were generated.")
    array = np.ascontiguousarray(np.stack(frames).astype(np.float32) / 255.0)
    return torch.from_numpy(array)


def resample_frames_to_fps(frames_rgb, input_fps, output_fps=TARGET_FPS, max_frames=0):
    input_fps = int(input_fps)
    output_fps = int(output_fps)
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError("input_fps and output_fps must be positive.")
    if len(frames_rgb) == 0:
        raise ValueError("No IMAGE frames were provided.")

    if input_fps == output_fps:
        sampled = frames_rgb.copy()
    else:
        duration = len(frames_rgb) / float(input_fps)
        target_count = max(1, int(round(duration * output_fps)))
        indices = np.floor(np.arange(target_count) * input_fps / output_fps).astype(np.int64)
        indices = np.clip(indices, 0, len(frames_rgb) - 1)
        sampled = frames_rgb[indices]

    max_frames = int(max_frames or 0)
    if max_frames > 0:
        sampled = sampled[:max_frames]

    return np.ascontiguousarray(sampled)


def _audio_waveform_to_tensor(audio):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("AUDIO input must be a dict with 'waveform' and 'sample_rate'.")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().cpu().float()

    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 2:
        raise ValueError(
            "Expected AUDIO waveform with shape [batch, channels, samples], "
            f"[channels, samples], or [samples], got {tuple(waveform.shape)}"
        )

    return waveform.contiguous(), int(audio["sample_rate"])


def trim_or_pad_audio(audio, duration_sec=None, pad_to_duration=False):
    waveform, sample_rate = _audio_waveform_to_tensor(audio)

    if duration_sec is not None and duration_sec > 0:
        target_samples = max(1, int(round(float(duration_sec) * sample_rate)))
        if waveform.shape[-1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif pad_to_duration and waveform.shape[-1] < target_samples:
            pad = torch.zeros((waveform.shape[0], target_samples - waveform.shape[-1]), dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=-1)

    return {"waveform": waveform.unsqueeze(0).contiguous(), "sample_rate": sample_rate}


def _resample_waveform(waveform, sample_rate, target_sample_rate):
    if int(sample_rate) == int(target_sample_rate):
        return waveform

    try:
        import torchaudio

        resampler = torchaudio.transforms.Resample(
            orig_freq=int(sample_rate),
            new_freq=int(target_sample_rate),
        )
        return resampler(waveform)
    except Exception as exc:
        raise RuntimeError(
            "Failed to resample audio with torchaudio. Install torchaudio or provide 16kHz audio."
        ) from exc


def save_audio_to_wav(audio, output_path, duration_sec=None, pad_to_duration=False, mono=False, sample_rate=None):
    adjusted = trim_or_pad_audio(audio, duration_sec=duration_sec, pad_to_duration=pad_to_duration)
    waveform = adjusted["waveform"].squeeze(0).clamp(-1.0, 1.0)
    source_sample_rate = int(adjusted["sample_rate"])

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sample_rate = int(sample_rate or source_sample_rate)
    waveform = _resample_waveform(waveform, source_sample_rate, target_sample_rate)

    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    audio_np = waveform.transpose(0, 1).numpy()
    sf.write(str(output_file), audio_np, target_sample_rate, subtype="PCM_16")
    return output_file


def encode_rgb_frames_to_mp4(frames_rgb, output_path, fps=TARGET_FPS):
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    frames = np.ascontiguousarray(frames_rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape [frames, height, width, 3], got {frames.shape}")

    frame_count, height, width, _ = frames.shape
    if frame_count <= 0:
        raise ValueError("No frames were provided to encode.")

    command = [
        check_ffmpeg(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_file),
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        _, stderr = process.communicate()
    except BrokenPipeError:
        _, stderr = process.communicate()
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()

    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to encode HighSync output video: {message}")

    return output_file


def denoise_audio_file(input_audio_path, output_audio_path, device, cache):
    try:
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
    except ImportError as exc:
        raise RuntimeError(
            "denoiser and torchaudio are required when denoise_audio=True. "
            "Install this wrapper's requirements.txt or disable denoise_audio."
        ) from exc

    model = cache.get("denoiser_model")
    model_device = torch.device(device)
    if model is None:
        model = pretrained.dns64().to(model_device)
        model.eval()
        cache["denoiser_model"] = model

    audio_np, sample_rate = sf.read(str(Path(input_audio_path)), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio_np.T.copy())
    wav = convert_audio(wav.to(model_device), sample_rate, model.sample_rate, model.chin)

    with torch.no_grad():
        denoised = model(wav[None])[0]

    output_file = resolve_path(output_audio_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    audio_np = denoised.detach().cpu().clamp(-1.0, 1.0).numpy()[0]
    sf.write(str(output_file), audio_np, int(model.sample_rate), subtype="PCM_16")
    return output_file
