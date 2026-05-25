"""Run upstream HighSync inference.py on the same local test assets.

This script intentionally calls ``high_sync.inference.inference_process`` instead
of the ComfyUI nodes. It prepares ASCII-only temp paths first because the
upstream script builds a few ffmpeg commands as unquoted shell strings.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_SOURCE_VIDEO = "D:/测试素材/20260522数字人/20260522-103808.mp4"
DEFAULT_DRIVING_AUDIO = "D:/测试素材/20260522数字人/ComfyUI_00009_.mp3"
DEFAULT_MODELS_DIR = "D:/ComfyUI/models/highsync"
DEFAULT_OUTPUT_ROOT = "D:/ComfyUI/output/highsync_official_ab"
DEFAULT_FFMPEG = "D:/ffmpeg/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _as_posix(path):
    return str(Path(path).resolve()).replace("\\", "/")


def _find_ffmpeg(explicit_path=None):
    candidates = [
        explicit_path,
        shutil.which("ffmpeg"),
        shutil.which("ffmpeg.exe"),
        DEFAULT_FFMPEG,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("ffmpeg.exe was not found. Pass --ffmpeg D:/path/to/ffmpeg.exe.")


def _run(command, cwd=None, env=None):
    print("[official-ab]", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _validate_models(models_dir):
    models_dir = Path(models_dir)
    required = [
        models_dir / "denoising_unet-500.pth",
        models_dir / "reference_unet-500.pth",
        models_dir / "sd-vae-ft-mse" / "config.json",
        models_dir / "sd-image-variations-diffusers" / "unet" / "config.json",
        models_dir / "audio_processor" / "whisper_tiny.pt",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing HighSync model files:\n{formatted}")
    return models_dir.resolve()


def _prepare_config(repo_root, work_dir, models_dir, inference_steps, cfg_scale):
    cfg = OmegaConf.load(repo_root / "high_sync" / "configs" / "inference" / "default.yaml")

    cfg.mask_path = _as_posix(repo_root / "high_sync" / "masks" / "mask_64.png")
    cfg.mask_512_path = _as_posix(repo_root / "high_sync" / "masks" / "mask_512.png")
    cfg.denoising_unet_path = _as_posix(models_dir / "denoising_unet-500.pth")
    cfg.reference_unet_path = _as_posix(models_dir / "reference_unet-500.pth")
    cfg.base_model_path = _as_posix(models_dir / "sd-image-variations-diffusers")
    cfg.audio_model_path = _as_posix(models_dir / "audio_processor" / "whisper_tiny.pt")
    cfg.vae.model_path = _as_posix(models_dir / "sd-vae-ft-mse")
    cfg.save_path = "cache"
    cfg.inference_steps = int(inference_steps)
    cfg.cfg_scale = float(cfg_scale)

    config_path = work_dir / "official_config.yaml"
    OmegaConf.save(cfg, config_path)
    return config_path


def _prepare_inputs(ffmpeg, source_video, driving_audio, work_dir, max_frames):
    source_video = Path(source_video)
    driving_audio = Path(driving_audio)
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    if not driving_audio.exists():
        raise FileNotFoundError(f"Driving audio not found: {driving_audio}")

    prepared_video = work_dir / "input_25fps.mp4"
    prepared_audio = work_dir / "input_audio_16k.wav"

    video_command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_video),
        "-an",
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
    ]
    if max_frames > 0:
        video_command.extend(["-frames:v", str(max_frames)])
    video_command.append(str(prepared_video))
    _run(video_command)

    audio_command = [
        ffmpeg,
        "-y",
        "-i",
        str(driving_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if max_frames > 0:
        audio_command.extend(["-t", f"{max_frames / 25.0:.3f}"])
    audio_command.extend(["-c:a", "pcm_s16le", str(prepared_audio)])
    _run(audio_command)

    return prepared_video, prepared_audio


def _build_env(repo_root, ffmpeg):
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    env["PATH"] = f"{Path(ffmpeg).parent}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    return env


def _patched_process_audio(input_audio_path, output_audio_path):
    import numpy as np
    import soundfile as sf
    import torch
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pretrained.dns64().to(device)
    model.eval()

    audio_np, sample_rate = sf.read(str(input_audio_path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(audio_np.T)).to(device)
    wav = convert_audio(wav, sample_rate, model.sample_rate, model.chin)

    with torch.no_grad():
        denoised = model(wav[None])[0]

    sf.write(
        str(output_audio_path),
        denoised.detach().cpu().clamp(-1.0, 1.0).numpy()[0],
        int(model.sample_rate),
        subtype="PCM_16",
    )


def _run_official_in_process(repo_root, work_dir, ffmpeg, config_path, prepared_video, prepared_audio, output, inference_frames_num):
    import argparse as argparse_module

    sys.path.insert(0, str(repo_root))
    os.environ["PATH"] = f"{Path(ffmpeg).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    from high_sync import inference as official_inference

    official_inference.process_audio = _patched_process_audio

    previous_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        official_args = argparse_module.Namespace(
            config=str(config_path),
            source_video=prepared_video.name,
            driving_audio=prepared_audio.name,
            output=output,
            inference_frames_num=int(inference_frames_num),
        )
        official_inference.inference_process(official_args)
    finally:
        os.chdir(previous_cwd)


def main():
    parser = argparse.ArgumentParser(description="Run official HighSync inference.py for ComfyUI wrapper A/B testing.")
    parser.add_argument("--source-video", default=DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--driving-audio", default=DEFAULT_DRIVING_AUDIO)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--inference-frames-num", type=int, default=1500)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=3.5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video; use a small number for smoke tests.")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare converted inputs and config, do not run inference.")
    parser.add_argument(
        "--pure-subprocess",
        action="store_true",
        help="Run python -m high_sync.inference exactly. On this Windows env it currently fails in torchaudio/TorchCodec.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    ffmpeg = _find_ffmpeg(args.ffmpeg)
    models_dir = _validate_models(args.models_dir)

    task_id = datetime.now().strftime("official_%Y%m%d_%H%M%S")
    work_dir = Path(args.output_root) / task_id
    (work_dir / "out").mkdir(parents=True, exist_ok=True)
    (work_dir / "cache").mkdir(parents=True, exist_ok=True)

    prepared_video, prepared_audio = _prepare_inputs(
        ffmpeg=ffmpeg,
        source_video=args.source_video,
        driving_audio=args.driving_audio,
        work_dir=work_dir,
        max_frames=max(0, int(args.max_frames)),
    )
    config_path = _prepare_config(
        repo_root=repo_root,
        work_dir=work_dir,
        models_dir=models_dir,
        inference_steps=args.inference_steps,
        cfg_scale=args.cfg_scale,
    )

    final_output = work_dir / "out" / "final.mp4"
    print(f"[official-ab] work_dir={work_dir}", flush=True)
    print(f"[official-ab] config={config_path}", flush=True)
    print(f"[official-ab] prepared_video={prepared_video}", flush=True)
    print(f"[official-ab] prepared_audio={prepared_audio}", flush=True)
    print(f"[official-ab] final_output={final_output}", flush=True)

    if args.prepare_only:
        print("[official-ab] prepare-only requested; inference was not started.", flush=True)
        return

    if args.pure_subprocess:
        command = [
            sys.executable,
            "-m",
            "high_sync.inference",
            "-c",
            str(config_path),
            "--source_video",
            prepared_video.name,
            "--driving_audio",
            prepared_audio.name,
            "--output",
            "out/final.mp4",
            "--inference_frames_num",
            str(args.inference_frames_num),
        ]
        _run(command, cwd=work_dir, env=_build_env(repo_root, ffmpeg))
    else:
        print(
            "[official-ab] running official inference_process with only process_audio patched "
            "to avoid Windows torchaudio/TorchCodec audio loading.",
            flush=True,
        )
        _run_official_in_process(
            repo_root=repo_root,
            work_dir=work_dir,
            ffmpeg=ffmpeg,
            config_path=config_path,
            prepared_video=prepared_video,
            prepared_audio=prepared_audio,
            output="out/final.mp4",
            inference_frames_num=args.inference_frames_num,
        )

    if not final_output.exists():
        raise FileNotFoundError(f"Official inference finished but output was not created: {final_output}")
    print(f"[official-ab] done: {final_output}", flush=True)


if __name__ == "__main__":
    main()
