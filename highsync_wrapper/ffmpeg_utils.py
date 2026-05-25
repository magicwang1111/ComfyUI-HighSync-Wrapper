import shutil
import subprocess

from .file_utils import resolve_path


def check_ffmpeg():
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and add it to PATH.")
    return ffmpeg


def run_ffmpeg(args, error_label):
    command = [check_ffmpeg()] + [str(arg) for arg in args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"{error_label} failed with exit code {result.returncode}: {stderr}")


def mux_video_audio(video_path, audio_path, output_path):
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            resolve_path(video_path),
            "-i",
            resolve_path(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_file,
        ],
        "audio/video mux",
    )
    return output_file
