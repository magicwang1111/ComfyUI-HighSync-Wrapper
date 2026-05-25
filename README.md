# ComfyUI HighSync Wrapper

HighSync lip-sync nodes for ComfyUI. This wrapper uses ComfyUI `IMAGE` frames plus native `AUDIO`, runs the HighSync pipeline in-process, and returns processed frames, audio, output fps, and an encoded mp4 path.

## Nodes

- `HighSync Download Models`: checks or downloads the required model files into `D:/ComfyUI/models/highsync`.
- `HighSync Model Loader`: loads and caches the VAE, UNets, scheduler, and Whisper audio processor.
- `HighSync LipSync`: simple node for `IMAGE + AUDIO` input.
- `HighSync LipSync Advanced`: exposes inference steps, CFG scale, seed, chunk size, denoising, trimming, and intermediate-output options.

## Model Layout

The default model directory is:

```text
D:/ComfyUI/models/highsync
```

It must contain:

```text
denoising_unet-500.pth
reference_unet-500.pth
sd-vae-ft-mse/
sd-image-variations-diffusers/
audio_processor/whisper_tiny.pt
```

Use `HighSync Download Models` with `download=True`, or download `saeed-5959/high_sync` from Hugging Face into that folder.

## Usage

1. Install `requirements.txt` into the same Python environment that runs ComfyUI.
2. Restart ComfyUI.
3. Use VideoHelperSuite or another loader to provide video frames as `IMAGE` and audio as `AUDIO`.
4. Connect `HighSync Model Loader` to `HighSync LipSync`.

The wrapper converts video frames to 25fps and audio to mono 16kHz for HighSync inference. The returned frame rate is always `25` in this v1 implementation.

## Notes

- `max_frames=0` means process all available frames after 25fps conversion.
- `denoise_audio=False` is the default because the official `denoiser` package pins older Hydra/OmegaConf versions than current ComfyUI environments. To enable it, install the package without pulling its old dependencies: `pip install --no-deps denoiser==0.1.5`.
- When `denoise_audio=True`, the official denoiser is loaded lazily and cached. The final mp4 still uses the adjusted original audio track to preserve voice quality.
- DeepFace/TensorFlow are optional. The runtime tries DeepFace if it is already installed, otherwise it falls back to OpenCV Haar face detection.
- Outputs are written under `ComfyUI/output/highsync/<task_id>/final.mp4`.
