from pathlib import Path

from .download_utils import validate_required_models


_MODEL_CACHE = {}


def _get_repo_root():
    return Path(__file__).resolve().parents[1]


def _torch_load(path, torch):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_torch_device(torch):
    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _weight_dtype(torch, weight_dtype):
    dtype_text = str(weight_dtype or "fp16").lower()
    if dtype_text == "fp16":
        return torch.float16
    if dtype_text == "bf16":
        return torch.bfloat16
    if dtype_text == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported HighSync weight_dtype: {weight_dtype}")


def _load_config(models_root, weight_dtype):
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("omegaconf is required to load HighSync config. Install requirements.txt first.") from exc

    config_path = _get_repo_root() / "high_sync" / "configs" / "inference" / "default.yaml"
    config = OmegaConf.load(str(config_path))

    config.weight_dtype = str(weight_dtype)
    config.denoising_unet_path = str(models_root / "denoising_unet-500.pth")
    config.reference_unet_path = str(models_root / "reference_unet-500.pth")
    config.base_model_path = str(models_root / "sd-image-variations-diffusers")
    config.audio_model_path = str(models_root / "audio_processor" / "whisper_tiny.pt")
    config.vae.model_path = str(models_root / "sd-vae-ft-mse")
    config.mask_path = str(_get_repo_root() / "high_sync" / "masks" / "mask_64.png")
    config.mask_512_path = str(_get_repo_root() / "high_sync" / "masks" / "mask_512.png")

    return config


def _build_scheduler(config):
    try:
        from diffusers import DDIMScheduler
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("diffusers and omegaconf are required for HighSync. Install requirements.txt first.") from exc

    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs, resolve=True)
    if bool(config.enable_zero_snr):
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    return DDIMScheduler(**sched_kwargs)


def load_highsync_model(models_dir, weight_dtype="fp16"):
    try:
        import torch
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError(
            "torch and diffusers are required to load HighSync. Install this wrapper's requirements.txt first."
        ) from exc

    models_root = validate_required_models(models_dir)
    device = _get_torch_device(torch)
    dtype = _weight_dtype(torch, weight_dtype)
    cache_key = (str(models_root), str(weight_dtype).lower(), str(device))

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    config = _load_config(models_root, weight_dtype)

    from omegaconf import OmegaConf

    from high_sync.image_processor import ImageProcessor
    from high_sync.src.models.unet_2d_condition import UNet2DConditionModel
    from high_sync.src.models.unet_3d_echo import EchoUNet3DConditionModel
    from high_sync.src.models.whisper.audio2feature import load_audio_model
    from high_sync.src.pipelines.face_animate import FaceAnimatePipeline

    scheduler = _build_scheduler(config)

    vae = AutoencoderKL.from_pretrained(str(config.vae.model_path))
    reference_unet = UNet2DConditionModel.from_pretrained(str(config.base_model_path), subfolder="unet")
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        str(config.base_model_path),
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs, resolve=True),
    )

    reference_unet.load_state_dict(_torch_load(str(config.reference_unet_path), torch))
    denoising_unet.load_state_dict(_torch_load(str(config.denoising_unet_path), torch))

    vae.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)

    vae.eval()
    reference_unet.eval()
    denoising_unet.eval()

    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    )
    pipeline.to(device=device, dtype=dtype)

    audio_processor = load_audio_model(model_path=str(config.audio_model_path), device=device)

    model = {
        "models_dir": str(models_root),
        "device": device,
        "weight_dtype": dtype,
        "config": config,
        "pipeline": pipeline,
        "audio_processor": audio_processor,
        "image_processor_cls": ImageProcessor,
        "denoiser_model": None,
    }
    _MODEL_CACHE[cache_key] = model
    return model
