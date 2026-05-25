try:
    from .highsync_wrapper.download_utils import download_or_check_models
except ImportError:
    from highsync_wrapper.download_utils import download_or_check_models


DEFAULT_MODELS_DIR = "D:/ComfyUI/models/highsync"


class HighSyncDownloadModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "download": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("models_dir",)
    FUNCTION = "run"
    CATEGORY = "video/lipsync"

    def run(self, download):
        models_dir = download_or_check_models(
            models_dir=DEFAULT_MODELS_DIR,
            repo_id="saeed-5959/high_sync",
            download=download,
        )
        return (models_dir,)


class HighSyncModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models_dir": ("STRING", {"default": DEFAULT_MODELS_DIR, "multiline": False}),
                "weight_dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("HIGHSYNC_MODEL",)
    RETURN_NAMES = ("highsync_model",)
    FUNCTION = "load"
    CATEGORY = "video/lipsync"

    def load(self, models_dir, weight_dtype):
        try:
            from .highsync_wrapper.model_utils import load_highsync_model
        except ImportError:
            from highsync_wrapper.model_utils import load_highsync_model

        model = load_highsync_model(models_dir=models_dir, weight_dtype=weight_dtype)
        return (model,)


class HighSyncLipSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "highsync_model": ("HIGHSYNC_MODEL",),
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "input_fps": ("INT", {"default": 25, "min": 1, "max": 120, "step": 1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "frame_rate", "output_video_path")
    FUNCTION = "run"
    CATEGORY = "video/lipsync"

    def run(self, highsync_model, images, audio, input_fps, max_frames):
        try:
            from .highsync_wrapper.runner import run_highsync
        except ImportError:
            from highsync_wrapper.runner import run_highsync

        return run_highsync(
            highsync_model=highsync_model,
            images=images,
            audio=audio,
            input_fps=input_fps,
            max_frames=max_frames,
            inference_steps=None,
            cfg_scale=None,
            seed=42,
            preprocess_chunk_frames=1500,
            denoise_audio=False,
            trim_audio_to_video=True,
            keep_intermediate_outputs=True,
        )


class HighSyncLipSyncAdvanced(HighSyncLipSync):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "highsync_model": ("HIGHSYNC_MODEL",),
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "input_fps": ("INT", {"default": 25, "min": 1, "max": 120, "step": 1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "inference_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "preprocess_chunk_frames": ("INT", {"default": 1500, "min": 12, "max": 10000, "step": 12}),
                "denoise_audio": ("BOOLEAN", {"default": False}),
                "trim_audio_to_video": ("BOOLEAN", {"default": True}),
                "keep_intermediate_outputs": ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "run_advanced"
    CATEGORY = "video/lipsync"

    def run_advanced(
        self,
        highsync_model,
        images,
        audio,
        input_fps,
        max_frames,
        inference_steps,
        cfg_scale,
        seed,
        preprocess_chunk_frames,
        denoise_audio,
        trim_audio_to_video,
        keep_intermediate_outputs,
    ):
        try:
            from .highsync_wrapper.runner import run_highsync
        except ImportError:
            from highsync_wrapper.runner import run_highsync

        return run_highsync(
            highsync_model=highsync_model,
            images=images,
            audio=audio,
            input_fps=input_fps,
            max_frames=max_frames,
            inference_steps=inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            preprocess_chunk_frames=preprocess_chunk_frames,
            denoise_audio=denoise_audio,
            trim_audio_to_video=trim_audio_to_video,
            keep_intermediate_outputs=keep_intermediate_outputs,
        )


NODE_CLASS_MAPPINGS = {
    "HighSyncDownloadModels": HighSyncDownloadModels,
    "HighSyncModelLoader": HighSyncModelLoader,
    "HighSyncLipSync": HighSyncLipSync,
    "HighSyncLipSyncAdvanced": HighSyncLipSyncAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HighSyncDownloadModels": "HighSync Download Models",
    "HighSyncModelLoader": "HighSync Model Loader",
    "HighSyncLipSync": "HighSync LipSync",
    "HighSyncLipSyncAdvanced": "HighSync LipSync Advanced",
}
