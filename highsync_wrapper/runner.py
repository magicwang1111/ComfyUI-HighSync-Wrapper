import shutil

import cv2
import numpy as np
import torch

from .ffmpeg_utils import mux_video_audio
from .file_utils import create_task_dir
from .media_utils import (
    TARGET_AUDIO_SR,
    TARGET_FPS,
    denoise_audio_file,
    encode_rgb_frames_to_mp4,
    frames_uint8_rgb_to_image_tensor,
    images_to_uint8_rgb,
    resample_frames_to_fps,
    save_audio_to_wav,
    trim_or_pad_audio,
)


def _maybe_throw_if_interrupted():
    try:
        import comfy.model_management as model_management
    except Exception:
        return
    model_management.throw_exception_if_processing_interrupted()


def _empty_cache(device):
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except Exception:
        return


def create_mask(mask_path):
    raw_mask = cv2.imread(str(mask_path))
    if raw_mask is None:
        raise FileNotFoundError(f"HighSync mask could not be read: {mask_path}")
    mask = 1 - (raw_mask / 255.0)

    h, w, c = mask.shape
    new_mask = np.zeros((h, w, c + 1), dtype=np.float32)
    new_mask[:, :, :c] = mask
    new_mask[:, :, c] = mask[:, :, c - 1]
    return torch.FloatTensor(new_mask).permute(2, 0, 1).unsqueeze(1)


def create_mask_alpha(width, height, coordinate):
    left, top, right, bottom = coordinate
    mask = np.ones((height, width, 3), dtype=np.float32)
    mask[top:bottom, left:right] = 0

    pad = min(int((right - left) / 15), int((bottom - top) / 15))
    if pad <= 0:
        return mask

    for idx in range(left, right):
        mask[bottom - pad:bottom, idx] = np.array([np.linspace(0, 1, pad)] * 3).transpose(1, 0)
        mask[top:top + pad, idx] = np.array([np.linspace(1, 0, pad)] * 3).transpose(1, 0)

    for idx in range(top, bottom):
        mask[idx, right - pad:right] = np.array([np.linspace(0, 1, pad)] * 3).transpose(1, 0)
        mask[idx, left:left + pad] = np.array([np.linspace(1, 0, pad)] * 3).transpose(1, 0)

    for idx, start, end in zip(range(top, top + pad), np.linspace(1, 0.7, pad), np.linspace(0.7, 0, pad)):
        mask[idx, left:left + pad] = np.array([np.linspace(start, end, pad)] * 3).transpose(1, 0)
        mask[idx, right - pad:right] = np.array([np.linspace(end, start, pad)] * 3).transpose(1, 0)

    for idx, start, end in zip(range(bottom - pad, bottom), np.linspace(0.7, 1, pad), np.linspace(0, 0.7, pad)):
        mask[idx, left:left + pad] = np.array([np.linspace(start, end, pad)] * 3).transpose(1, 0)
        mask[idx, right - pad:right] = np.array([np.linspace(end, start, pad)] * 3).transpose(1, 0)

    return mask


def _prepare_audio_files(task, audio, duration_sec, trim_audio_to_video, denoise_audio, device, model_cache):
    audio_for_video = trim_or_pad_audio(
        audio,
        duration_sec=duration_sec if trim_audio_to_video else None,
        pad_to_duration=bool(trim_audio_to_video),
    )

    mux_audio_path = save_audio_to_wav(
        audio_for_video,
        task["media_dir"] / "mux_audio.wav",
    )
    feature_audio_path = save_audio_to_wav(
        audio_for_video,
        task["media_dir"] / "feature_audio_16k.wav",
        mono=True,
        sample_rate=TARGET_AUDIO_SR,
    )

    if denoise_audio:
        feature_audio_path = denoise_audio_file(
            feature_audio_path,
            task["media_dir"] / "feature_audio_16k_denoised.wav",
            device=device,
            cache=model_cache,
        )

    return audio_for_video, mux_audio_path, feature_audio_path


def _ensure_highsync_model(highsync_model):
    required = ("config", "pipeline", "audio_processor", "image_processor_cls", "device", "weight_dtype")
    if not isinstance(highsync_model, dict) or any(key not in highsync_model for key in required):
        raise ValueError("highsync_model must come from the HighSync Model Loader node.")


def _append_original_frame(output_frames_rgb, frame_bgr):
    output_frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _run_generation_chunk(
    pipeline,
    image_processor,
    config,
    audio_fea_final,
    frames_video,
    output_frames_rgb,
    count_img,
    generator,
    device,
    weight_dtype,
    inference_steps,
    cfg_scale,
):
    source_image_pixels_all, bbox_all, frame_list_all = image_processor.preprocess_frames_jump(frames_video)

    img_size = (int(config.data.source_image.width), int(config.data.source_image.height))
    sample_frames = int(config.data.n_sample_frames)
    mask_path = str(config.mask_path)

    for source_image_pixels_list, bbox_list, frame_list in zip(source_image_pixels_all, bbox_all, frame_list_all):
        _maybe_throw_if_interrupted()

        if not frame_list:
            continue

        if bbox_list[0] != (0, 0, 0, 0):
            audio_tensor = audio_fea_final[:, count_img: count_img + len(frame_list)]
            main_length = len(frame_list)
            diff = sample_frames - main_length

            if diff > 0:
                source_image_pixels_list.extend([source_image_pixels_list[-1]] * diff)
                audio_tensor = torch.cat((audio_tensor, audio_tensor[:, -1:].repeat(1, diff, 1, 1)), dim=1)

            source_image_pixels = torch.cat(source_image_pixels_list, dim=0)
            mask_main = create_mask(mask_path).unsqueeze(0).to(device=device, dtype=weight_dtype)
            _empty_cache(device)

            with torch.no_grad():
                pipeline_output = pipeline(
                    gt_encode=source_image_pixels,
                    audio_tensor=audio_tensor,
                    mask_main=mask_main,
                    width=img_size[0],
                    height=img_size[1],
                    video_length=len(source_image_pixels_list),
                    num_inference_steps=int(inference_steps),
                    guidance_scale=float(cfg_scale),
                    generator=generator,
                )

            _empty_cache(device)

            images_out = pipeline_output.videos.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()[:main_length]
            images_out = np.clip(images_out * 255, 0, 255).astype(np.uint8)

            for image_out, bbox, frame in zip(images_out, bbox_list, frame_list):
                frame_h, frame_w, _ = frame.shape
                if bbox == (0, 0, 0, 0):
                    _append_original_frame(output_frames_rgb, frame)
                else:
                    image_out_bgr = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
                    frame_in_out = frame.copy()
                    frame_in_out[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cv2.resize(
                        image_out_bgr,
                        (bbox[2] - bbox[0], bbox[3] - bbox[1]),
                    )
                    mask = create_mask_alpha(frame_w, frame_h, bbox)
                    blended = (frame * mask + frame_in_out * (1 - mask)).astype(np.uint8)
                    _append_original_frame(output_frames_rgb, blended)

                count_img += 1
        else:
            for frame in frame_list:
                _append_original_frame(output_frames_rgb, frame)
                count_img += 1

    return count_img


def run_highsync(
    highsync_model,
    images,
    audio,
    input_fps=25,
    max_frames=0,
    inference_steps=None,
    cfg_scale=None,
    seed=42,
    preprocess_chunk_frames=1500,
    denoise_audio=True,
    trim_audio_to_video=True,
    keep_intermediate_outputs=True,
):
    _ensure_highsync_model(highsync_model)

    config = highsync_model["config"]
    pipeline = highsync_model["pipeline"]
    audio_processor = highsync_model["audio_processor"]
    image_processor = highsync_model["image_processor_cls"]((int(config.data.source_image.width), int(config.data.source_image.height)))
    device = torch.device(highsync_model["device"])
    weight_dtype = highsync_model["weight_dtype"]

    inference_steps = int(inference_steps if inference_steps is not None else config.inference_steps)
    cfg_scale = float(cfg_scale if cfg_scale is not None else config.cfg_scale)
    preprocess_chunk_frames = max(12, int(preprocess_chunk_frames))

    task = create_task_dir()

    frames_rgb = images_to_uint8_rgb(images)
    frames_rgb = resample_frames_to_fps(frames_rgb, input_fps=input_fps, output_fps=TARGET_FPS, max_frames=max_frames)
    frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames_rgb]
    initial_duration_sec = len(frames_bgr) / float(TARGET_FPS)

    _, _, feature_audio_path = _prepare_audio_files(
        task=task,
        audio=audio,
        duration_sec=initial_duration_sec,
        trim_audio_to_video=trim_audio_to_video,
        denoise_audio=denoise_audio,
        device=device,
        model_cache=highsync_model,
    )

    whisper_feature = audio_processor.audio2feat(str(feature_audio_path))
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=TARGET_FPS)
    audio_frame_num = int(whisper_chunks.shape[0])
    video_length = min(len(frames_bgr), audio_frame_num)
    if video_length <= 0:
        raise RuntimeError("HighSync could not align any video frames with the provided audio.")

    frames_bgr = frames_bgr[:video_length]
    audio_fea_final = torch.as_tensor(whisper_chunks[:video_length], dtype=pipeline.vae.dtype, device=pipeline.vae.device)
    audio_fea_final = audio_fea_final.unsqueeze(0)

    final_duration_sec = video_length / float(TARGET_FPS)
    output_audio, mux_audio_path, _ = _prepare_audio_files(
        task=task,
        audio=audio,
        duration_sec=final_duration_sec,
        trim_audio_to_video=trim_audio_to_video,
        denoise_audio=False,
        device=device,
        model_cache=highsync_model,
    )

    torch.manual_seed(int(seed))
    generator = torch.manual_seed(int(seed))

    output_frames_rgb = []
    frames_video = []
    count_motion = 0
    count_img = 0

    while True:
        _maybe_throw_if_interrupted()

        if count_motion >= video_length:
            finish = True
        else:
            frames_video.append(frames_bgr[count_motion])
            count_motion += 1
            finish = False

        if len(frames_video) == 0:
            break

        if (count_motion % preprocess_chunk_frames == 0) or finish:
            print(f"[HighSync] processing frames {count_img}-{count_motion} / {video_length}")
            count_img = _run_generation_chunk(
                pipeline=pipeline,
                image_processor=image_processor,
                config=config,
                audio_fea_final=audio_fea_final,
                frames_video=frames_video,
                output_frames_rgb=output_frames_rgb,
                count_img=count_img,
                generator=generator,
                device=device,
                weight_dtype=weight_dtype,
                inference_steps=inference_steps,
                cfg_scale=cfg_scale,
            )
            frames_video = []

        if finish:
            break

    if len(output_frames_rgb) == 0:
        raise RuntimeError("HighSync finished without producing output frames.")

    output_frames = np.ascontiguousarray(np.stack(output_frames_rgb))
    no_audio_path = encode_rgb_frames_to_mp4(output_frames, task["media_dir"] / "highsync_no_audio.mp4", fps=TARGET_FPS)
    final_video_path = mux_video_audio(no_audio_path, mux_audio_path, task["task_dir"] / "final.mp4")

    output_images = frames_uint8_rgb_to_image_tensor(output_frames)

    if not keep_intermediate_outputs:
        for path in (task["media_dir"], task["frames_dir"], task["logs_dir"]):
            shutil.rmtree(path, ignore_errors=True)

    return output_images, output_audio, TARGET_FPS, str(final_video_path.resolve())
