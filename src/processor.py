import os
# Patch for MoviePy 1.0.3 compatibility with newer Pillow versions
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Direct imports to avoid broken 'moviepy.editor' / 'Freeze' module on RunPod
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx.loop import loop

import subprocess
import tempfile

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "output")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def script_to_srt(script: str, duration_sec: float) -> str:
    """Split script into sentences and build SRT with evenly distributed timings."""
    import re
    sentences = [s.strip() for s in re.split(r"[.!?]+", script) if s.strip()]
    if not sentences:
        sentences = [script[:200] if script else " "]
    n = len(sentences)
    srt_lines = []
    for i, line in enumerate(sentences):
        start = (i / n) * duration_sec
        end = ((i + 1) / n) * duration_sec
        srt_lines.append(f"{i + 1}\n{_sec_to_srt(start)} --> {_sec_to_srt(end)}\n{line}\n")
    return "\n".join(srt_lines)


def _sec_to_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_whisper_srt(audio_path, model_size="base"):
    """Transcribe audio using OpenAI Whisper to generate accurate SRT subtitles."""
    import whisper
    # Load model (cached by whisper)
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    srt_lines = []
    for i, seg in enumerate(result['segments']):
        start = _sec_to_srt(seg['start'])
        end = _sec_to_srt(seg['end'])
        text = seg['text'].strip()
        srt_lines.append(f"{i + 1}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)


def trim_final_video(segments, audio_path=None, output_path=None, segment_length_sec=15, subtitle_text=None, 
                     use_gpu=False, crop_headers=True, mute_original=True, fade_duration=0.5):
    """Stitches video segments. Optional: audio_path (voiceover), subtitle_text (burn-in subtitles). High quality: libx264, 8000k."""
    if not segments:
        return None
    _ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "final_video.mp4")
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    clips = []
    try:
        for video_file, start_ts in segments:
            video = VideoFileClip(video_file)
            # Fix: Subtract small buffer (0.1s) to prevent MoviePy "0 bytes read" warning at EOF
            safe_duration = max(0, video.duration - 0.1)
            end_ts = min(start_ts + segment_length_sec, safe_duration)
            if start_ts >= end_ts:
                start_ts = max(0, end_ts - segment_length_sec)
            
            # --- VidRush Polish ---
            sub = video.subclip(start_ts, end_ts)
            
            if mute_original:
                sub = sub.without_audio()
            
            if crop_headers:
                w, h = sub.size
                sub = sub.crop(y1=h*0.1, y2=h*0.9).resize((w, h))
                
            if fade_duration > 0:
                sub = sub.crossfadein(fade_duration)
                
            clips.append(sub)

        # Padding negative fade_duration creates the overlap for crossfade
        padding = -fade_duration if fade_duration > 0 else 0
        final_video = concatenate_videoclips(clips, method="compose", padding=padding)
        total_duration = final_video.duration

        if audio_path and os.path.exists(audio_path):
            audio = AudioFileClip(audio_path)
            
            # Sync: If audio is longer than video, loop video to match audio duration
            if audio.duration > final_video.duration:
                final_video = final_video.fx(loop, duration=audio.duration)
            
            final_video = final_video.set_audio(audio)

        # Write without subtitles first
        # GPU Acceleration check
        codec = "h264_nvenc" if use_gpu else "libx264"
        preset = "fast" if use_gpu else "medium"
        
        temp_video = output_path
        final_video.write_videofile(
            temp_video,
            codec=codec,
            audio=True if audio_path else False,
            bitrate="8000k",
            preset=preset,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
            logger=None,
        )
        final_video.close()
        for c in clips:
            try:
                c.close()
            except Exception:
                pass

        # Burn subtitles if provided (SRT in output dir so path is simple)
        if subtitle_text and subtitle_text.strip():
            srt_path = os.path.join(OUTPUT_DIR, "temp_subs.srt")
            # Use Whisper for accurate timing if audio exists, otherwise fallback to script split
            if audio_path and os.path.exists(audio_path):
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(generate_whisper_srt(audio_path))
            else:
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(script_to_srt(subtitle_text, total_duration))

            out_with_subs = output_path.replace(".mp4", "_subtitled.mp4")
            try:
                # Use forward slashes for ffmpeg filter
                srt_ff = srt_path.replace("\\", "/")
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_video,
                    "-vf", f"subtitles='{srt_ff}':force_style='FontSize=24,PrimaryColour=&Hffffff&'",
                    "-c:a", "copy", out_with_subs
                ], check=True, capture_output=True, timeout=300)
                if os.path.exists(out_with_subs):
                    os.replace(out_with_subs, temp_video)
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                pass
            finally:
                try:
                    os.unlink(srt_path)
                except Exception:
                    pass
        return output_path
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass


def analyze_frames(video_path, prompt, progress_callback=None, model=None, preprocess=None, device=None):
    """CLIP analysis: find best-matching timestamp for prompt. Uses OpenCLIP (ViT-B-32). Samples 1 frame per second."""
    try:
        import cv2
        import torch
        import open_clip
        from PIL import Image
    except ImportError:
        return 0.0
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    try:
        if model is None or preprocess is None:
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion400m_e32", force_quick_gelu=True)
            model = model.to(device).eval()
            
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_tokens = tokenizer([prompt[:77]]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
    except Exception:
        return 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    best_score, best_ts, count = -1.0, 0.0, 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if progress_callback and count % 10 == 0:
                progress_callback(min(1.0, count / total_frames))

            if count % max(1, int(fps)) == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_input = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(img_input).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    score = (image_features @ text_features.T).item()
                if score > best_score:
                    best_score, best_ts = score, (count / fps)
            count += 1
    finally:
        cap.release()
    return best_ts


def check_runpod() -> tuple[bool, str]:
    """Return (success, message) for RunPod status (env only; no ping)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    import os
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID") or os.environ.get("RUNPOD_POD_ID", "tx1n5nqj29goku")
    if not endpoint or str(endpoint).strip().startswith("your_"):
        return False, "Not configured (set RUNPOD_POD_ID or ENDPOINT_ID in .env)"
    return True, "Configured"
