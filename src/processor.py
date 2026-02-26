import os
import numpy as np
# Patch for MoviePy 1.0.3 compatibility with newer Pillow versions
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# Updated imports for MoviePy 1.0.3 compatibility
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from moviepy.video.fx.loop import loop
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.audio.fx.volumex import volumex
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from moviepy.video.fx.resize import resize
from moviepy.video.fx.colorx import colorx
from moviepy.video.fx.lum_contrast import lum_contrast

import subprocess
import tempfile
import torch
from transformers import AutoProcessor, AutoModel

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "output")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model globally to keep it in the RTX 4090 VRAM
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/xclip-base-patch32"
try:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
except Exception as e:
    print(f"⚠️ Failed to load X-CLIP model: {e}")
    processor = None
    model = None

def calculate_xclip_score(video_path, visual_prompt):
    """
    Analyzes a video's content and returns a similarity score (0.0 to 1.0) 
    compared to the visual prompt.
    """
    if model is None or processor is None:
        return 0.0
        
    try:
        import cv2
        # 1. Extract 16 frames evenly across the video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 16: return 0.0
        
        indices = np.linspace(0, total_frames - 1, 16).astype(int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # 2. Process video and text for X-CLIP
        inputs = processor(
            text=[visual_prompt],
            videos=list(frames),
            return_tensors="pt",
            padding=True
        ).to(device)

        # 3. Calculate Similarity Score
        with torch.no_grad():
            outputs = model(**inputs)
            # Normalize and get cosine similarity
            logits_per_video = outputs.logits_per_video  
            score = torch.sigmoid(logits_per_video).item()
            
        return score

    except Exception as e:
        print(f"⚠️ X-CLIP Error on {video_path}: {e}")
        return 0.0


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
    fp16 = False if model.device.type == "cpu" else True
    result = model.transcribe(audio_path, fp16=fp16)
    srt_lines = []
    for i, seg in enumerate(result['segments']):
        start = _sec_to_srt(seg['start'])
        end = _sec_to_srt(seg['end'])
        text = seg['text'].strip()
        srt_lines.append(f"{i + 1}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)


def apply_ai_transition(clip, transition_type, duration=0.5):
    """Routes the clip through the specific mathematical transform for the transition."""
    if transition_type == "zoom_in":
        # Simulate a push-in camera motion (scales from 1.0 to 1.1x over the clip length)
        return clip.fx(resize, lambda t: 1.0 + 0.1 * (t / clip.duration))
    
    elif transition_type == "dip_to_black":
        # Cinematic fade from black
        return clip.fx(fadein, duration)
        
    elif transition_type == "flash":
        # Simulates a film flash/light leak by maxing brightness for the first 0.2 seconds
        # We apply a colorx multiplier that drops off quickly
        def flash_effect(get_frame, t):
            frame = get_frame(t)
            if t < 0.2:
                img = frame.astype(np.float64)
                factor = min(2.0, 1.0 + (0.2 - t) * 5) # Boost brightness
                return np.clip(img * factor, 0, 255).astype(np.uint8)
            return frame
        return clip.fl(flash_effect)
        
    elif transition_type == "blur":
        # Fast crossfade handles blur simulation smoothly without massive GPU cost
        return clip.crossfadein(duration)
        
    # Default: Standard smooth crossfade
    return clip.crossfadein(duration)

def apply_ken_burns(image_path, duration=10, output_size=(1920, 1080)):
    """Creates a slow cinematic zoom from a static image."""
    clip = ImageClip(image_path).set_duration(duration)
    
    # Simple zoom-in effect (100% to 120% size over the duration)
    clip = clip.resize(lambda t: 1 + 0.02 * t) 
    clip = clip.set_position('center').resize(height=output_size[1])
    
    return clip

def apply_historical_look(clip):
    """Ages modern footage to match 1920s style."""
    # Add warmth/sepia and boost contrast
    return clip.fx(colorx, 1.2).fx(lum_contrast, lum=0, contrast=0.3)

def _has_nvenc():
    """Check if the installed FFmpeg supports h264_nvenc."""
    try:
        # We use -v quiet to avoid spam, but -encoders prints to stdout
        cmd = ["ffmpeg", "-encoders"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return "h264_nvenc" in res.stdout
    except Exception:
        return False

def trim_final_video(segments, audio_path=None, output_path=None, segment_length_sec=15, subtitle_text=None, 
                     use_gpu=False, crop_headers=True, mute_original=True, fade_duration=0.5, save_srt_path=None, 
                     sync_to_audio=False, bg_music_path=None, bg_music_volume=0.1):
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
    srt_path = save_srt_path if save_srt_path else os.path.join(OUTPUT_DIR, "temp_subs.srt")
    srt_generated = False
    seen_files = set()

    try:
        # --- AUTO-TRIM LOGIC (Sync to Narration) ---
        if sync_to_audio and audio_path and os.path.exists(audio_path):
            import whisper
            # Load model (cached)
            model = whisper.load_model("base")
            fp16 = False if model.device.type == "cpu" else True
            result = model.transcribe(audio_path, fp16=fp16)
            
            # Generate SRT immediately from this transcription to save time
            srt_lines = []
            for i, seg in enumerate(result['segments']):
                start_srt = _sec_to_srt(seg['start'])
                end_srt = _sec_to_srt(seg['end'])
                text = seg['text'].strip()
                srt_lines.append(f"{i + 1}\n{start_srt} --> {end_srt}\n{text}\n")
            
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines))
            srt_generated = True

            # Create clips matching sentence durations
            for i, seg in enumerate(result['segments']):
                sentence_duration = seg['end'] - seg['start']
                # Add fade buffer so crossfade doesn't eat the word
                clip_duration = sentence_duration + fade_duration
                
                vid_idx = i % len(segments)
                segment_data = segments[vid_idx]
                video_file = segment_data['path']
                transition_type = segment_data.get('transition', 'fade')
                source_type = segment_data.get('source', 'youtube')
                
                # Handle image source for Ken Burns effect
                if source_type == 'unsplash_photo':
                    sub = apply_ken_burns(video_file, duration=clip_duration)
                    sub = apply_ai_transition(sub, transition_type, fade_duration)
                    clips.append(sub)
                    continue

                # Standard video processing
                start_ts = segment_data['start']
                video = VideoFileClip(video_file)
                
                # Source-specific safety buffers
                if source_type == "youtube":
                    start_ts = max(start_ts, 15.0) # Skip intro
                    if start_ts + clip_duration > video.duration - 20.0: # Avoid outro
                        start_ts = max(15.0, video.duration - 20.0 - clip_duration)
                else: # pexels, pixabay
                    if start_ts + clip_duration > video.duration - 2.0: # Safety buffer
                        start_ts = max(0, video.duration - 2.0 - clip_duration)
                
                sub = video.subclip(start_ts, start_ts + clip_duration)

                # Apply Historical Look to modern sources
                if source_type in ["pexels", "pixabay", "coverr"]:
                    sub = apply_historical_look(sub)
                
                if mute_original: sub = sub.without_audio()
                if crop_headers:
                    w, h = sub.size
                    if source_type == "youtube":
                        sub = sub.crop(y1=h*0.1, y2=h*0.85) # Aggressive crop for YouTube
                    sub = sub.resize(height=1080)
                    if sub.w > 1920: sub = sub.crop(x_center=sub.w/2, width=1920)
                    else: sub = sub.resize(width=1920)
                
                sub = apply_ai_transition(sub, transition_type, fade_duration)
                clips.append(sub)

        # --- STANDARD LOGIC (Fixed Segment Length) ---
        else:
            for segment_data in segments:
                video_file = segment_data['path']
                start_ts = segment_data['start']
                transition_type = segment_data.get('transition', 'fade')
                source_type = segment_data.get('source', 'youtube')

                if video_file in seen_files: continue
                seen_files.add(video_file)

                # Handle image source for Ken Burns effect
                if source_type == 'unsplash_photo':
                    sub = apply_ken_burns(video_file, duration=segment_length_sec)
                    sub = apply_ai_transition(sub, transition_type, fade_duration)
                    clips.append(sub)
                    continue

                video = VideoFileClip(video_file)
                end_ts = min(start_ts + segment_length_sec, video.duration)
                if start_ts >= end_ts:
                    start_ts = max(0, end_ts - segment_length_sec)
                
                sub = video.subclip(start_ts, end_ts)

                # Apply Historical Look to modern sources
                if source_type in ["pexels", "pixabay", "coverr"]:
                    sub = apply_historical_look(sub)

                if mute_original: sub = sub.without_audio()
                if crop_headers:
                    w, h = sub.size
                    if source_type == "youtube":
                        sub = sub.crop(y1=h*0.1, y2=h*0.85) # Aggressive crop for YouTube

                    sub = sub.resize(height=1080)
                    if sub.w > 1920: sub = sub.crop(x_center=sub.w/2, width=1920)
                    else: sub = sub.resize(width=1920)
                
                sub = apply_ai_transition(sub, transition_type, fade_duration)
                clips.append(sub)

        # Padding negative fade_duration creates the overlap for crossfade
        padding = -fade_duration if fade_duration > 0 else 0
        final_video = concatenate_videoclips(clips, method="compose", padding=padding)
        total_duration = final_video.duration

        # --- AUDIO MIXING (Voiceover + Background Music) ---
        audio_tracks = []

        # 1. Voiceover (Primary)
        if audio_path and os.path.exists(audio_path):
            voiceover = AudioFileClip(audio_path)
            audio_tracks.append(voiceover)
            
            # Sync: If audio is longer than video, loop video to match audio duration
            if voiceover.duration > final_video.duration:
                final_video = final_video.fx(loop, duration=voiceover.duration)

        # 2. Background Music (Ducked)
        if bg_music_path and os.path.exists(bg_music_path):
            bgm = AudioFileClip(bg_music_path)
            bgm = bgm.fx(audio_loop, duration=final_video.duration) # Loop to fill video
            bgm = bgm.fx(volumex, bg_music_volume) # Ducking: Lower volume (default 0.1)
            audio_tracks.append(bgm)

        if audio_tracks:
            final_audio = CompositeAudioClip(audio_tracks)
            final_video = final_video.set_audio(final_audio)

        # Write without subtitles first
        # GPU Acceleration check
        codec = "libx264"
        preset = "medium"
        
        if use_gpu and _has_nvenc():
            codec = "h264_nvenc"
            preset = "fast"
        
        temp_video = output_path
        final_video.write_videofile(
            temp_video,
            codec=codec,
            audio=bool(audio_tracks),
            bitrate="10000k",
            fps=24,
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
            # Only generate if we haven't already done so in the sync block
            if not srt_generated:
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
                # Escape colons for Windows paths (e.g. C:/ -> C\:/) inside ffmpeg filter string
                srt_ff = srt_ff.replace(":", "\\:")
                
                # Pro Viral Style: Smaller Font (18), Yellow Text, Outline=2, MarginV=20
                viral_style = "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=20,Bold=-1"
                
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_video,
                    "-vf", f"subtitles='{srt_ff}':force_style='{viral_style}'",
                    "-c:a", "copy", out_with_subs
                ], check=True, capture_output=True, timeout=300)
                if os.path.exists(out_with_subs):
                    os.replace(out_with_subs, temp_video)
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
                print(f"Subtitle burn-in failed: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"FFmpeg stderr: {e.stderr.decode()}")
            finally:
                if not save_srt_path:
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
        # VidGear optional import for multi-threaded decoding
        try:
            from vidgear.gears import CamGear
        except ImportError:
            CamGear = None
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
    
    # --- VidGear Integration ---
    stream = None
    cap = None
    
    if CamGear:
        try:
            # logging=False keeps it quiet. CamGear is multi-threaded.
            stream = CamGear(source=video_path, logging=False).start()
            fps = stream.framerate
            # Access internal stream for metadata
            total_frames = int(stream.stream.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        except Exception:
            stream = None

    if stream is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

    best_score, best_ts, count = -1.0, 0.0, 0
    try:
        while True:
            if stream:
                frame = stream.read()
                if frame is None: break
            else:
                ret, frame = cap.read()
                if not ret: break

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
        if stream:
            stream.stop()
        elif cap:
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
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID") or os.environ.get("RUNPOD_POD_ID", "mw31zouly6drzw")
    if not endpoint or str(endpoint).strip().startswith("your_"):
        return False, "Not configured (set RUNPOD_POD_ID or ENDPOINT_ID in .env)"
    return True, "Configured"
