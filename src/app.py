import streamlit as st
import os
from scout import get_youtube_links, download_video, check_youtube
from processor import analyze_frames, trim_final_video, check_runpod
from prompt_tuner import tune_script_for_search, check_openai, generate_narration_script
from voiceover import generate_voiceover_openai
from voice_engine import generate_voiceover as generate_voiceover_elevenlabs
from script_engine import expand_script
from cloud_worker import run_task_on_cloud, ping_runpod, terminate_runpod, cleanup_cloud_files, render_video_on_cloud

st.set_page_config(page_title="VidRush - Silent Prototype", layout="wide")

st.title("🎬 videoaut: A video Pro Engine")
st.caption("Script → Edit → Voiceover & subtitles (OpenAI TTS). Or use **Generate Full Documentary** for scene-by-scene CLIP matching.")
st.info("💡 **Sidebar (←)** → Target duration, segment length (10/15/30/60 sec). **Generate Full Documentary** = GPT-4o scenes → 1 video per scene → CLIP → high-quality stitch (FFmpeg required).")

# --- API status ---
with st.expander("🔌 API & service status (OpenAI / YouTube)", expanded=True):
    o_ok, o_msg = check_openai()
    st.write(f"**OpenAI:** {'✅ ' if o_ok else '❌ '} {o_msg}")
    y_ok, y_msg = check_youtube()
    st.write(f"**YouTube (yt-dlp):** {'✅ ' if y_ok else '❌ '} {y_msg}")
    if not os.environ.get("YOUTUBE_API_KEY"):
        st.warning("⚠️ No YOUTUBE_API_KEY found. Search will be slow (scraping mode).")
    try:
        import vidgear
        st.write(f"**VidGear:** ✅ Installed (Accelerated Processing)")
    except ImportError:
        st.write("**VidGear:** ❌ Not installed (Using standard OpenCV)")

# --- Settings ---
with st.sidebar:
    st.subheader("Video length")
    target_mins = st.selectbox("Target duration", [1, 2, 5, 10], index=0, format_func=lambda x: f"{x} min")
    segment_sec = st.selectbox("Segment length (per clip)", [10, 15, 30, 60], index=0, format_func=lambda x: f"{x} sec")
    use_openai_tune = st.checkbox("Use OpenAI to tune script → better search", value=True)

    # RunPod Toggle
    has_rp_keys = os.environ.get("RUNPOD_API_KEY") and (os.environ.get("RUNPOD_ENDPOINT_ID") or os.environ.get("RUNPOD_POD_ID"))
    enable_runpod = st.checkbox("Use RunPod Cloud GPU", value=False, disabled=not has_rp_keys, help="Offloads AI processing to RunPod if keys are set in .env")

    if enable_runpod:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📡 Ping"):
                ok, msg = ping_runpod()
                if ok: st.success("OK")
                else: st.error(f"Fail: {msg}")
        with c2:
            if st.button("🧹 Clean"):
                ok, msg = cleanup_cloud_files()
                if ok: st.success(f"Del: {msg.get('files_removed', 0)}")
        
        if st.button("🛑 Terminate Pod (Stop Billing)", type="primary", help="Permanently deletes the Pod to stop all costs."):
            ok, msg = terminate_runpod()
            if ok: st.success("Pod Terminated! Billing stopped.")
            else: st.error(f"Termination Failed: {msg}")

    add_voiceover = st.checkbox("Add voiceover from script", value=True)
    tts_provider = st.selectbox("Voice Provider", ["OpenAI", "ElevenLabs"], index=0) if add_voiceover else "OpenAI"
    
    eleven_voice = "Adam"
    if add_voiceover and tts_provider == "ElevenLabs":
        eleven_voice = st.selectbox("ElevenLabs Voice", ["Adam", "Antoni", "Bella", "Josh", "Rachel"], index=0)
        
    add_subtitles = st.checkbox("Add subtitles from script", value=True)
    auto_trim = st.checkbox("Auto-Trim (Sync to Narration)", value=False, help="Automatically cuts video clips to match the length of each sentence in the voiceover.")

    st.subheader("Audio Settings")
    bg_music_file = st.file_uploader("Background Music (Optional)", type=["mp3", "wav"])
    bg_music_vol = st.slider("Music Volume (Ducking)", 0.0, 0.5, 0.1, 0.01, help="Volume of background music relative to voiceover (0.1 = 10%).")

    st.subheader("Video Style")
    transition_mode = st.radio("Transition Style", ["Smooth Crossfade (0.5s)", "Hard Cut (Fast)"], index=1, help="Hard Cut is faster to render and preferred for fast-paced videos.")
    fade_dur = 0.5 if "Smooth" in transition_mode else 0.0

    st.subheader("AI Models")
    script_model = st.selectbox(
        "Script/Search Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Model for generating narration and tuning search queries. `gpt-4o-mini` is fast and cheap."
    )
    doc_plan_model = st.selectbox(
        "Documentary Plan Model",
        ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Model for expanding a topic into visual scenes. `gpt-4o` is recommended for this."
    )

    num_clips = max(1, (target_mins * 60) // segment_sec)
    st.caption(f"Will fetch up to **{num_clips}** clips (~{target_mins} min total).")

# Save uploaded background music
bg_music_path = None
if bg_music_file:
    os.makedirs(os.path.join("data", "audio"), exist_ok=True)
    bg_music_path = os.path.abspath(os.path.join("data", "audio", "bg_music_temp.mp3"))
    with open(bg_music_path, "wb") as f:
        f.write(bg_music_file.getbuffer())

with st.expander("🌿 Need inspiration? View Natural Prompts"):
    st.markdown("""
    - **Mountain:** Cinematic drone footage of the Swiss Alps, snow-capped peaks, green valleys.
    - **Forest:** Sunlight filtering through a dense green forest (god rays), mossy trees, ferns.
    - **Beach:** Vibrant sunset over a tropical beach, waves crashing on the sand, palm trees.
    - **Desert:** Wide angle shot of red sand dunes in the desert, wind blowing sand.
    - **Underwater:** Coral reef teeming with colorful fish, clear blue water, sunlight shafts.
    """)

prompt = st.text_input("Describe the scenes you want (or paste your idea):", "Cinematic views of ancient Rome")

# --- Step 1: Generate script with OpenAI so user can see and edit ---
st.subheader("📝 Script (for voiceover & subtitles)")
if "script_text" not in st.session_state:
    st.session_state.script_text = ""

if st.button("✨ Generate script with OpenAI"):
    with st.spinner(f"OpenAI ({script_model}) is writing the narration script..."):
        st.session_state.script_text = generate_narration_script(prompt, target_duration_min=target_mins, model=script_model)
    st.success("Edit the script below if needed, then click **Generate Video**.")

script_for_video = st.text_area(
    "Edit the script below. This exact text is used for voiceover and subtitles.",
    value=st.session_state.script_text,
    height=180,
    placeholder="Generate script with the button above, or type your own. Voiceover and subtitles are read from here.",
)
st.session_state.script_text = script_for_video
has_script = bool(script_for_video.strip())
if has_script:
    st.caption("✅ Script ready → voiceover and subtitles will use this text when you click **Generate Video**.")
else:
    st.caption("⚠️ No script yet. Generate one above or type your own; otherwise video will be silent with no subtitles.")

# Search query: use script summary or prompt
search_input = (script_for_video.split(".")[0][:200] if script_for_video.strip() else prompt) or prompt

# --- Documentary Pro: scene-by-scene ---
st.subheader(f"🎞️ Generate Full Documentary ({num_clips} scenes)")
st.caption(f"GPT-4o expands your topic into {num_clips} visual scenes → 1 YouTube clip per scene → CLIP picks best moment → high-quality stitch. Requires FFmpeg.")

if "doc_state" not in st.session_state:
    st.session_state.doc_state = "idle"
if "doc_data" not in st.session_state:
    st.session_state.doc_data = {}

if st.button("1. Plan & Fetch Scenes"):
    with st.status(f"Step 1: Planning & Fetching ({target_mins} min)...", expanded=True) as status:
        # Use the generated script if available, otherwise the prompt
        source_text = script_for_video if script_for_video.strip() else prompt
        st.write(f"🧠 **{doc_plan_model}: expanding script into {num_clips} visual scenes**")
        visual_plan = expand_script(source_text, num_scenes=num_clips, model=doc_plan_model)
        if not visual_plan:
            st.error("Could not generate scenes. Check OPENAI_API_KEY and try again.")
            st.stop()
        
        clips_data = []
        for i, scene_data in enumerate(visual_plan):
            # Handle both old string format (if fallback occurs) and new dict format
            if isinstance(scene_data, dict):
                scene_desc = scene_data.get("desc", "Cinematic stock footage")
                transition = scene_data.get("transition", "fade")
                emotion = scene_data.get("emotion", "neutral")
            else:
                scene_desc = scene_data
                transition = "fade"
                emotion = "neutral"

            st.write(f"🎞️ **Scene {i+1}/{num_clips}:** '{scene_desc[:40]}...' | 🧠 Mood: {emotion.upper()} | 🎬 FX: {transition}")
            links = get_youtube_links(scene_desc, limit=1)
            if not links:
                st.warning(f"No result for scene {i+1}. (Check internet or set YOUTUBE_API_KEY)")
                continue
            
            prog_bar = st.progress(0, text="Processing...")
            try:
                # Download
                path = download_video(links[0], progress_callback=lambda p: prog_bar.progress(p, text=f"Downloading: {int(p*100)}%"))
                
                if path and os.path.exists(path):
                    # AI Analysis (Cloud or Local)
                    ts = 0.0
                    use_runpod = enable_runpod and has_rp_keys
                    
                    if use_runpod:
                        # Try Cloud First
                        res = run_task_on_cloud(links[0], scene_desc)
                        if res and "timestamp" in res:
                            ts = res["timestamp"]
                            st.write(f"☁️ Cloud matched at {ts:.1f}s")
                    
                    if ts == 0.0:
                        # Fallback to Local if Cloud failed or disabled
                        ts = analyze_frames(path, scene_desc, progress_callback=lambda p: prog_bar.progress(p, text=f"💻 Local Analysis: {int(p*100)}%"))
                    
                    clips_data.append({
                        "path": path, 
                        "start": ts, 
                        "name": os.path.basename(path), 
                        "desc": scene_desc, 
                        "url": links[0],
                        "transition": transition
                    })
                    st.write(f"✔️ Matched at {ts:.1f}s")
                prog_bar.empty()
            except Exception as e:
                st.warning(f"Scene {i+1} failed: {e}")
                prog_bar.empty()
        
        if not clips_data:
            st.error("No video segments were downloaded or analyzed.")
            st.stop()
            
        # Generate Voiceover
        doc_audio = None
        if add_voiceover and script_for_video.strip():
            st.write(f"🔊 **Generating Voiceover ({tts_provider})...**")
            try:
                if tts_provider == "ElevenLabs":
                    doc_audio = generate_voiceover_elevenlabs(script_for_video, voice_name=eleven_voice)
                else:
                    doc_audio = generate_voiceover_openai(script_for_video)
            except Exception as e:
                st.warning(f"Voiceover failed: {e}")

        st.session_state.doc_data = {"clips": clips_data, "audio": doc_audio, "source_text": source_text}
        st.session_state.doc_state = "review"
        status.update(label="✅ Scenes ready! Scroll down to review.", state="complete")

if st.session_state.doc_state == "review":
    st.divider()
    st.subheader("✂️ Step 2: Review & Edit Documentary Scenes")
    doc_source = st.session_state.doc_data.get("source_text", "")
    if doc_source:
        st.caption(f"**Based on:** {doc_source[:100]}..." if len(doc_source) > 100 else f"**Based on:** {doc_source}")

    if st.session_state.doc_data.get("audio"):
        st.write("🔊 **Preview Voiceover:**")
        st.audio(st.session_state.doc_data["audio"])

    with st.form("doc_review_form"):
        edited_clips = []
        for i, clip in enumerate(st.session_state.doc_data["clips"]):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**Scene {i+1}:** {clip['desc']}")
                st.caption(f"File: `{clip['name']}`")
                with st.expander("Preview Clip"):
                    st.video(clip["path"])
                    st.info(f"📝 **Scene Prompt:** {clip['desc']}")
            with c2:
                new_start = st.number_input(f"Start (s)", min_value=0.0, value=float(clip["start"]), step=1.0, key=f"doc_start_{i}")
                clip["start"] = new_start
            edited_clips.append(clip)
            st.divider()
        
        c_ren1, c_ren2 = st.columns(2)
        with c_ren1:
            render_local = st.form_submit_button("💻 Render Locally (CPU)")
        with c_ren2:
            render_cloud = st.form_submit_button("☁️ Render on Cloud (GPU)")

    if render_local or render_cloud:
        success = False
        try:
            doc_audio = st.session_state.doc_data.get("audio")
            subs = script_for_video.strip() if add_subtitles and script_for_video.strip() else None
            # Pass the 3-part tuple: path, start_time, transition
            winning_segments = [(c["path"], c["start"], c.get("transition", "fade")) for c in edited_clips]
            srt_path = os.path.abspath(os.path.join("data", "output", "documentary_subtitles.srt"))
            
            if render_cloud and enable_runpod:
                st.info("🚀 Sending render job to RunPod RTX 4090...")
                # Prepare segments for cloud (needs URL)
                cloud_segments = [{"url": c.get("url"), "start": c["start"]} for c in edited_clips if c.get("url")]
                
                vid_bytes, err = render_video_on_cloud(cloud_segments, doc_audio, subs)
                if vid_bytes:
                    final_path = os.path.join("data", "output", "cloud_render.mp4")
                    with open(final_path, "wb") as f: f.write(vid_bytes)
                    st.success("✅ Cloud Render Complete!")
                else:
                    st.error(f"Cloud Render Failed: {err}")
                    st.stop()
            else:
                st.write("🛠️ **Rendering locally...**")
                final_path = trim_final_video(winning_segments, audio_path=doc_audio, segment_length_sec=segment_sec, subtitle_text=subs,
                                              crop_headers=True, mute_original=True, fade_duration=fade_dur, save_srt_path=srt_path, 
                                              sync_to_audio=auto_trim, bg_music_path=bg_music_path, bg_music_volume=bg_music_vol)
                st.success("✅ Documentary complete!")
                
            if os.path.exists(final_path):
                # Cleanup raw files
                st.write("🧹 **Cleaning up raw files...**")
                for seg in winning_segments:
                    path = seg[0]
                    if os.path.exists(path):
                        try: os.remove(path)
                        except: pass
                
                st.session_state.doc_final_video = final_path
                st.session_state.doc_state = "complete"
                success = True
        except Exception as e:
            st.error(f"Stitching failed: {e}")
        
        if success:
            st.rerun()

if st.session_state.doc_state == "complete":
    st.divider()
    st.subheader("🍿 Final Documentary Ready")
    final_path = st.session_state.doc_final_video
    doc_audio = st.session_state.doc_data.get("audio")
    srt_path = os.path.abspath(os.path.join("data", "output", "documentary_subtitles.srt"))
    
    st.write(f"**Location:** `{final_path}`")
    st.video(final_path)
    
    safe_name = "".join(c for c in st.session_state.doc_data.get("source_text", "doc") if c.isalnum() or c in " -_").strip()[:30]
    
    with open(final_path, "rb") as f:
        st.download_button("💾 Download Video (MP4)", f, file_name=f"vidrush_{safe_name}.mp4", key="dl_doc_final", mime="video/mp4")
    
    if st.button("🔄 Start New Documentary"):
        st.session_state.doc_state = "idle"
        st.rerun()


st.divider()
st.subheader(" Generate Video (multi-clip from search)")

if "gen_vid_state" not in st.session_state:
    st.session_state.gen_vid_state = "idle" # idle, review
if "gen_vid_data" not in st.session_state:
    st.session_state.gen_vid_data = {}

if st.button("1. Fetch & Analyze Videos"):
    with st.status("Step 1: Fetching & Analyzing...", expanded=True) as status:
        # Step 0: Search queries (from tuned script or prompt)
        search_queries = [search_input]
        if use_openai_tune:
            st.write(f"📝 **{script_model}: tuning for search**")
            search_queries = tune_script_for_search(search_input, num_queries=5, model=script_model)
            search_queries = search_queries or [search_input]

        # Step 1: YouTube Search
        st.write("🔍 **Location: YouTube Cloud**")
        links = []
        per_query = max(1, (num_clips + len(search_queries) - 1) // len(search_queries))
        for q in search_queries:
            if len(links) >= num_clips:
                break
            part = get_youtube_links(q, limit=per_query)
            for u in part:
                if u not in links:
                    links.append(u)
                if len(links) >= num_clips:
                    break
        links = links[:num_clips]
        if not links:
            status.update(label="❌ No search results", state="error")
            st.error("No videos found. Try a different prompt or check your internet.")
            st.stop()

        # Step 2: Download
        st.write("📥 **Location: /data/raw_videos**")
        downloaded_items = [] # List of (url, file_path)
        dl_bar = st.progress(0, text="Starting downloads...")
        for url in links:
            try:
                dl_bar.progress(0, text=f"Downloading {url}...")
                file_path = download_video(url, progress_callback=lambda p: dl_bar.progress(p, text=f"Downloading {url}: {int(p*100)}%"))
                if file_path and os.path.exists(file_path):
                    downloaded_items.append((url, file_path))
                    st.write(f"Downloaded: {os.path.basename(file_path)}")
                else:
                    st.warning(f"Download failed or missing file for {url}")
            except Exception as e:
                st.warning(f"Download failed for {url}: {e}")
        dl_bar.empty()
        if not downloaded_items:
            status.update(label="❌ No videos downloaded", state="error")
            st.error("No videos were downloaded. Check your internet or try different links.")
            st.stop()

        # Step 3: AI Analysis
        # Check for RunPod API Key to decide execution mode
        use_runpod = enable_runpod and has_rp_keys
        
        location_label = "RunPod Cloud GPU ☁️" if use_runpod else "Local GPU/CPU 💻"
        st.write(f"🧠 **Location: {location_label}**")
        
        clips_data = []
        
        for url, file_path in downloaded_items:
            if use_runpod:
                # Offload to cloud worker
                st.write(f"🚀 Sending job to RunPod: {os.path.basename(file_path)}...")
                with st.spinner("☁️ RunPod GPU is analyzing video..."):
                    result = run_task_on_cloud(url, prompt)
                
                if result and "timestamp" in result:
                    timestamp = result["timestamp"]
                    st.success(f"✅ RunPod finished! Timestamp: {timestamp:.2f}s")
                else:
                    err_msg = result.get("error") if isinstance(result, dict) else str(result)
                    st.warning(f"⚠️ RunPod failed: {err_msg}. Falling back to local CPU...")
                    # Local fallback
                    ana_bar = st.progress(0, text=f"Analyzing {os.path.basename(file_path)} locally...")
                    timestamp = analyze_frames(file_path, prompt, progress_callback=lambda p: ana_bar.progress(p, text=f"AI Analysis: {int(p*100)}%"))
                    ana_bar.empty()
            else:
                # Local fallback
                ana_bar = st.progress(0, text=f"Analyzing {os.path.basename(file_path)}...")
                timestamp = analyze_frames(file_path, prompt, progress_callback=lambda p: ana_bar.progress(p, text=f"AI Analysis: {int(p*100)}%"))
                ana_bar.empty()
            
            clips_data.append({"path": file_path, "start": timestamp, "name": os.path.basename(file_path), "desc": prompt, "url": url})
            st.write(f"✔️ Matched segment at {timestamp}s in {os.path.basename(file_path)}")

        # Step 4: Voiceover from script (OpenAI TTS)
        audio_path = None
        if add_voiceover and script_for_video.strip():
            st.write(f"🔊 **Voiceover from script ({tts_provider})**")
            try:
                if tts_provider == "ElevenLabs":
                    audio_path = generate_voiceover_elevenlabs(script_for_video, voice_name=eleven_voice)
                else:
                    audio_path = generate_voiceover_openai(script_for_video)
                st.write(f"Voiceover saved: {os.path.basename(audio_path)}")
            except Exception as e:
                st.warning(f"Voiceover failed: {e}. Continuing without audio.")
        elif add_voiceover and not script_for_video.strip():
            st.write("⏭️ Skipping voiceover (no script in the box above).")

        # Save state for review
        st.session_state.gen_vid_data = {
            "clips": clips_data,
            "audio": audio_path,
            "downloaded": downloaded_items,
            "prompt": prompt
        }
        st.session_state.gen_vid_state = "review"
        status.update(label="✅ Ready for review! Scroll down to edit.", state="complete")

# --- Review & Render Section ---
if st.session_state.gen_vid_state == "review":
    st.divider()
    st.subheader("✂️ Step 2: Review & Edit Clips")
    review_prompt = st.session_state.gen_vid_data.get("prompt", "")
    if review_prompt:
        st.caption(f"**Prompt:** {review_prompt}")

    st.info("Adjust the start times below if needed, then click Render.")
    
    if st.session_state.gen_vid_data.get("audio"):
        st.write("🔊 **Preview Voiceover:**")
        st.audio(st.session_state.gen_vid_data["audio"])

    with st.form("review_form"):
        edited_clips = []
        for i, clip in enumerate(st.session_state.gen_vid_data["clips"]):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**Clip {i+1}:** `{clip['name']}`")
                with st.expander("Preview Video"):
                    st.video(clip["path"])
                    if clip.get("desc"):
                        st.info(f"📝 **Prompt:** {clip['desc']}")
            with c2:
                new_start = st.number_input(
                    f"Start Time (sec)", 
                    min_value=0.0, 
                    value=float(clip["start"]), 
                    step=1.0, 
                    key=f"start_{i}"
                )
                clip["start"] = new_start
            edited_clips.append(clip)
            st.divider()
        
        render_clicked = st.form_submit_button("🎬 Render Final Video")

    if render_clicked:
        # Step 5: Stitching + subtitles from script
        winning_segments = [(c["path"], c["start"]) for c in edited_clips]
        audio_path = st.session_state.gen_vid_data["audio"]
        downloaded_items = st.session_state.gen_vid_data["downloaded"]

        if add_subtitles and script_for_video.strip():
            st.write("📜 **Subtitles from script** → burning into video.")
        elif add_subtitles and not script_for_video.strip():
            st.write("⏭️ Skipping subtitles (no script).")
        st.write("🎬 **Location: data/output**")
        if not winning_segments:
            st.error("No segments to stitch.")
            st.stop()
        success = False
        try:
            srt_path = os.path.abspath(os.path.join("data", "output", "video_subtitles.srt"))
            final_path = trim_final_video(
                winning_segments,
                audio_path=audio_path,
                segment_length_sec=segment_sec,
                subtitle_text=script_for_video.strip() if add_subtitles and script_for_video.strip() else None,
                save_srt_path=srt_path,
                fade_duration=fade_dur,
                sync_to_audio=auto_trim,
                bg_music_path=bg_music_path,
                bg_music_volume=bg_music_vol
            )
            if not os.path.exists(final_path):
                st.error(f"Video file was not created at: {final_path}")
            else:
                st.success("✅ Render complete!")
                
                # Cleanup raw files
                st.write("🧹 **Cleaning up raw files...**")
                for _, file in downloaded_items:
                    if os.path.exists(file):
                        try: os.remove(file)
                        except: pass
                
                st.session_state.gen_vid_final_video = final_path
                st.session_state.gen_vid_state = "complete"
                success = True

        except Exception as e:
            st.error(f"Could not create final video: {e}")
            st.stop()
        
        if success:
            st.rerun()

if st.session_state.gen_vid_state == "complete":
    st.divider()
    st.subheader("🍿 Final Video Ready")
    final_path = st.session_state.gen_vid_final_video
    st.write(f"**Location:** `{final_path}`")
    st.video(final_path)
    
    safe_name = "".join(c for c in st.session_state.gen_vid_data.get("prompt", "video") if c.isalnum() or c in " -_").strip()[:30]
    with open(final_path, "rb") as f:
        st.download_button("💾 Download MP4", f, file_name=f"vidrush_{safe_name}.mp4", key="dl_gen_final", mime="video/mp4")
        
    if st.button("🔄 Create Another Video"):
        st.session_state.gen_vid_state = "idle"
        st.rerun()

# --- NEW SECTION: Upload & Analyze ---
st.divider()
st.subheader("📂 Upload & Analyze Local Video")
st.caption("Upload a video from your computer to find a specific scene using AI.")

raw_dir = os.path.join("data", "raw_videos")
existing_files = [f for f in os.listdir(raw_dir) if f.endswith((".mp4", ".mov", ".avi"))] if os.path.exists(raw_dir) else []

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
with col2:
    selected_existing = st.selectbox("OR Select from data/raw_videos", ["(None)"] + existing_files)

local_path = None
if uploaded_file:
    # 1. Save the file locally
    upload_dir = os.path.join("data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    local_path = os.path.join(upload_dir, uploaded_file.name)
    with open(local_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
elif selected_existing != "(None)":
    local_path = os.path.join(raw_dir, selected_existing)

upload_prompt = st.text_input("Describe the moment you want to find:", placeholder="e.g. a person smiling")

if local_path:
    st.video(local_path)
    
    if upload_prompt and st.button("🔍 Analyze Video"):
        use_runpod = enable_runpod and has_rp_keys
        timestamp = 0.0

        if use_runpod:
            st.info("☁️ Uploading to RunPod GPU for analysis (this may take a moment)...")
            res = run_task_on_cloud(prompt=upload_prompt, file_path=local_path)
            if res and "timestamp" in res:
                timestamp = res["timestamp"]
            else:
                st.error(f"Cloud analysis failed: {res}")
        else:
            st.info("💻 Processing locally...")
            prog_bar = st.progress(0, text="Initializing AI...")
            timestamp = analyze_frames(local_path, upload_prompt, progress_callback=lambda p: prog_bar.progress(p, text=f"Scanning video: {int(p*100)}%"))
            prog_bar.empty()
        
        if timestamp > 0:
            st.success(f"✅ Found best match at **{timestamp:.2f}s**")
            st.video(local_path, start_time=int(timestamp))
