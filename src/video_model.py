import torch
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModel

class XClipHandler:
    def __init__(self, model_name="microsoft/xclip-base-patch32", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⏳ Loading X-CLIP ({model_name}) on {self.device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("✅ X-CLIP Ready.")
        except Exception as e:
            print(f"❌ Failed to load X-CLIP: {e}")
            self.model = None

    def _extract_frames(self, video_path, num_frames=8):
        """Extracts evenly spaced frames from a video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return None
        
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(frames) < num_frames:
            # Pad if video is too short
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
                
        return frames

    def encode_video(self, video_path):
        if self.model is None: return None
        frames = self._extract_frames(video_path)
        if not frames: return None
        
        try:
            inputs = self.processor(videos=list(frames), return_tensors="pt").to(self.device)
            with torch.no_grad():
                video_features = self.model.get_video_features(**inputs)
                video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            return video_features.cpu().numpy()
        except Exception as e:
            print(f"Error encoding video {video_path}: {e}")
            return None

    def encode_text(self, text):
        if self.model is None: return None
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None

    def compute_similarity(self, text_emb, video_emb):
        if text_emb is None or video_emb is None: return 0.0
        return (text_emb @ video_emb.T).item()