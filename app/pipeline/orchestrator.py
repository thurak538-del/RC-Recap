import os
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from app.pipeline.downloader import VideoDownloader
from app.pipeline.audio_extractor import AudioExtractor
from app.pipeline.gemini_rewriter import GeminiRewriter
from app.pipeline.local_transcriber import LocalWhisperTranscriber
from app.pipeline.local_translator import LocalNLLBTranslator
from app.pipeline.tts_engine import TTSEngine
from app.pipeline.audio_mixer import AudioMixer
from app.pipeline.vocal_separator import VocalSeparator
from app.pipeline.subtitle_burner import SubtitleBurner
from app.pipeline.gemini_blur_detector import GeminiSubtitleBandDetector, padded_band
from app.pipeline.burmese_text import normalize_burmese_digits, prepare_burmese_tts_text

# Exact Burmese stage labels
STAGES = [
    "ဗီဒီယို ဒေါင်းလုဒ်လုပ်နေပါတယ်...",
    "အသံဖိုင် ထုတ်ယူနေပါတယ်...",
    "အသံကို စာသားအဖြစ် ပြောင်းနေပါတယ်...",
    "စာသားကို ဘာသာပြန် / ပြန်လည်ရေးသားနေပါတယ်...",
    "အသံဖိုင် ဖန်တီးနေပါတယ်...",
    "ဗီဒီယိုနဲ့ အသံ ပေါင်းနေပါတယ်...",
    "စာတန်းထိုးနေပါတယ်...",
    "ပြီးပါပြီ ✓"
]


class PipelineOrchestrator:
    def __init__(
        self,
        job_id: str,
        job_dir: Path,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.job_id = job_id
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.event_callback = event_callback
        self.status = "idle"
        self.current_stage = ""
        self.artifacts: Dict[str, str] = {}

    def _notify(self, stage: str, stage_index: int, message: str, progress: float, data: Optional[Dict[str, Any]] = None):
        self.current_stage = stage
        payload = {
            "job_id": self.job_id,
            "stage": stage,
            "stage_index": stage_index,
            "total_stages": len(STAGES),
            "message": message,
            "progress": round(progress, 1),
            "timestamp": time.time(),
            "data": data or {}
        }
        if self.event_callback:
            self.event_callback(payload)

    def _get_media_duration(self, file_path: Path) -> float:
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(res.stdout.strip() or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _get_transcript_duration(transcription: Dict[str, Any], fallback: float = 0.0) -> float:
        """Use the end of the last spoken segment, not silent video tail, for expansion."""
        segments = transcription.get("segments") or []
        ends = []
        for segment in segments:
            try:
                end = float(segment.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end > 0:
                ends.append(end)
        if ends:
            return max(ends)
        try:
            reported = float(transcription.get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            reported = 0.0
        return reported if reported > 0 else float(fallback or 0.0)

    @staticmethod
    def _attach_original_timing(processed: list, original: list) -> list:
        """DUB mode: every processed (translated) segment must carry the ORIGINAL
        video start/end so the voice can be placed exactly when the person speaks."""
        by_id = {}
        for src in original or []:
            if src.get("id") is not None:
                by_id[src["id"]] = src
        out, missing = [], 0
        for i, seg in enumerate(processed or []):
            seg = dict(seg)
            if seg.get("start") is None or seg.get("end") is None:
                src = by_id.get(seg.get("id")) if seg.get("id") is not None else None
                if src is None and len(processed) == len(original or []):
                    src = original[i]
                if src is not None:
                    seg["start"] = src.get("start")
                    seg["end"] = src.get("end")
                else:
                    missing += 1
            out.append(seg)
        if missing:
            print(f"[DUB SYNC] WARNING: {missing} segment(s) have no original timing; they will follow the previous sentence.", flush=True)
        return out

    def run(
        self,
        video_url: Optional[str],
        uploaded_video_path: Optional[Path],
        groq_api_key: str,
        gemini_api_key: str,
        ai_mode: str = "cloud",
        voice_engine: str = "edge_tts",
        edge_tts_voice: str = "my-MM-ThihaNeural",
        voxcpm_voice_path: Optional[str] = None,
        voxcpm_ref_text: Optional[str] = None,
        voxcpm_device: Optional[str] = None,
        gemini_mode: str = "translate",
        target_language: str = "my",
        font_color: str = "#FFFFFF",
        font_size_px: int = 70,
        font_style: str = "Z10-Cartoon",
        pos_x_pct: float = 50.0,
        pos_y_pct: float = 82.0,
        subtitle_animation: str = "fade",
        enable_subtitles: bool = True,
        auto_blur_subtitles: bool = False,
        auto_blur_padding_pct: float = 1.5,
        output_resolution: str = "1080p",
        preserve_original_background: bool = False,
        processing_mode: str = "recap",
        enable_4k_filter: bool = False,
        mirror_mode_7s: bool = False,
    ) -> Dict[str, Any]:
        try:
            self.status = "running"

            # ----------------------------------------------------
            # STAGE 1: ဗီဒီယို ဒေါင်းလုဒ်လုပ်နေပါတယ်...
            # ----------------------------------------------------
            stage_1 = STAGES[0]
            self._notify(stage_1, 1, "ဒေါင်းလုဒ် စတင်နေပါပြီ...", 0.0)
            if uploaded_video_path and Path(uploaded_video_path).exists():
                video_file = self.job_dir / f"downloaded_video{Path(uploaded_video_path).suffix}"
                shutil.copy2(str(uploaded_video_path), str(video_file))
                self._notify(stage_1, 1, "တင်ထားသော ဗီဒီယိုဖိုင်ကို အသုံးပြုနေပါသည်..."
