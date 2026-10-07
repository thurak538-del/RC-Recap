"""Offline transcription using faster-whisper with a pre-downloaded Kaggle model."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class LocalWhisperTranscriber:
    _models: Dict[str, Any] = {}

    def __init__(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        self.progress_callback = progress_callback

    def _load_model(self):
        from faster_whisper import WhisperModel

        model_path = os.getenv("RECAP_WHISPER_MODEL", "large-v3")
        device = os.getenv("RECAP_WHISPER_DEVICE", "cuda")
        device_index = int(os.getenv("RECAP_WHISPER_DEVICE_INDEX", "0"))
        cache_key = f"{model_path}|{device}|{device_index}|{os.getenv('RECAP_WHISPER_COMPUTE', 'float16')}"
        if cache_key not in self._models:
            compute_type = os.getenv("RECAP_WHISPER_COMPUTE", "float16" if device == "cuda" else "int8")
            self._models[cache_key] = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                device_index=device_index,
                download_root=os.getenv("HF_HOME", "/kaggle/working/huggingface"),
            )
        return self._models[cache_key]

    # ------------------------------------------------------------------
    # DUB mode helpers: tight, sentence-sized segments from word timestamps
    # ------------------------------------------------------------------
    SENTENCE_END = (".", "?", "!", "\u3002", "\uff01", "\uff1f", "\u2026")

    @classmethod
    def _split_words(cls, words, max_len: float, pause: float, min_len: float = 0.5, sentence_min: float = 1.5):
        """Group words into speech chunks: break on a pause, on a finished sentence,
        or when a chunk gets longer than max_len seconds. Tiny chunks are merged back."""
        chunks, cur = [], []

        def close():
            nonlocal cur
            if cur:
                chunks.append(cur)
                cur = []

        for w in words:
            if cur:
                gap = float(w.start) - float(cur[-1].end)
                if gap >= pause or (float(w.end) - float(cur[0].start)) > max_len:
                    close()
            cur.append(w)
            token = (w.word or "").strip()
            if token.endswith(cls.SENTENCE_END) and (float(cur[-1].end) - float(cur[0].start)) >= sentence_min:
                close()
        close()

        merged = []
        for ch in chunks:
            dur = float(ch[-1].end) - float(ch[0].start)
            if merged and dur < min_len and (float(ch[0].start) - float(merged[-1][-1].end)) < pause:
                merged[-1].extend(ch)
            else:
                merged.append(ch)
        return merged

    def transcribe(self, audio_path: Path, output_dir: Optional[Path] = None, precise_timing: bool = False) -> Dict[str, Any]:
        audio_path = Path(audio_path)
        output_dir = Path(output_dir or audio_path.parent)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if self.progress_callback:
            self.progress_callback("Local Whisper model ကို load လုပ်နေပါသည်...", 10.0)

        model = self._load_model()
        # faster-whisper's path decoder passes ``metadata_errors`` to PyAV.
        # Some Kaggle/PyAV combinations removed that keyword from av.open(),
        # so decode our already-normalized 16 kHz WAV with soundfile and pass
        # samples directly. This also avoids a second FFmpeg/PyAV decode path.
        try:
            import soundfile as sf
            import numpy as np

            audio_samples, _sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if getattr(audio_samples, "ndim", 1) > 1:
                audio_samples = audio_samples.mean(axis=1)
            audio_input = np.asarray(audio_samples, dtype=np.float32)
        except Exception:
            # Keep a path fallback for installations without soundfile.
            audio_input = str(audio_path)
        # precise_timing=True (Dubbing): word timestamps + tighter VAD padding so every
        # sentence starts/ends exactly when the person speaks.
        transcribe_kwargs = dict(
            beam_size=int(os.getenv("RECAP_WHISPER_BEAM_SIZE", "5")),
            vad_filter=True,
            condition_on_previous_text=True,
            word_timestamps=bool(precise_timing),
        )
        if precise_timing:
            transcribe_kwargs["vad_parameters"] = {"min_silence_duration_ms": 350, "speech_pad_ms": 120}
        segments_iter, info = model.transcribe(audio_input, **transcribe_kwargs)
        max_seg = float(os.getenv("RECAP_DUB_MAX_SEG", "7.0"))
        pause = float(os.getenv("RECAP_DUB_PAUSE", "0.6"))
        segments: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        last_end = 0.0
        for idx, segment in enumerate(segments_iter):
            text = (segment.text or "").strip()
            if not text:
                continue
            words = [w for w in (getattr(segment, "words", None) or []) if (w.word or "").strip()]
            if precise_timing and words:
                for chunk in self._split_words(words, max_seg, pause):
                    chunk_text = "".join(w.word for w in chunk).strip()
                    if not chunk_text:
                        continue
                    start = max(float(chunk[0].start), last_end)
                    end = max(float(chunk[-1].end), start + 0.05)
                    segments.append({"id": len(segments), "start": round(start, 3), "end": round(end, 3), "text": chunk_text})
                    text_parts.append(chunk_text)
                    last_end = end
            else:
                segments.append({"id": len(segments), "start": float(segment.start), "end": float(segment.end), "text": text})
                text_parts.append(text)
                last_end = float(segment.end)
            if self.progress_callback:
                self.progress_callback(f"Local Whisper transcribe လုပ်နေပါသည်... ({len(segments)} segments)", min(90.0, 20.0 + len(segments) * 0.5))

        full_text = " ".join(text_parts).strip()
        language = getattr(info, "language", "auto") or "auto"
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        result = {"text": full_text, "segments": segments, "language": language, "duration": duration, "precise_timing": bool(precise_timing)}
        (output_dir / "transcript.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_dir / "transcript.txt").open("w", encoding="utf-8") as f:
            f.write(full_text + "\n\n--- SEGMENTS WITH TIMESTAMPS ---\n")
            for seg in segments:
                f.write(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}\n")
        if self.progress_callback:
            self.progress_callback("Local Whisper transcript ပြီးပါပြီ။", 100.0)
        return {"json_path": output_dir / "transcript.json", "txt_path": output_dir / "transcript.txt", **result}
