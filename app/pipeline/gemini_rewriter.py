import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

from google import genai
from google.genai import types
from app.pipeline.burmese_text import normalize_myanmar_text

TARGET_LANGUAGE_NAMES = {
    "my": "Burmese (မြန်မာဘာသာ)",
    "en": "English",
    "th": "Thai (ထိုင်းဘာသာ)",
    "ja": "Japanese (ဂျပန်ဘာသာ)",
    "ko": "Korean (ကိုရီးယားဘာသာ)",
    "zh": "Chinese (Mandarin, Simplified)",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "vi": "Vietnamese",
    "hi": "Hindi",
    "id": "Indonesian",
}


class GeminiRewriter:
    """Faithful, natural translation for timed subtitle segments."""

    DEFAULT_MODEL = "gemini-flash-latest"
    FALLBACK_MODEL = "gemini-3.8-flash"
    FALLBACK_MODELS = [
        "gemini-flash-latest",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]

    def __init__(self, api_key: str, progress_callback: Optional[Callable[[str, float], None]] = None):
        if not api_key:
            raise ValueError("Gemini API Key is required. Please set it in Settings.")
        self.client = genai.Client(api_key=api_key)
        self.progress_callback = progress_callback

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned.strip())
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text or "")
            if not match:
                raise ValueError("Model did not return a JSON object.")
            return json.loads(match.group(0))

    @staticmethod
    def _validate_segments(source: List[Dict[str, Any]], output: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if len(source) != len(output):
            return False, f"segment count changed: expected {len(source)}, got {len(output)}"
        for index, (src, dst) in enumerate(zip(source, output)):
            if dst.get("id") != src.get("id"):
                return False, f"segment {index} id changed"
            if abs(float(dst.get("start", -1)) - float(src.get("start", -2))) > 0.01:
                return False, f"segment {index} start timestamp changed"
            if abs(float(dst.get("end", -1)) - float(src.get("end", -2))) > 0.01:
                return False, f"segment {index} end timestamp changed"
            if not isinstance(dst.get("text"), str) or not dst["text"].strip():
                return False, f"segment {index} has empty text"
        return True, "ok"

    @staticmethod
    def _normalise_segments(source: List[Dict[str, Any]], output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep source timing/id authoritative even if the model formats numbers differently."""
        return [
            {
                "id": src["id"],
                "start": float(src["start"]),
                "end": float(src["end"]),
                "text": normalize_myanmar_text(str(dst.get("text", "")).strip()),
            }
            for src, dst in zip(source, output)
        ]

    @staticmethod
    def _normalise_hook(parsed: Dict[str, Any], source_segments: List[Dict[str, Any]], active_mode: str) -> Optional[Dict[str, Any]]:
        """Accept only a short, source-linked hook for Recap/Story modes."""
        if active_mode not in {"recap", "story"}:
            return None
        raw = parsed.get("hook") or {}
        if not isinstance(raw, dict) or raw.get("use_hook") is not True:
            return None
        text = normalize_myanmar_text(str(raw.get("text", "")).strip())
        source_ids = raw.get("source_segment_ids") or []
        valid_ids = {seg.get("id") for seg in source_segments}
        if not text or not isinstance(source_ids, list) or not source_ids:
            return None
        if not all(item in valid_ids for item in source_ids):
            return None
        # Keep the opening hook short enough to speak in roughly three
        # seconds; this is a guardrail in addition to the model instruction.
        if len(text) > 100 or len(source_ids) > 4:
            return None
        return {
            "id": "hook",
            "start": 0.0,
            "end": 0.0,
            "text": text,
            "role": "hook",
            "source_segment_ids": source_ids,
            "hook_type": str(raw.get("hook_type") or "conflict"),
            "confidence": float(raw.get("confidence") or 0.0),
        }

    @staticmethod
    def _dub_budgets(source_segments: List[Dict[str, Any]]) -> Dict[Any, int]:
        """DUB mode: max Burmese characters that can be spoken inside each segment's
        time window (the window may borrow a little of the silent gap before the next one).
        Tune with env RECAP_DUB_CHARS_PER_SEC (default 13)."""
        cps = float(os.getenv("RECAP_DUB_CHARS_PER_SEC", "13"))
        budgets: Dict[Any, int] = {}
        for i, seg in enumerate(source_segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            slot = max(end - start, 0.3)
            if i + 1 < len(source_segments):
                gap_room = float(source_segments[i + 1].get("start", end)) - start - 0.1
            else:
                gap_room = slot + 1.0
            room = min(max(slot, gap_room), slot + 1.5)
            budgets[seg.get("id")] = max(8, int(room * cps))
        return budgets

    def _shorten_overlong(
        self,
        segments: List[Dict[str, Any]],
        budgets: Dict[Any, int],
        lang_name: str,
        model_order: List[str],
        rounds: int = 2,
        tolerance: float = 1.15,
    ) -> List[Dict[str, Any]]:
        """DUB mode: ask the model to shorten only the lines that cannot be spoken in time."""
        current = {seg["id"]: seg["text"] for seg in segments}
        for round_no in range(1, rounds + 1):
            over = [
                {"id": sid, "text": text, "max_chars": budgets.get(sid, 9999)}
                for sid, text in current.items()
                if len(text) > budgets.get(sid, 9999) * tolerance
            ]
            if not over:
                break
            if self.progress_callback:
                self.progress_callback(
                    f"Dubbing အချိန်နဲ့ကိုက်အောင် စာကြောင်း {len(over)} ခုကို တိုနေပါသည် (round {round_no})...", 70.0
                )
            prompt = f"""
Shorten each {lang_name} dubbing line so it can be spoken within its time limit.
Keep the same core meaning, speaker emotion, intensity and names. Natural spoken language.
Do not add new information. Do not merge or reorder lines.
Each "text" MUST be at most "max_chars" Unicode characters. Count carefully and be shorter if unsure.
Return ONLY JSON: {{"segments": [{{"id": 0, "text": "shortened line"}}]}}

INPUT:
{json.dumps(over, ensure_ascii=False)}
"""
            try:
                parsed = self._extract_json(self._generate(prompt, model_order))
            except Exception as exc:
                print(f"[DUB TEXT] shorten round {round_no} failed: {exc}", flush=True)
                break
            for item in parsed.get("segments", []) or []:
                sid = item.get("id")
                new_text = normalize_myanmar_text(str(item.get("text", "")).strip())
                if sid in current and new_text and len(new_text) < len(current[sid]):
                    current[sid] = new_text
        still_over = sum(1 for sid, text in current.items() if len(text) > budgets.get(sid, 9999) * tolerance)
        print(f"[DUB TEXT] {len(current)} lines | still longer than time window: {still_over}", flush=True)
        return [{**seg, "text": current[seg["id"]]} for seg in segments]

    def _generate(self, prompt: str, model_candidates: List[str]) -> str:
        last_err = None
        for candidate in dict.fromkeys(model_candidates):
            for attempt in range(1, 4):
                try:
                    response = self.client.models.generate_content(
                        model=candidate,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.15,
                            response_mime_type="application/json",
                        ),
                    )
                    if response.text:
                        return response.text
                    last_err = RuntimeError(f"{candidate} returned an empty response")
                    break
                except Exception as exc:
                    last_err = exc
                    error_text = str(exc).lower()
                    transient = any(code in error_text for code in ("503", "429", "500", "temporarily unavailable", "overloaded"))
                    if transient and attempt < 3:
                        time.sleep(2 * attempt)
                        continue
                    break
        raise RuntimeError(f"Gemini API processing failed: {last_err}")

    def _available_model_candidates(self) -> List[str]:
        """Add generate-capable Gemini models exposed by the current API key."""
        candidates = list(self.FALLBACK_MODELS)
        try:
            for item in self.client.models.list():
                name = getattr(item, "name", "") or ""
                actions = getattr(item, "supported_actions", None) or []
                if name.startswith("models/"):
                    name = name[7:]
                if name.startswith("gemini") and (not actions or "generateContent" in actions):
                    candidates.append(name)
        except Exception:
            # Model listing is optional; the fixed list still provides fallback.
            pass
        return list(dict.fromkeys(candidates))

    def process(
        self,
        groq_result: Dict[str, Any],
        output_dir: Path,
        mode: str = "translate",
        target_language: str = "my",
        model_name: str = DEFAULT_MODEL,
        source_duration: float = 0.0,
        processing_mode: str = "recap",
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_segments = groq_result.get("segments", [])
        if not source_segments:
            raise ValueError("Transcript has no timed segments to translate.")

        lang_name = TARGET_LANGUAGE_NAMES.get(target_language, target_language)
        source_text = groq_result.get("text", "")
        duration_rule = ""
        active_mode = str(processing_mode or "recap").lower()
        if source_duration > 0 and active_mode == "story":
            duration_rule = f"""
STORY COVERAGE AND LENGTH RULE:
The source video is approximately {source_duration:.1f} seconds long. This is a
full storytelling rewrite, not a short summary, but it must not repeat the source
duration minute-for-minute. Cover the important sequence of events, actions,
decisions, emotions, dangers, reveals, and consequences from the source. Do not
collapse a long video into a tiny recap, but also do not narrate every pause or
repeated visual detail. Target a natural spoken script of roughly 60–70% of the
source duration when the source is long (for example, a 9-minute video should
normally become about 5–6 minutes). Preserve the story's key meaning and suspense.
The renderer will adjust the video to the final TTS duration.
"""
        elif active_mode == "dubbing":
            duration_rule = """
DUBBING TIMING RULE (HIGHEST PRIORITY in Dubbing mode):
Every source segment has a "max_chars" value: the maximum number of Burmese characters
that can be spoken inside that segment's time window while the speaker is talking on screen.
* The translated text of each segment MUST NOT be longer than its max_chars.
* Use short, natural spoken Burmese. Drop filler words, repeated words and minor details.
* If the full literal meaning does not fit, keep the main meaning, emotion and names,
  and shorten the rest. Timing overrides the "do not remove details" rules below.
* Never expand, pad, repeat or add narrator-style commentary.
"""
        elif source_duration > 0:
            duration_rule = f"""
NATURAL LENGTH RULE:
The source narration is approximately {source_duration:.1f} seconds long. Make the
spoken translation only slightly longer than the source, normally by about 10 to
15 seconds and never intentionally more than about 20 seconds longer.
Use natural connective wording only when it clarifies an event already present in
the source. Do not pad the narration to reach one minute, do not target 61 seconds,
and do not repeat a sentence, event, or conclusion to increase the duration.
Keep the original segment ids and timestamps; put the naturally expanded wording
into the corresponding segment texts.
"""
        dub_budgets = self._dub_budgets(source_segments) if active_mode == "dubbing" else {}
        prompt_segments = (
            [{**seg, "max_chars": dub_budgets.get(seg.get("id"), 9999)} for seg in source_segments]
            if dub_budgets else source_segments
        )
        if self.progress_callback:
            self.progress_callback("Transcript အပြည့်ကို ဖတ်ပြီး video အမျိုးအစား ခွဲနေပါသည်...", 10.0)

        story_mode_prompt = """
You are a professional movie/drama/anime recap writer.

Transform the provided video into an engaging Burmese storytelling recap.

Requirements:
- Write in natural conversational Burmese.
- Tell the story naturally, not scene-by-scene summarization.
- Make the audience feel like they are experiencing the events together with the characters.
- Focus on tension, emotions, danger, clever decisions, mistakes, twists, and unexpected moments.
- Maintain strong viewer curiosity throughout the story.
- Use natural storytelling transitions.
- Avoid repetitive phrases and AI-sounding narration.
- Keep the pacing smooth and engaging.
- Build suspense naturally before important reveals.
- Make every paragraph give viewers a reason to continue watching.

Hook Style:
- Do not use generic hooks such as “ဒါပေမယ့် သူ မသိသေးတာက...” or “နောက်ထပ် ဖြစ်လာမယ့်အရာက...”.
- Create curiosity from the actual situation. Hooks may show that a character misunderstands danger, that a decision changes everything, that a hidden detail matters, or that expectations and reality diverge.

Writing Style:
- Natural spoken Burmese; short and clear sentences.
- Emotional but believable, with no exaggerated clickbait.
- No bullet points, chapter labels, or scene-by-scene headings.
- Use one continuous storytelling flow that sounds like a human storyteller, not an AI summary.

Output:
Generate a complete Burmese recap script optimized for YouTube, TikTok, and Facebook storytelling videos with strong retention and natural audience engagement.
"""
        educational_rewrite_prompt = """
You are an expert Burmese educational storyteller and translator.

Transform the source narration into natural, engaging Burmese while preserving all original facts and meaning.

IMPORTANT:
- Do NOT summarize.
- Do NOT skip information.
- Do NOT change the meaning.
- Do NOT invent facts.
- Keep all important details from the original.

GOAL:
Rewrite the narration so it feels more interesting, easier to follow, and more engaging than a direct translation. Create a curious, educational documentary feeling without clickbait.

STYLE:
- Natural spoken Burmese.
- Educational documentary style.
- Curious and engaging tone.
- Slightly conversational, clear, and easy to understand.
- Sound like a human narrator.

RETENTION RULES:
- Add smooth transitions between ideas.
- Briefly create curiosity before explaining a new concept.
- Highlight surprising facts naturally.
- Emphasize cause-and-effect relationships.
- Make technical explanations easier to visualize.
- Slightly expand explanations only when needed for clarity.

LENGTH:
- The Burmese version may be up to 10-20% longer than the original when needed for clarity and flow.
- Never make it shorter by removing information.

HOOK RULE:
- The first 1-3 sentences should create immediate curiosity about the topic.
- Use a source-grounded hook, not clickbait.
- Do not invent a mystery, fact, event, or ending that is absent from the source.
"""
        mode_instructions = {
            "recap": educational_rewrite_prompt,
            "story": story_mode_prompt,
            "dubbing": "Translate as natural spoken dubbing for the original scene. Preserve each speaker's meaning, emotion, intensity, and timing; do not summarize or turn dialogue into a narrator recap.",
        }.get(str(processing_mode or "recap").lower(), "Rewrite as a natural movie-recap narration.")
        prompt = f"""
Translate the transcript into natural spoken {lang_name}.
MODE: {processing_mode}
MODE INSTRUCTION: {mode_instructions}

Rules:

* Translate the MEANING, not word-for-word.
* Make it sound like a native speaker naturally telling a story, NOT like a book or machine translation.
* Do not copy the original sentence structure if it sounds unnatural.
* Do not summarize or remove important information.
* Do not add explanations, details, opinions, or information that is not in the original.
* Follow the mode-specific timing and coverage rule below; do not pad or repeat content.
* Use natural conversational grammar and expressions.
* Avoid overly formal/literary language.
* Avoid unnecessary pronouns and words such as “၎င်း”, “၎င်းတို့”, “ဖြစ်သည်”, “ဖြစ်ကြသည်” in Burmese when they make the sentence sound unnatural.
* Preserve names, numbers, actions, events, and important details accurately.
* Make every sentence smooth and easy to understand when heard through TTS.
* Think like a native speaker explaining what happened in a movie to a friend.
* Translate the story, not the words.
{duration_rule}

SOURCE-COVERAGE SAFETY:
* Translate every sentence and every proposition in every source segment.
* The output for each segment must carry the complete meaning of its corresponding source segment.
* Do not remove details, qualifiers, uncertainty, repetition that carries emphasis, or emotional meaning.
* Keep every segment. Do not merge, split, reorder, or invent segments.
* Keep every input id, start, and end EXACTLY unchanged.
* Use the full transcript only to resolve pronouns or context; never use it to summarize segments.

OUTPUT: Return ONLY valid JSON with this exact shape:
{{
  "content_type": "entertainment|educational|emotional_story|news_documentary|conversation",
  "tone": "short description",
  "glossary": [{{"source": "term", "target": "translation"}}],
  "full_text": "joined translated text",
  "hook": {{
    "use_hook": true,
    "text": "1 to 3 short source-grounded Burmese sentences",
    "source_segment_ids": [0],
    "hook_type": "danger|mystery|conflict|unexpected_change|emotion|none",
    "confidence": 0.0
  }},
  "segments": [
    {{"id": 0, "start": 0.0, "end": 2.5, "text": "translation"}}
  ]
}}

SOURCE FULL TRANSCRIPT:
{source_text}

SOURCE TIMED SEGMENTS:
{json.dumps(prompt_segments, ensure_ascii=False, indent=2)}

HOOK RULES (Recap and Story only):
* First inspect the full transcript and choose the strongest source-supported conflict,
  danger, mystery, unexpected change, or turning point.
* Write one very short natural Burmese opening hook from that event, without inventing facts.
* Do not reveal the complete ending, identity reveal, or full twist.
* The hook is a preview, not a new event and not a summary of the whole video.
* Link it to the exact source segment ids used.
* Target approximately 3 seconds when spoken: one short sentence or two very short clauses,
  normally no more than about 100 Unicode characters. If it cannot fit naturally, return
  use_hook=false and an empty text.
* If no clear supportable hook exists, return use_hook=false and an empty text.
* For Dubbing mode, always return use_hook=false; translate original dialogue only.
* The main segments must still preserve every source segment and must not be removed.
"""

        if self.progress_callback:
            self.progress_callback("Context-aware ဘာသာပြန်နေပါသည်... (meaning ကို ထိန်းထားပါသည်)", 35.0)
        # Try all known Flash fallbacks, then models exposed by this API key.
        model_order = self._available_model_candidates()
        response_text = self._generate(prompt, model_order)
        try:
            parsed = self._extract_json(response_text)
            translated = parsed.get("segments", [])
            valid, reason = self._validate_segments(source_segments, translated)
        except Exception as exc:
            valid, reason, parsed = False, str(exc), {}

        if not valid:
            if self.progress_callback:
                self.progress_callback("ဘာသာပြန်ရလဒ်ကို မူရင်း timestamp နဲ့ ပြန်စစ်နေပါသည်...", 62.0)
            repair_prompt = f"""
Repair this translation JSON without rewriting the wording.
Return ONLY JSON with content_type, tone, glossary, full_text, and segments.
The segments list MUST have exactly the same count as SOURCE, and each id/start/end
MUST be copied exactly from SOURCE. Only fix missing/invalid structure; preserve text.
SOURCE={json.dumps(source_segments, ensure_ascii=False)}
BAD_RESULT={json.dumps(parsed, ensure_ascii=False)}
VALIDATION_ERROR={reason}
"""
            repaired = self._extract_json(self._generate(repair_prompt, model_order))
            translated = repaired.get("segments", [])
            valid, reason = self._validate_segments(source_segments, translated)
            parsed = repaired

        if not valid:
            raise RuntimeError(f"Translation validation failed: {reason}")

        processed_segments = self._normalise_segments(source_segments, translated)
        if dub_budgets:
            processed_segments = self._shorten_overlong(processed_segments, dub_budgets, lang_name, model_order)
        hook = self._normalise_hook(parsed, source_segments, active_mode)
        if hook:
            processed_segments = [hook, *processed_segments]
        processed_full_text = "\n".join(s["text"] for s in processed_segments)
        metadata = {
            "content_type": parsed.get("content_type", "conversation"),
            "tone": parsed.get("tone", "natural and faithful"),
            "glossary": parsed.get("glossary", []),
            "source_segment_count": len(source_segments),
            "meaning_policy": "faithful_natural_translation_no_summarization",
            "model_order": model_order,
            "hook": hook,
        }

        if self.progress_callback:
            self.progress_callback("ဘာသာပြန်ရလဒ်ကို အဓိပ္ပာယ်/အရှည် စစ်ဆေးနေပါသည်...", 85.0)

        (output_dir / "translation_analysis.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        processed_txt_path = output_dir / "processed_transcript.txt"
        processed_txt_path.write_text(
            processed_full_text + "\n\n--- PROCESSED SEGMENTS WITH TIMESTAMPS ---\n" +
            "".join(f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}\n" for s in processed_segments),
            encoding="utf-8",
        )
        processed_json_path = output_dir / "processed_transcript.json"
        processed_json_path.write_text(
            json.dumps({**metadata, "full_text": processed_full_text, "segments": processed_segments,
                        "target_language": target_language, "mode": mode}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.progress_callback:
            self.progress_callback("Controlled natural ဘာသာပြန်ပြီးပါပြီ။", 100.0)
        return {
            "txt_path": processed_txt_path,
            "json_path": processed_json_path,
            "full_text": processed_full_text,
            "segments": processed_segments,
            "hook": hook,
            "analysis": metadata,
        }
