"""
Gemini (free tier) transcription for Hindi/Hinglish huddles.

One request per ~15-min part returns, per speaker turn, Roman Hinglish + English with the speaker
named from Recall's speaker log — replacing both Sarvam passes. Free-tier handling:
  - busy (503) / per-minute 429  -> wait (Google's RetryInfo when given, else backoff) and retry
  - per-day 429                  -> park that model until the daily reset, fall back to the next model
  - every model parked           -> GeminiQuotaExhausted; the poller retries after the reset
Finished parts are cached in Redis, so a quota stop or instance restart resumes at the next part
instead of re-spending quota on parts already done.
"""
import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone

import httpx
from upstash_redis.asyncio import Redis

from services.clickup_brain import _timeline_segments
from services.transcriber import SarvamTranscriptResult, _ffmpeg_executable

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash,gemini-2.5-flash").split(",") if m.strip()]
GEMINI_PART_SECONDS = int(os.getenv("GEMINI_PART_SECONDS", "900"))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_ATTEMPTS_PER_MODEL = 8
PART_CACHE_TTL = 7 * 24 * 60 * 60

_redis = Redis(url=os.getenv("UPSTASH_REDIS_URL"), token=os.getenv("UPSTASH_REDIS_TOKEN"))


class GeminiQuotaExhausted(Exception):
    """Every configured model has used its daily free quota."""


PROMPT = """This is part {part} of {parts} of an internal team meeting recorded from a Slack huddle (this clip covers {span} of the meeting). People mostly speak Hindi mixed with English (Hinglish); product and technical terms are usually in English. Participants: {names}.

Transcribe this ENTIRE clip from the first word to the last, verbatim. Never summarize, skip, or compress any part.
Output one segment per speaker turn; split turns longer than ~30 seconds into several segments. Short replies ("haan", "theek hai", "hmm") are their own segment with their own speaker.
For each segment:
- start, end: time from the start of THIS clip, MM:SS
- speaker: the participant's exact name from the list above. Decide using BOTH the voice and the platform speaker log below (it shows whose mic was active when, relative to this clip; it can be off by about a second and include short noise blips). Use "Unknown" only if truly unclear.
- hinglish: exactly what was said, Hindi written in Roman script (e.g. "kal tak ho jayega"), English words kept in English. No Devanagari.
- english: faithful, natural English translation of that segment
Write [inaudible] for unclear words. Ignore silence and background noise; never invent speech. If the clip has no speech, return no segments.

Platform speaker log (MM:SS-MM:SS name):
{log}"""


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _parse_ts(value) -> float | None:
    """'MM:SS', 'HH:MM:SS', 'M:SS.s' or plain seconds -> seconds."""
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(str(value).strip().split(":"))))
    except ValueError:
        return None


def _seconds_until_quota_reset(now: datetime | None = None) -> int:
    # ponytail: daily free quotas reset at midnight Pacific = 08:00 UTC in PST; during PDT this
    # parks a model ~1h longer than needed. Use zoneinfo if that hour ever matters.
    now = now or datetime.now(timezone.utc)
    reset = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return int((reset - now).total_seconds()) + 60


def _classify_error(status: int, error: dict) -> tuple[str, float | None]:
    """-> ('daily' | 'retry' | 'next_model', seconds Google asked us to wait)."""
    details = error.get("details") or []
    quota_ids = [v.get("quotaId", "") for d in details for v in (d.get("violations") or []) if isinstance(v, dict)]
    delay = next((d["retryDelay"] for d in details if d.get("retryDelay")), None)
    wait = _parse_ts(str(delay).rstrip("s")) if delay else None
    if status == 429 and any("PerDay" in q for q in quota_ids):
        return "daily", None
    if status in (429, 500, 502, 503, 504):
        return "retry", wait
    return "next_model", None


async def _model_parked(model: str) -> bool:
    try:
        return await _redis.exists(f"gemini_exhausted:{model}") == 1
    except Exception:
        return False


async def gemini_available() -> bool:
    """True if a key is configured and at least one model still has daily quota."""
    if not GEMINI_API_KEY:
        return False
    for model in GEMINI_MODELS:
        if not await _model_parked(model):
            return True
    return False


async def _generate(parts: list, schema: dict, label: str) -> tuple[str, dict]:
    """Run one generateContent request across the model chain. Returns (model, parsed JSON)."""
    async with httpx.AsyncClient(timeout=600) as client:
        for model in GEMINI_MODELS:
            if await _model_parked(model):
                continue
            config = {"temperature": 0, "maxOutputTokens": 65536,
                      "responseMimeType": "application/json", "responseSchema": schema}
            if not model.startswith("gemini-2"):
                config["thinkingConfig"] = {"thinkingLevel": "low"}  # transcription needs no deep reasoning
            body = {"contents": [{"parts": parts}], "generationConfig": config}

            for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
                try:
                    resp = await client.post(GEMINI_URL.format(model=model),
                                             headers={"x-goog-api-key": GEMINI_API_KEY}, json=body)
                except httpx.TransportError as e:
                    kind, wait, message = "retry", None, f"network: {e!r}"
                else:
                    if resp.status_code == 200:
                        data = resp.json()
                        cand = (data.get("candidates") or [{}])[0]
                        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", [])
                                       if not p.get("thought"))
                        try:
                            if cand.get("finishReason") == "STOP":
                                print(f"[Gemini] {label}: {model} ok, usage={data.get('usageMetadata', {}).get('totalTokenCount')} tokens")
                                return model, json.loads(text)
                        except json.JSONDecodeError:
                            pass
                        # Truncated / blocked / malformed output: same request would fail the same way here.
                        print(f"[Gemini] {label}: {model} unusable output (finishReason={cand.get('finishReason')}); next model")
                        break
                    try:
                        error = resp.json().get("error") or {}
                    except ValueError:
                        error = {"message": resp.text[:200]}
                    kind, wait = _classify_error(resp.status_code, error)
                    message = f"HTTP {resp.status_code} {error.get('status', '')}: {str(error.get('message', ''))[:150]}"

                if kind == "daily":
                    ttl = _seconds_until_quota_reset()
                    print(f"[Gemini] {label}: {model} daily quota used up; parked for {ttl // 3600}h. {message}")
                    try:
                        await _redis.set(f"gemini_exhausted:{model}", "1", ex=ttl)
                    except Exception as e:
                        print(f"[Gemini] could not park {model}: {e}")
                    break
                if kind == "next_model":
                    print(f"[Gemini] {label}: {model} failed ({message}); next model")
                    break
                if attempt == MAX_ATTEMPTS_PER_MODEL:
                    print(f"[Gemini] {label}: {model} still failing after {attempt} attempts; next model")
                    break
                wait = wait + 1 if wait else min(120, 10 * 2 ** (attempt - 1))
                print(f"[Gemini] {label}: {model} attempt {attempt} {message} — waiting {wait:.0f}s")
                await asyncio.sleep(wait)

    if not await gemini_available():
        raise GeminiQuotaExhausted("all Gemini models used their daily free quota")
    raise RuntimeError(f"[Gemini] {label}: every model failed")


def _prepare_audio(media_path: str, out_dir: str) -> tuple[str, float, list[float]]:
    """Mono 16 kHz 32 kbps mp3 (Gemini downsamples to 16 kbps anyway) + pause midpoints, in one ffmpeg pass."""
    mp3 = os.path.join(out_dir, "full.mp3")
    proc = subprocess.run(
        [_ffmpeg_executable(), "-y", "-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
         "-af", "silencedetect=noise=-35dB:d=0.6", mp3],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"[Gemini] ffmpeg failed: {proc.stderr[-300:]}")
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", proc.stderr).groups()
    duration = int(h) * 3600 + int(m) * 60 + float(s)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return mp3, duration, [(a + b) / 2 for a, b in zip(starts, ends)]


def _plan_parts(duration: float, pauses: list[float], part_seconds: float = GEMINI_PART_SECONDS,
                window: float = 60.0) -> list[tuple[float, float]]:
    """Cut every ~part_seconds at the pause nearest the target (within ±window) so no word is split.
    The last part may run up to 1.2x long rather than leave a tiny tail."""
    parts, start = [], 0.0
    while duration - start > part_seconds * 1.2:
        target = start + part_seconds
        near = [p for p in pauses if abs(p - target) <= window and p > start + 60]
        cut = min(near, key=lambda p: abs(p - target)) if near else target
        parts.append((start, cut))
        start = cut
    parts.append((start, duration))
    return parts


def _cut(mp3: str, start: float, end: float, out_dir: str, index: int) -> bytes:
    path = os.path.join(out_dir, f"part{index}.mp3")
    subprocess.run([_ffmpeg_executable(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-i", mp3, "-c", "copy", path], check=True)
    with open(path, "rb") as f:
        return f.read()


async def transcribe_with_gemini(
    media_path: str,
    bot_id: str,
    participants: list[str],
    speaker_timeline: list[dict],
) -> tuple[SarvamTranscriptResult, SarvamTranscriptResult]:
    """Returns (English transcript, Roman-Hinglish transcript); entries already carry speaker_name."""
    loop = asyncio.get_running_loop()
    with tempfile.TemporaryDirectory() as tmp:
        mp3, duration, pauses = await loop.run_in_executor(None, _prepare_audio, media_path, tmp)
        plan = _plan_parts(duration, pauses)
        print(f"[Gemini] {duration / 60:.1f} min audio -> {len(plan)} part(s)")

        timeline = _timeline_segments(speaker_timeline, duration)
        names = sorted({n for n in participants + [s["name"] for s in timeline] if n and not n.startswith("aws:")})
        speaker_field = {"type": "string", "enum": names + ["Unknown"]} if names else {"type": "string"}
        schema = {"type": "object", "required": ["segments"], "properties": {"segments": {"type": "array", "items": {
            "type": "object", "required": ["start", "end", "speaker", "hinglish", "english"], "properties": {
                "start": {"type": "string"}, "end": {"type": "string"}, "speaker": speaker_field,
                "hinglish": {"type": "string"}, "english": {"type": "string"}}}}}}

        english, hinglish, models = [], [], set()
        for i, (start, end) in enumerate(plan, 1):
            cache_key = f"gemini_part:{bot_id}:{int(start)}-{int(end)}"
            cached = None
            try:
                cached = await _redis.get(cache_key)
            except Exception as e:
                print(f"[Gemini] cache read failed: {e}")
            if cached:
                result = json.loads(cached)
                print(f"[Gemini] part {i}/{len(plan)}: reused cached result")
            else:
                log = "\n".join(
                    f"{_fmt(max(0.0, s['start'] - start))}-{_fmt(min(end, s['end']) - start)} {s['name']}"
                    for s in timeline if s["end"] > start and s["start"] < end and not s["name"].startswith("aws:")
                ) or "(no speaker activity recorded)"
                prompt = PROMPT.format(part=i, parts=len(plan), span=f"{_fmt(start)}-{_fmt(end)}",
                                       names=", ".join(names) or "unknown", log=log)
                audio = await loop.run_in_executor(None, _cut, mp3, start, end, tmp, i)
                model, out = await _generate(
                    [{"inline_data": {"mime_type": "audio/mp3", "data": base64.b64encode(audio).decode()}},
                     {"text": prompt}],
                    schema, f"part {i}/{len(plan)}",
                )
                result = {"model": model, "segments": out.get("segments") or []}
                try:
                    await _redis.set(cache_key, json.dumps(result, ensure_ascii=False), ex=PART_CACHE_TTL)
                except Exception as e:
                    print(f"[Gemini] cache write failed: {e}")

            models.add(result["model"])
            prev_end = start
            for seg in result["segments"]:
                s = _parse_ts(seg.get("start"))
                e = _parse_ts(seg.get("end"))
                seg_start = min(end, start + s) if s is not None else prev_end
                seg_end = max(seg_start, min(end, start + e)) if e is not None else seg_start
                prev_end = seg_end
                base = {"speaker_id": seg.get("speaker") or "Unknown", "speaker_name": seg.get("speaker") or "Unknown",
                        "start_time_seconds": round(seg_start, 3), "end_time_seconds": round(seg_end, 3)}
                english.append({**base, "transcript": (seg.get("english") or "").strip()})
                hinglish.append({**base, "transcript": (seg.get("hinglish") or "").strip()})

    source = "Gemini " + ", ".join(sorted(models))
    print(f"[Gemini] done: {len(english)} segments via {source}")
    as_text = lambda entries: "\n".join(f"{e['speaker_name']}: {e['transcript']}" for e in entries if e["transcript"])
    return (SarvamTranscriptResult(text=as_text(english), source=source, entries=english),
            SarvamTranscriptResult(text=as_text(hinglish), source=source, entries=hinglish))
