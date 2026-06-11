import asyncio
import httpx
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# All 4 Sarvam API keys — rotates to next when credits exhausted (402)
SARVAM_KEYS = [
    k for k in [
        os.getenv("SARVAM_API_KEY"),
        os.getenv("SARVAM_API_KEY_1"),
        os.getenv("SARVAM_API_KEY_2"),
        os.getenv("SARVAM_API_KEY_3"),
    ] if k
]
_sarvam_key_index = 0   # tracks which key is currently active

SARVAM_URL = "https://api.sarvam.ai/speech-to-text-translate"

CHUNK_SECONDS = 25   # Sarvam AI limit is 30s, use 25s to be safe
MAX_RETRIES = 8      # max retries on 429 rate limit
MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("SARVAM_MIN_REQUEST_INTERVAL_SECONDS", "8"))
EXHAUSTED_LOG_INTERVAL_SECONDS = int(os.getenv("SARVAM_EXHAUSTED_LOG_INTERVAL_SECONDS", "60"))
SARVAM_BATCH_MODEL = os.getenv("SARVAM_BATCH_MODEL", "saaras:v3")
SARVAM_BATCH_MODE = os.getenv("SARVAM_BATCH_MODE", "translate")
SARVAM_BATCH_LANGUAGE_CODE = os.getenv("SARVAM_BATCH_LANGUAGE_CODE", "hi-IN")
SARVAM_BATCH_PART_SECONDS = int(os.getenv("SARVAM_BATCH_PART_SECONDS", "3300"))
SARVAM_BATCH_POLL_SECONDS = int(os.getenv("SARVAM_BATCH_POLL_SECONDS", "10"))
SARVAM_BATCH_TIMEOUT_SECONDS = int(os.getenv("SARVAM_BATCH_TIMEOUT_SECONDS", "7200"))
SARVAM_BATCH_UPLOAD_TIMEOUT_SECONDS = int(os.getenv("SARVAM_BATCH_UPLOAD_TIMEOUT_SECONDS", "900"))
SARVAM_BATCH_MAX_RETRIES = int(os.getenv("SARVAM_BATCH_MAX_RETRIES", "4"))
SARVAM_NUM_SPEAKERS = os.getenv("SARVAM_NUM_SPEAKERS")

_sarvam_request_lock = asyncio.Lock()
_last_sarvam_request_at = 0.0
_sarvam_all_keys_exhausted = False
_sarvam_exhaustion_logger_task: asyncio.Task | None = None


@dataclass
class SarvamTranscriptResult:
    text: str
    source: str
    entries: list[dict] = field(default_factory=list)
    timestamps: dict | None = None
    raw: dict | list | None = None


async def _log_sarvam_exhaustion_until_recovered():
    while _sarvam_all_keys_exhausted:
        print(
            "[Sarvam] ALL API KEY CREDITS EXHAUSTED. "
            "Add credits or configure another SARVAM_API_KEY. "
            f"Configured keys: {len(SARVAM_KEYS)}"
        )
        await asyncio.sleep(EXHAUSTED_LOG_INTERVAL_SECONDS)


def _start_sarvam_exhaustion_logger():
    global _sarvam_all_keys_exhausted, _sarvam_exhaustion_logger_task
    _sarvam_all_keys_exhausted = True
    if _sarvam_exhaustion_logger_task and not _sarvam_exhaustion_logger_task.done():
        return
    _sarvam_exhaustion_logger_task = asyncio.create_task(_log_sarvam_exhaustion_until_recovered())


def _clear_sarvam_exhaustion_logger():
    global _sarvam_all_keys_exhausted, _sarvam_exhaustion_logger_task
    if not _sarvam_all_keys_exhausted:
        return
    _sarvam_all_keys_exhausted = False
    if _sarvam_exhaustion_logger_task and not _sarvam_exhaustion_logger_task.done():
        _sarvam_exhaustion_logger_task.cancel()
    _sarvam_exhaustion_logger_task = None
    print("[Sarvam] API key credits available again. Continuing transcription.")


def _retry_after_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), MIN_REQUEST_INTERVAL_SECONDS)
        except ValueError:
            pass
    return min(15 * (2 ** (attempt - 1)), 300)


async def _post_to_sarvam(headers: dict, files: dict, data: dict) -> httpx.Response:
    global _last_sarvam_request_at
    async with _sarvam_request_lock:
        now = asyncio.get_running_loop().time()
        wait = MIN_REQUEST_INTERVAL_SECONDS - (now - _last_sarvam_request_at)
        if wait > 0:
            await asyncio.sleep(wait)

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                SARVAM_URL, headers=headers, files=files, data=data
            )
        _last_sarvam_request_at = asyncio.get_running_loop().time()
        return response


def _ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError(
            "ffmpeg is required for transcription. Install ffmpeg or install imageio-ffmpeg."
        ) from exc


def _split_audio_segments_sync(media_path: str, segment_seconds: int, prefix: str) -> dict:
    work_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
    try:
        full_mp3_path = os.path.join(work_dir, "source_full.mp3")
        ffmpeg = _ffmpeg_executable()
        subprocess.run(
            [ffmpeg, "-y", "-i", media_path, "-vn", "-acodec", "mp3", "-q:a", "5", full_mp3_path],
            check=True,
            capture_output=True,
        )
        pattern = os.path.join(work_dir, "part_%04d.mp3")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                full_mp3_path,
                "-f",
                "segment",
                "-segment_time",
                str(segment_seconds),
                "-reset_timestamps",
                "1",
                "-acodec",
                "copy",
                pattern,
            ],
            check=True,
            capture_output=True,
        )
        paths = sorted(str(path) for path in Path(work_dir).glob("part_*.mp3"))
        if not paths:
            raise RuntimeError("ffmpeg produced no audio segments")
        offsets = {
            os.path.basename(path): float(index * segment_seconds)
            for index, path in enumerate(paths)
        }
        return {"work_dir": work_dir, "paths": paths, "offsets": offsets}
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def _extract_chunks_sync(media_path: str) -> list[str]:
    """
    Blocking function — runs in thread executor so it doesn't freeze the event loop.
    Converts media file to MP3 chunks of CHUNK_SECONDS each.
    Works for both MP4 (video) and MP3 (audio) input.
    """
    prepared = _split_audio_segments_sync(media_path, CHUNK_SECONDS, "sarvam_rest")
    return prepared["paths"]

    base = media_path.rsplit(".", 1)[0]
    full_mp3_path = base + "_full.mp3"
    chunk_paths = []

    try:
        # Extract/convert to full MP3 (-q:a 5 = low bitrate, keeps file small)
        subprocess.run(
            ["ffmpeg", "-y", "-i", media_path, "-vn", "-acodec", "mp3", "-q:a", "5", full_mp3_path],
            check=True, capture_output=True
        )

        # Get duration in seconds
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", full_mp3_path],
            capture_output=True, text=True, check=True
        )
        duration_str = result.stdout.strip()
        if not duration_str:
            raise Exception("ffprobe returned empty duration — file may be corrupt or silent-only")
        duration = float(duration_str)

        # Split into CHUNK_SECONDS chunks
        start = 0
        i = 0
        while start < duration:
            chunk_path = base + f"_chunk{i}.mp3"
            subprocess.run(
                ["ffmpeg", "-y", "-i", full_mp3_path, "-ss", str(start),
                 "-t", str(CHUNK_SECONDS), "-acodec", "copy", chunk_path],
                check=True, capture_output=True
            )
            chunk_paths.append(chunk_path)
            start += CHUNK_SECONDS
            i += 1

    except Exception:
        # Clean up any partial files before re-raising
        for path in chunk_paths:
            if os.path.exists(path):
                os.unlink(path)
        raise

    finally:
        if os.path.exists(full_mp3_path):
            os.unlink(full_mp3_path)

    return chunk_paths


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def _entry_text(entry: dict) -> str:
    for key in ("transcript", "text", "translated_text", "sentence", "utterance"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _entry_speaker(entry: dict) -> str:
    for key in ("speaker_id", "speaker", "speaker_label", "speaker_tag", "channel"):
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _error_payload(response: httpx.Response | None) -> dict:
    if response is None:
        return {}
    try:
        data = response.json()
    except Exception:
        return {"message": response.text}
    return data if isinstance(data, dict) else {"message": str(data)}


def classify_sarvam_batch_error(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    payload = _error_payload(response)
    payload_text = json.dumps(payload, default=str).lower()
    message = str(exc).lower()
    combined = f"{payload_text} {message}"

    if status_code == 402 or "insufficient_quota" in combined or "credit" in combined:
        return "credit_exhausted"
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "retryable"
    if status_code in {429, 500, 502, 503, 504}:
        return "retryable"
    if any(token in combined for token in ("rate_limit", "timed out", "timeout", "temporarily", "503", "429")):
        return "retryable"
    return "fatal"


def _time_value(entry: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in entry:
            continue
        value = entry.get(key)
        if isinstance(value, dict):
            for nested_key in ("seconds", "relative", "time", "timestamp"):
                nested = value.get(nested_key)
                if nested is not None:
                    value = nested
                    break
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if key.endswith("_ms") or key.endswith("Ms") or seconds > 86_400:
            seconds = seconds / 1000.0
        return seconds
    return None


def _normalise_segment_list(items: list, offset_seconds: float) -> list[dict]:
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _entry_text(item)
        start = _time_value(
            item,
            (
                "start_time_seconds",
                "start_time",
                "start_seconds",
                "start",
                "start_ms",
                "startTime",
                "startTimestamp",
            ),
        )
        end = _time_value(
            item,
            (
                "end_time_seconds",
                "end_time",
                "end_seconds",
                "end",
                "end_ms",
                "endTime",
                "endTimestamp",
            ),
        )
        if not text or start is None:
            continue
        if end is None or end < start:
            end = start
        entries.append(
            {
                "speaker_id": _entry_speaker(item),
                "start_time_seconds": round(start + offset_seconds, 3),
                "end_time_seconds": round(end + offset_seconds, 3),
                "transcript": text,
            }
        )
    return entries


def _find_segment_entries(payload, offset_seconds: float) -> list[dict]:
    if isinstance(payload, dict):
        preferred_keys = (
            "diarized_transcript",
            "diarizedTranscript",
            "speaker_transcript",
            "speakerTranscript",
            "entries",
            "utterances",
            "segments",
            "results",
            "transcripts",
        )
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                entries = _normalise_segment_list(value, offset_seconds)
                if entries:
                    return entries
        for value in payload.values():
            entries = _find_segment_entries(value, offset_seconds)
            if entries:
                return entries
    elif isinstance(payload, list):
        entries = _normalise_segment_list(payload, offset_seconds)
        if entries:
            return entries
        for value in payload:
            entries = _find_segment_entries(value, offset_seconds)
            if entries:
                return entries
    return []


def _payload_text(payload) -> str:
    if isinstance(payload, dict):
        for key in ("transcript", "text", "translated_text", "translation"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            text = _payload_text(value)
            if text:
                return text
    elif isinstance(payload, list):
        parts = [_payload_text(value) for value in payload]
        return " ".join(part for part in parts if part).strip()
    return ""


def _coalesce_entries(entries: list[dict], max_gap_seconds: float = 0.8) -> list[dict]:
    if not entries:
        return entries
    entries = sorted(entries, key=lambda item: item.get("start_time_seconds", 0.0))
    merged = [dict(entries[0])]
    for entry in entries[1:]:
        last = merged[-1]
        same_speaker = last.get("speaker_id") == entry.get("speaker_id")
        gap = float(entry.get("start_time_seconds", 0.0)) - float(last.get("end_time_seconds", 0.0))
        if same_speaker and gap <= max_gap_seconds:
            last["end_time_seconds"] = entry.get("end_time_seconds", last.get("end_time_seconds"))
            last["transcript"] = f"{last.get('transcript', '').strip()} {entry.get('transcript', '').strip()}".strip()
        else:
            merged.append(dict(entry))
    return merged


def normalize_sarvam_batch_payload(payload, offset_seconds: float = 0.0) -> SarvamTranscriptResult:
    entries = _coalesce_entries(_find_segment_entries(payload, offset_seconds))
    text = _payload_text(payload)
    if not text and entries:
        text = " ".join(entry["transcript"] for entry in entries if entry.get("transcript")).strip()
    return SarvamTranscriptResult(text=text, source="batch", entries=entries, raw=payload)


def parse_sarvam_output_content(content: str, filename: str):
    if filename.lower().endswith(".txt"):
        return {"transcript": content.strip()}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"transcript": content.strip()}


def combine_sarvam_batch_outputs(
    output_payloads: list[tuple[str, dict | list]],
    offsets_by_input_file: dict[str, float],
) -> SarvamTranscriptResult:
    combined_entries = []
    combined_text = []
    raw_payloads = []

    for input_file, payload in output_payloads:
        offset = offsets_by_input_file.get(input_file)
        if offset is None:
            raise RuntimeError(f"Missing batch offset for Sarvam input file: {input_file}")
        result = normalize_sarvam_batch_payload(payload, offset_seconds=offset)
        combined_entries.extend(result.entries)
        if result.text:
            combined_text.append(result.text)
        raw_payloads.append(payload)

    text = " ".join(combined_text).strip()
    entries = sorted(combined_entries, key=lambda item: item.get("start_time_seconds", 0.0))
    if not text and entries:
        text = " ".join(entry["transcript"] for entry in entries if entry.get("transcript")).strip()

    return SarvamTranscriptResult(
        text=text,
        source="batch",
        entries=entries,
        raw=raw_payloads,
    )


def _media_duration_seconds(media_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            media_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = result.stdout.strip()
    if not duration:
        raise Exception("ffprobe returned empty duration")
    return float(duration)


def _prepare_batch_audio_parts_sync(media_path: str) -> dict:
    prepared = _split_audio_segments_sync(media_path, SARVAM_BATCH_PART_SECONDS, "sarvam_batch")
    prepared["duration"] = 0.0
    return prepared

    work_dir = tempfile.mkdtemp(prefix="sarvam_batch_")
    try:
        full_mp3_path = os.path.join(work_dir, "source_full.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", media_path, "-vn", "-acodec", "mp3", "-q:a", "5", full_mp3_path],
            check=True,
            capture_output=True,
        )
        duration = _media_duration_seconds(full_mp3_path)
        paths = []
        offsets = {}

        if SARVAM_BATCH_PART_SECONDS > 0 and duration > SARVAM_BATCH_PART_SECONDS:
            start = 0.0
            index = 0
            while start < duration:
                part_path = os.path.join(work_dir, f"part_{index:04d}.mp3")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        full_mp3_path,
                        "-ss",
                        str(start),
                        "-t",
                        str(SARVAM_BATCH_PART_SECONDS),
                        "-acodec",
                        "copy",
                        part_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                paths.append(part_path)
                offsets[os.path.basename(part_path)] = start
                start += SARVAM_BATCH_PART_SECONDS
                index += 1
        else:
            paths.append(full_mp3_path)
            offsets[os.path.basename(full_mp3_path)] = 0.0

        return {"work_dir": work_dir, "paths": paths, "offsets": offsets, "duration": duration}
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def _run_sarvam_batch_job_sync(
    media_path: str,
    api_key: str,
    batch_mode: str | None = None,
) -> SarvamTranscriptResult:
    try:
        from sarvamai import SarvamAI
    except ImportError as exc:
        raise RuntimeError("sarvamai package is required for SARVAM_TRANSCRIPTION_MODE=batch") from exc

    prepared = _prepare_batch_audio_parts_sync(media_path)
    work_dir = prepared["work_dir"]
    try:
        client = SarvamAI(api_subscription_key=api_key, timeout=SARVAM_BATCH_TIMEOUT_SECONDS)
        kwargs = {
            "model": SARVAM_BATCH_MODEL,
            "mode": batch_mode or SARVAM_BATCH_MODE,
            "language_code": SARVAM_BATCH_LANGUAGE_CODE,
            "with_diarization": True,
            "with_timestamps": True,
        }
        if SARVAM_NUM_SPEAKERS:
            kwargs["num_speakers"] = int(SARVAM_NUM_SPEAKERS)

        job = client.speech_to_text_job.create_job(**kwargs)
        print(f"[Sarvam Batch] Created job {job.job_id}; uploading {len(prepared['paths'])} file(s)")
        job.upload_files(prepared["paths"], timeout=SARVAM_BATCH_UPLOAD_TIMEOUT_SECONDS)
        job.start()
        final_status = job.wait_until_complete(
            poll_interval=SARVAM_BATCH_POLL_SECONDS,
            timeout=SARVAM_BATCH_TIMEOUT_SECONDS,
        )
        if job.is_failed() or not job.is_successful():
            raise RuntimeError(f"Sarvam batch job failed: {json.dumps(final_status, default=_json_default)}")

        output_dir = os.path.join(work_dir, "outputs")
        mappings = job.get_output_mappings()
        job.download_outputs(output_dir)
        output_payloads = []
        for mapping in mappings:
            input_file = mapping.get("input_file")
            output_file = mapping.get("output_file")
            candidates = [
                Path(output_dir) / f"{input_file}.json",
                Path(output_dir) / f"{output_file}.json",
                Path(output_dir) / str(output_file or ""),
            ]
            output_path = next((path for path in candidates if path.exists() and path.is_file()), None)
            if not output_path:
                continue
            with open(output_path, "r", encoding="utf-8") as f:
                output_payloads.append((input_file, parse_sarvam_output_content(f.read(), output_path.name)))

        if not output_payloads:
            output_files = sorted(path for path in Path(output_dir).iterdir() if path.is_file())
            output_payloads = []
            for path in output_files:
                input_name = path.name[:-5] if path.name.endswith(".json") else path.name
                with open(path, "r", encoding="utf-8") as f:
                    output_payloads.append((input_name, parse_sarvam_output_content(f.read(), path.name)))

        if not output_payloads:
            raise RuntimeError("Sarvam batch job completed but returned no JSON output files")

        result = combine_sarvam_batch_outputs(output_payloads, prepared["offsets"])
        if not result.text.strip():
            raise RuntimeError("Sarvam batch job returned an empty transcript")
        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def _transcribe_audio_batch(
    media_path: str,
    batch_mode: str | None = None,
) -> SarvamTranscriptResult:
    global _sarvam_key_index
    if not SARVAM_KEYS:
        raise Exception("[Sarvam Batch] No API keys configured. Set SARVAM_API_KEY or SARVAM_API_KEY_1..3.")
    errors = []
    for attempt in range(1, SARVAM_BATCH_MAX_RETRIES + 1):
        keys_tried = 0
        exhausted_count = 0
        while keys_tried < len(SARVAM_KEYS):
            key_slot = _sarvam_key_index + 1
            try:
                result = await asyncio.to_thread(
                    _run_sarvam_batch_job_sync,
                    media_path,
                    SARVAM_KEYS[_sarvam_key_index],
                    batch_mode,
                )
                _clear_sarvam_exhaustion_logger()
                return result
            except Exception as exc:
                error_kind = classify_sarvam_batch_error(exc)
                errors.append(f"attempt {attempt}, key {key_slot}: {exc}")
                keys_tried += 1

                if error_kind == "credit_exhausted":
                    exhausted_count += 1
                    print(f"[Sarvam Batch] API key {key_slot}/{len(SARVAM_KEYS)} credits exhausted.")
                    _sarvam_key_index = (_sarvam_key_index + 1) % len(SARVAM_KEYS)
                    continue

                if error_kind == "retryable":
                    print(
                        f"[Sarvam Batch] Transient error on key {key_slot}/{len(SARVAM_KEYS)} "
                        f"(attempt {attempt}/{SARVAM_BATCH_MAX_RETRIES}): {exc}"
                    )
                    _sarvam_key_index = (_sarvam_key_index + 1) % len(SARVAM_KEYS)
                    break

                raise

        if exhausted_count == len(SARVAM_KEYS):
            _start_sarvam_exhaustion_logger()
            raise Exception("[Sarvam Batch] ALL API KEY CREDITS EXHAUSTED. " + " | ".join(errors))

        if attempt < SARVAM_BATCH_MAX_RETRIES:
            wait = min(30 * (2 ** (attempt - 1)), 300)
            print(f"[Sarvam Batch] Waiting {wait}s before retry {attempt + 1}/{SARVAM_BATCH_MAX_RETRIES}.")
            await asyncio.sleep(wait)

    raise Exception("[Sarvam Batch] failed after retries. " + " | ".join(errors))


async def transcribe_audio_batch_mode(media_path: str, batch_mode: str) -> SarvamTranscriptResult:
    return await _transcribe_audio_batch(media_path, batch_mode=batch_mode)


async def _transcribe_audio_rest_detailed(media_path: str) -> SarvamTranscriptResult:
    """
    Accepts a file path to MP4 or MP3.
    Splits into 25s chunks in a thread executor (non-blocking).
    Sends each chunk to Sarvam AI with rate limit handling.
    Returns full English transcript.
    """
    loop = asyncio.get_event_loop()

    if not SARVAM_KEYS:
        raise Exception("[Sarvam] No API keys configured. Set SARVAM_API_KEY or SARVAM_API_KEY_1..3.")

    # Run blocking ffmpeg in thread pool — won't freeze the server
    chunk_paths = await loop.run_in_executor(None, _extract_chunks_sync, media_path)
    print(f"[Sarvam] Split into {len(chunk_paths)} chunks")

    transcripts = []

    try:
        for i, chunk_path in enumerate(chunk_paths):
            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()
            os.unlink(chunk_path)
            chunk_paths[i] = None  # mark as cleaned up

            print(f"[Sarvam] Sending chunk {i+1}/{len(chunk_paths)}...")

            files = {"file": ("chunk.mp3", chunk_bytes, "audio/mpeg")}
            data = {
                "model": "saaras:v3",
                "language_code": "hi-IN",
                "target_language_code": "en-IN"
            }

            global _sarvam_key_index
            for attempt in range(1, MAX_RETRIES + 1):
                keys_tried = 0
                exhausted_count = 0
                response = None
                while keys_tried < len(SARVAM_KEYS):
                    key_slot = _sarvam_key_index + 1
                    headers = {"api-subscription-key": SARVAM_KEYS[_sarvam_key_index]}
                    response = await _post_to_sarvam(headers, files, data)
                    keys_tried += 1

                    # Credits exhausted on this key — rotate to next.
                    if response.status_code == 402:
                        exhausted_count += 1
                        print(f"[Sarvam] API key {key_slot}/{len(SARVAM_KEYS)} credits exhausted.")
                        _sarvam_key_index = (_sarvam_key_index + 1) % len(SARVAM_KEYS)
                        if keys_tried < len(SARVAM_KEYS):
                            print(f"[Sarvam] Switching to API key {_sarvam_key_index + 1}/{len(SARVAM_KEYS)}...")
                        continue

                    # Rate limited on this key — rotate to next key immediately
                    # instead of sleeping on the same one.
                    if response.status_code == 429:
                        print(f"[Sarvam] API key {key_slot}/{len(SARVAM_KEYS)} rate limited (429).")
                        _sarvam_key_index = (_sarvam_key_index + 1) % len(SARVAM_KEYS)
                        if keys_tried < len(SARVAM_KEYS):
                            print(f"[Sarvam] Switching to API key {_sarvam_key_index + 1}/{len(SARVAM_KEYS)}...")
                        continue

                    # Usable response (200 or non-rotation error) — exit inner loop.
                    _clear_sarvam_exhaustion_logger()
                    break
                else:
                    # Every configured key tried this attempt — all 402 and/or 429.
                    if exhausted_count == len(SARVAM_KEYS):
                        _start_sarvam_exhaustion_logger()
                        raise Exception("[Sarvam] ALL API KEY CREDITS EXHAUSTED. Add credits or configure another SARVAM_API_KEY.")
                    # All keys rate limited (or mix of 402 + 429). Back off, then retry.
                    wait = _retry_after_seconds(response, attempt)
                    print(f"[Sarvam] All {len(SARVAM_KEYS)} keys rate limited on chunk {i+1}. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    print(f"[Sarvam Error] Chunk {i+1}: {response.status_code}: {response.text}")
                response.raise_for_status()
                break
            else:
                raise Exception(f"[Sarvam] Chunk {i+1} failed after {MAX_RETRIES} retries.")

            transcripts.append(response.json().get("transcript", ""))

    finally:
        # Clean up any leftover chunk files if we crashed partway through
        for path in chunk_paths:
            if path and os.path.exists(path):
                os.unlink(path)

    return SarvamTranscriptResult(text=" ".join(transcripts), source="rest_chunked", entries=[])


async def transcribe_audio_detailed(media_path: str) -> SarvamTranscriptResult:
    mode = os.getenv("SARVAM_TRANSCRIPTION_MODE", "batch_with_rest_fallback").strip().lower()
    if mode in {"batch", "batch_with_rest_fallback", "auto"}:
        try:
            return await _transcribe_audio_batch(media_path)
        except Exception as exc:
            if mode == "batch":
                raise
            print(f"[Sarvam Batch] Failed, falling back to existing REST chunk flow: {exc}")

    return await _transcribe_audio_rest_detailed(media_path)


async def transcribe_audio(media_path: str) -> str:
    return (await transcribe_audio_detailed(media_path)).text
