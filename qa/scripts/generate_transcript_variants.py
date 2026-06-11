import argparse
import asyncio
import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import transcriber
from services.clickup_brain import assign_participant_names, format_speaker_transcript


RECALL_BASE_URL = "https://ap-northeast-1.recall.ai/api/v1"
DEFAULT_VALIDATION_MODEL = os.getenv("HINGLISH_VALIDATION_MODEL", "gpt-5.1")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_bot_details(bot_id: str) -> dict:
    headers = {
        "Authorization": f"Token {os.getenv('RECALL_API_KEY')}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(f"{RECALL_BASE_URL}/bot/{bot_id}/", headers=headers)
        response.raise_for_status()
        return response.json()


def pick_media_url(details: dict) -> str:
    for recording in details.get("recordings", []) or []:
        shortcuts = recording.get("media_shortcuts", {}) or {}
        audio = shortcuts.get("audio_mixed") or {}
        video = shortcuts.get("video_mixed") or {}
        media_url = (
            (audio.get("data") or {}).get("download_url")
            or (video.get("data") or {}).get("download_url")
        )
        if media_url:
            return media_url
    raise RuntimeError("No Recall media download URL found")


async def fetch_participant_artifacts(details: dict) -> tuple[list[str], list[dict]]:
    recordings = details.get("recordings", []) or []
    if not recordings:
        return [], []

    participant_events = (
        (recordings[0].get("media_shortcuts", {}) or {}).get("participant_events") or {}
    )
    participant_data = participant_events.get("data") or {}
    participants_url = participant_data.get("participants_download_url", "")
    speaker_timeline_url = (
        participant_data.get("speaker_timeline_download_url")
        or participant_data.get("speakerTimeline_download_url")
        or participant_data.get("speaker_timeline_url")
        or ""
    )

    participants = []
    speaker_timeline = []
    async with httpx.AsyncClient(timeout=60) as client:
        if participants_url:
            response = await client.get(participants_url)
            response.raise_for_status()
            payload = response.json()
            participants = [
                item.get("name", "")
                for item in payload
                if isinstance(item, dict) and item.get("name")
            ]

        if speaker_timeline_url:
            response = await client.get(speaker_timeline_url)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                speaker_timeline = payload
            elif isinstance(payload, dict):
                for key in ("speaker_timeline", "timeline", "events", "results", "data"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        speaker_timeline = value
                        break

    return participants, speaker_timeline


def build_metadata(bot_id: str, details: dict, participants: list[str]) -> dict:
    recordings = details.get("recordings", []) or []
    ended_at = recordings[0].get("completed_at", "") if recordings else ""
    duration_seconds = None
    duration_minutes = 0
    try:
        if recordings:
            start_raw = recordings[0].get("started_at", "")
            end_raw = recordings[0].get("completed_at", "")
            if start_raw and end_raw:
                started = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                ended = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                duration_seconds = max(0.0, (ended - started).total_seconds())
                duration_minutes = int(duration_seconds / 60)
    except Exception:
        pass

    return {
        "meeting_id": bot_id,
        "participants": participants,
        "started_at": details.get("join_at", ""),
        "ended_at": ended_at,
        "duration_minutes": duration_minutes,
        "duration_seconds": duration_seconds,
        "slack_channel": details.get("meeting_url", ""),
    }


async def download_media(media_url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    downloaded = 0
    async with httpx.AsyncClient(timeout=900) as client:
        async with client.stream("GET", media_url) as response:
            response.raise_for_status()
            with open(path, "wb") as output:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
    print(f"Downloaded {downloaded / 1024 / 1024:.1f} MB")
    return path


async def transcribe_batch_mode(media_path: str, mode: str):
    print(f"Running Sarvam batch mode={mode}")
    result = await transcriber.transcribe_audio_batch_mode(media_path, mode)
    print(f"Sarvam mode={mode}: {len(result.entries)} entries, {len(result.text)} chars")
    return result


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def find_reference_text(entry: dict, english_entries: list[dict]) -> str:
    start = float(entry.get("start_time_seconds") or 0.0)
    end = float(entry.get("end_time_seconds") or start)
    best = None
    best_overlap = 0.0
    for reference in english_entries:
        ref_start = float(reference.get("start_time_seconds") or 0.0)
        ref_end = float(reference.get("end_time_seconds") or ref_start)
        overlap = overlap_seconds(start, end, ref_start, ref_end)
        if overlap > best_overlap:
            best = reference
            best_overlap = overlap
    if best and best_overlap > 0:
        return (best.get("transcript") or best.get("text") or "").strip()
    return ""


def validation_chunks(items: list[dict], max_chars: int = 9000):
    chunk = []
    size = 0
    for item in items:
        encoded = json.dumps(item, ensure_ascii=False)
        if chunk and size + len(encoded) > max_chars:
            yield chunk
            chunk = []
            size = 0
        chunk.append(item)
        size += len(encoded)
    if chunk:
        yield chunk


async def validate_hinglish_entries(hinglish_named, english_named, model: str):
    entries = hinglish_named.entries or []
    if not entries:
        return hinglish_named

    items = []
    for index, entry in enumerate(entries):
        raw_text = (entry.get("transcript") or entry.get("text") or "").strip()
        if not raw_text:
            continue
        items.append(
            {
                "id": index,
                "speaker": entry.get("speaker_name") or f"Speaker {entry.get('speaker_id', '?')}",
                "start": entry.get("start_time_seconds"),
                "end": entry.get("end_time_seconds"),
                "raw_hinglish": raw_text,
                "english_reference": find_reference_text(entry, english_named.entries or []),
            }
        )

    system_prompt = (
        "You clean Roman Hinglish ASR transcripts for internal Indian startup meetings. "
        "Correct only obvious speech-recognition, romanization, punctuation, spacing, and repeated-junk errors. "
        "Preserve original meaning, tone, and Hinglish style. Do not translate into formal English. "
        "Keep product/tool/project names exact when clear: ClickUp, Slack, Recall, Sarvam, CIOS, MCP, GPT, API, "
        "frontend, backend, AWS. Use the English reference only to disambiguate unclear words; never add new "
        "content from it. If speech is unclear, keep the closest text and use [unclear] sparingly. "
        "Return JSON only as {\"entries\":[{\"id\":number,\"text\":string}]} with the same ids."
    )
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    corrected_by_id = {}
    chunks = list(validation_chunks(items))
    for chunk_index, chunk in enumerate(chunks, 1):
        print(f"Validating Hinglish chunk {chunk_index}/{len(chunks)} ({len(chunk)} segments)")
        user_prompt = (
            "Clean these Roman Hinglish transcript segments conservatively:\n"
            + json.dumps({"entries": chunk}, ensure_ascii=False)
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=12000,
            timeout=180,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        for item in payload.get("entries", []):
            if (
                isinstance(item, dict)
                and isinstance(item.get("id"), int)
                and isinstance(item.get("text"), str)
            ):
                corrected_by_id[item["id"]] = item["text"].strip()

    corrected_entries = []
    for index, entry in enumerate(entries):
        item = dict(entry)
        corrected = corrected_by_id.get(index)
        if corrected:
            item["transcript"] = corrected
            item["text"] = corrected
        corrected_entries.append(item)

    text = " ".join(
        entry.get("transcript") or entry.get("text") or "" for entry in corrected_entries
    ).strip()
    return replace(
        hinglish_named,
        text=text,
        entries=corrected_entries,
        source="batch/translit + conservative validation",
    )


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Saved {path} ({len(text)} chars)")


async def generate(args):
    load_dotenv()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details = await get_bot_details(args.bot_id)
    media_url = pick_media_url(details)
    participants, speaker_timeline = await fetch_participant_artifacts(details)
    metadata = build_metadata(args.bot_id, details, participants)
    notes = {"meeting_title": args.title}

    media_path = await download_media(media_url)
    try:
        english = await transcribe_batch_mode(media_path, "translate")
        hinglish = await transcribe_batch_mode(media_path, "translit")

        duration_seconds = metadata.get("duration_seconds") or None
        english_named = assign_participant_names(english, speaker_timeline, duration_seconds)
        english_named = replace(english_named, source="batch/translate")

        hinglish_named = assign_participant_names(hinglish, speaker_timeline, duration_seconds)
        hinglish_validated = await validate_hinglish_entries(
            hinglish_named,
            english_named,
            args.validation_model,
        )

        english_text = format_speaker_transcript(english_named, metadata, notes)
        hinglish_text = format_speaker_transcript(hinglish_validated, metadata, notes).replace(
            "Speaker Transcript",
            "Validated Roman-Hinglish Speaker Transcript",
            1,
        )

        english_path = output_dir / f"{args.bot_id}-english-speaker-transcript.txt"
        hinglish_path = output_dir / f"{args.bot_id}-hinglish-validated-speaker-transcript.txt"
        write_text(english_path, english_text)
        write_text(hinglish_path, hinglish_text)
        write_text(
            output_dir / "manifest.json",
            json.dumps(
                {
                    "bot_id": args.bot_id,
                    "meeting_title": args.title,
                    "generated_at_utc": utc_now(),
                    "english_file": str(english_path),
                    "hinglish_file": str(hinglish_path),
                    "participants": participants,
                    "english_entries": len(english_named.entries),
                    "hinglish_entries": len(hinglish_validated.entries),
                    "validation_model": args.validation_model,
                    "validation": (
                        "Conservative pass over Sarvam translit output using the English "
                        "transcript only as ambiguity reference."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
    finally:
        try:
            os.unlink(media_path)
        except OSError:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate English and validated Roman-Hinglish speaker transcripts for a Recall bot."
    )
    parser.add_argument("--bot-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-model", default=DEFAULT_VALIDATION_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(generate(parse_args()))
