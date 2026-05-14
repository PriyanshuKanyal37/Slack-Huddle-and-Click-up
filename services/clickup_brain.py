import json
import os
from datetime import datetime, timedelta, timezone

import httpx

from services.transcriber import SarvamTranscriptResult


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
CLICKUP_BRAIN_CHANNEL_ID = os.getenv("CLICKUP_BRAIN_CHANNEL_ID") or os.getenv("SLACK_CLICKUP_BRAIN_CHANNEL_ID")
SLACK_MESSAGE_CHUNK_SIZE = 30_000
SPEAKER_MATCH_MIN_RATIO = 0.5
SPEAKER_MATCH_MIN_WIN_RATIO = 1.5
SPEAKER_TIMELINE_MERGE_GAP_SECONDS = 0.3


def _first_present(payload: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return payload.get(key)
    return None


def _timestamp_seconds(value, duration_seconds: float | None = None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("relative", "seconds", "time", "timestamp"):
            if key in value:
                return _timestamp_seconds(value.get(key), duration_seconds)
    if isinstance(value, str):
        try:
            return _timestamp_seconds(float(value), duration_seconds)
        except ValueError:
            return None
    return None


def _participant_name(event: dict) -> str:
    participant = event.get("participant") or event.get("user") or {}
    if isinstance(participant, dict):
        return (
            participant.get("name")
            or participant.get("display_name")
            or participant.get("real_name")
            or participant.get("id")
            or ""
        )
    if isinstance(participant, str):
        return participant
    return event.get("participant_name") or event.get("name") or event.get("speaker_name") or ""


def _timeline_segments(speaker_timeline: list[dict], duration_seconds: float | None = None) -> list[dict]:
    explicit_segments = []
    events = []
    raw_times = []
    for event in speaker_timeline or []:
        if not isinstance(event, dict):
            continue
        start_timestamp = _timestamp_seconds(
            _first_present(event, ("start_timestamp", "start_time", "start")),
            duration_seconds,
        )
        end_timestamp = _timestamp_seconds(
            _first_present(event, ("end_timestamp", "end_time", "end")),
            duration_seconds,
        )
        name = _participant_name(event)
        if name and start_timestamp is not None and end_timestamp is not None and end_timestamp > start_timestamp:
            raw_times.extend([start_timestamp, end_timestamp])
            explicit_segments.append(
                {
                    "start": max(0.0, start_timestamp),
                    "end": end_timestamp,
                    "name": name,
                }
            )
            continue

        timestamp = (
            _timestamp_seconds(event.get("timestamp"), duration_seconds)
            if "timestamp" in event
            else _timestamp_seconds(event.get("start_time_seconds"), duration_seconds)
        )
        if timestamp is None:
            timestamp = _timestamp_seconds(event.get("time"), duration_seconds)
        if name:
            raw_times.append(timestamp or 0.0)
            events.append({"time": max(0.0, timestamp or 0.0), "name": name})

    if explicit_segments:
        if _timeline_values_look_like_milliseconds(raw_times, duration_seconds):
            explicit_segments = [
                {
                    **segment,
                    "start": segment["start"] / 1000.0,
                    "end": segment["end"] / 1000.0,
                }
                for segment in explicit_segments
            ]
        return _merge_adjacent_timeline_segments(sorted(explicit_segments, key=lambda item: item["start"]))

    if _timeline_values_look_like_milliseconds(raw_times, duration_seconds):
        events = [{**event, "time": event["time"] / 1000.0} for event in events]

    events.sort(key=lambda item: item["time"])
    segments = []
    for index, event in enumerate(events):
        end = events[index + 1]["time"] if index + 1 < len(events) else duration_seconds
        if end is None:
            end = event["time"] + 30.0
        if end <= event["time"]:
            continue
        segments.append({"start": event["time"], "end": end, "name": event["name"]})
    return segments


def _timeline_values_look_like_milliseconds(
    values: list[float],
    duration_seconds: float | None = None,
) -> bool:
    numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric_values:
        return False
    largest = max(numeric_values)
    if duration_seconds and duration_seconds > 0:
        return largest > duration_seconds * 1.25 and largest / 1000.0 <= duration_seconds * 1.25
    return largest >= 1000.0


def _merge_adjacent_timeline_segments(segments: list[dict]) -> list[dict]:
    merged = []
    for segment in segments:
        if not merged:
            merged.append(dict(segment))
            continue
        previous = merged[-1]
        gap = segment["start"] - previous["end"]
        if segment["name"] == previous["name"] and gap <= SPEAKER_TIMELINE_MERGE_GAP_SECONDS:
            previous["end"] = max(previous["end"], segment["end"])
            continue
        merged.append(dict(segment))
    return merged


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_participant_names(
    transcript: SarvamTranscriptResult,
    speaker_timeline: list[dict],
    duration_seconds: float | None = None,
) -> SarvamTranscriptResult:
    segments = _timeline_segments(speaker_timeline, duration_seconds)
    mapped_entries = []

    for entry in transcript.entries:
        item = dict(entry)
        start = float(item.get("start_time_seconds") or 0.0)
        end = float(item.get("end_time_seconds") or start)
        duration = max(0.001, end - start)
        overlap_by_name: dict[str, float] = {}
        for segment in segments:
            overlap = _overlap_seconds(start, end, segment["start"], segment["end"])
            if overlap > 0:
                overlap_by_name[segment["name"]] = overlap_by_name.get(segment["name"], 0.0) + overlap

        if overlap_by_name:
            ranked_overlaps = sorted(overlap_by_name.items(), key=lambda pair: pair[1], reverse=True)
            speaker_name, overlap = ranked_overlaps[0]
            runner_up_overlap = ranked_overlaps[1][1] if len(ranked_overlaps) > 1 else 0.0
            confidence = min(1.0, overlap / duration)
            clear_winner = runner_up_overlap == 0.0 or overlap >= runner_up_overlap * SPEAKER_MATCH_MIN_WIN_RATIO
            if confidence >= SPEAKER_MATCH_MIN_RATIO and clear_winner:
                item["speaker_name"] = speaker_name
                item["speaker_match_ambiguous"] = False
            else:
                item["speaker_name"] = item.get("speaker_name") or f"Speaker {item.get('speaker_id', '?')}"
                item["possible_speaker_name"] = speaker_name
                item["speaker_match_ambiguous"] = True
            item["speaker_match_confidence"] = round(confidence, 3)
            item["speaker_overlap_seconds"] = round(overlap, 3)
        else:
            item["speaker_name"] = item.get("speaker_name") or f"Speaker {item.get('speaker_id', '?')}"
            item["speaker_match_confidence"] = 0.0
            item["speaker_overlap_seconds"] = 0.0
            item["speaker_match_ambiguous"] = False
        mapped_entries.append(item)

    return SarvamTranscriptResult(
        text=transcript.text,
        source=transcript.source,
        entries=mapped_entries,
        timestamps=transcript.timestamps,
        raw=transcript.raw,
    )


def _format_offset(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000))
    minutes, ms = divmod(total_ms, 60_000)
    secs, ms = divmod(ms, 1000)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _format_started_at(value: str) -> str:
    if not value:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        ist = dt.astimezone(timezone.utc) + timedelta(hours=5, minutes=30)
        return ist.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return value


def _participant_list(metadata: dict) -> str:
    names = []
    for participant in metadata.get("participants", []) or []:
        if isinstance(participant, dict):
            name = participant.get("name") or participant.get("display_name") or ""
        else:
            name = str(participant)
        if name:
            names.append(name)
    return ", ".join(names) or "Unknown"


def format_speaker_transcript(
    transcript: SarvamTranscriptResult,
    metadata: dict,
    notes: dict | None = None,
) -> str:
    notes = notes or {}
    title = notes.get("meeting_title") or "Untitled Meeting"
    lines = [
        f"Meeting: {title}",
        f"Meeting ID: {metadata.get('meeting_id', 'Unknown')}",
        f"Started: {_format_started_at(metadata.get('started_at', ''))}",
        f"Duration: {metadata.get('duration_minutes', 0)} min",
        f"Participants: {_participant_list(metadata)}",
        f"Transcript source: Sarvam AI {transcript.source}",
        "",
        "Speaker Transcript",
        "==================",
        "",
    ]

    entries = transcript.entries or []
    if not entries and transcript.text:
        lines.extend(["[00:00.000] Speaker unknown:", transcript.text.strip(), ""])
        return "\n".join(lines).strip() + "\n"

    for entry in entries:
        text = (entry.get("transcript") or entry.get("text") or "").strip()
        if not text:
            continue
        start = _format_offset(float(entry.get("start_time_seconds") or 0.0))
        end = _format_offset(float(entry.get("end_time_seconds") or 0.0))
        speaker = entry.get("speaker_name") or f"Speaker {entry.get('speaker_id', '?')}"
        if entry.get("speaker_match_ambiguous") and entry.get("possible_speaker_name"):
            confidence = entry.get("speaker_match_confidence", 0)
            speaker = f"{speaker} (possible {entry['possible_speaker_name']}, confidence {confidence})"
        lines.append(f"[{start} - {end}] {speaker}:")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_channel_summary(notes: dict, metadata: dict, transcript: SarvamTranscriptResult) -> str:
    title = notes.get("meeting_title") or "Untitled Meeting"
    takeaways = notes.get("key_takeaways") or []
    next_steps = notes.get("next_steps") or []

    lines = [
        f"*New Huddle Transcript:* {title}",
        f"*When:* {_format_started_at(metadata.get('started_at', ''))}",
        f"*Duration:* {metadata.get('duration_minutes', 0)} min",
        f"*Participants:* {_participant_list(metadata)}",
        f"*Transcript:* Sarvam {transcript.source}, {len(transcript.entries)} speaker segments",
    ]

    if notes.get("overview"):
        lines.extend(["", "*Overview:*", notes["overview"]])

    if takeaways:
        lines.append("")
        lines.append("*Key takeaways:*")
        lines.extend(f"- {item}" for item in takeaways[:8])

    if next_steps:
        lines.append("")
        lines.append("*Action Points:*")
        for item in next_steps[:8]:
            if isinstance(item, dict):
                task = item.get("task", "")
                owner = item.get("owner", "")
                deadline = item.get("deadline", "")
                clickup_task = item.get("clickup_task_name", "")
            else:
                task = str(item)
                owner = ""
                deadline = ""
                clickup_task = ""
            if task:
                lines.append(f"- {task}")
                if owner:
                    lines.append(f"  _Owner: {owner}_")
                if deadline:
                    lines.append(f"  _Deadline: {deadline}_")
                if clickup_task:
                    lines.append(f"  _Suggested ClickUp task: {clickup_task}_")

    lines.append("")
    if transcript.entries:
        lines.append("Full speaker-labeled transcript is attached as a .txt file.")
    else:
        lines.append("Full transcript is attached as a .txt file. Speaker labels were not available from the transcription result.")
    return "\n".join(lines)


def _chunk_text(text: str, chunk_size: int = SLACK_MESSAGE_CHUNK_SIZE) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = chunk_size
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return [chunk for chunk in chunks if chunk]


async def _slack_api_post(method: str, payload: dict) -> dict:
    if not SLACK_BOT_TOKEN:
        raise RuntimeError("SLACK_BOT_TOKEN is not configured")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"https://slack.com/api/{method}",
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {data.get('error')}")
    return data


async def upload_text_file_to_slack(
    channel_id: str,
    filename: str,
    title: str,
    content: str,
    initial_comment: str = "",
    thread_ts: str = "",
):
    encoded = content.encode("utf-8")
    if not SLACK_BOT_TOKEN:
        raise RuntimeError("SLACK_BOT_TOKEN is not configured")
    async with httpx.AsyncClient(timeout=30) as client:
        upload_resp = await client.post(
            "https://slack.com/api/files.getUploadURLExternal",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            data={"filename": filename, "length": str(len(encoded))},
        )
    upload_resp.raise_for_status()
    upload = upload_resp.json()
    if not upload.get("ok"):
        raise RuntimeError(f"Slack files.getUploadURLExternal failed: {upload.get('error')} {upload.get('response_metadata', {})}")
    upload_url = upload["upload_url"]
    file_id = upload["file_id"]

    async with httpx.AsyncClient(timeout=120) as client:
        upload_resp = await client.post(
            upload_url,
            files={"filename": (filename, encoded, "text/plain; charset=utf-8")},
        )
    upload_resp.raise_for_status()

    complete_payload = {
        "files": [{"id": file_id, "title": title}],
        "channel_id": channel_id,
    }
    if initial_comment:
        complete_payload["initial_comment"] = initial_comment
    if thread_ts:
        complete_payload["thread_ts"] = thread_ts
    await _slack_api_post("files.completeUploadExternal", complete_payload)


async def send_clickup_brain_channel_post(
    notes: dict,
    metadata: dict,
    transcript: SarvamTranscriptResult,
    transcript_text: str,
):
    if not CLICKUP_BRAIN_CHANNEL_ID:
        print("[ClickUp Brain] Channel env var not configured; skipping channel post.")
        return

    safe_meeting_id = str(metadata.get("meeting_id", "meeting")).replace("/", "-")
    filename = f"{safe_meeting_id}-speaker-transcript.txt"
    summary = build_channel_summary(notes, metadata, transcript)
    try:
        await upload_text_file_to_slack(
            CLICKUP_BRAIN_CHANNEL_ID,
            filename,
            filename,
            transcript_text,
            initial_comment=summary,
        )
        return
    except Exception as exc:
        print(f"[ClickUp Brain] File upload failed; posting transcript in thread: {exc}")

    fallback_summary = (
        f"{summary}\n\n"
        "_Slack file upload failed, so the full transcript is posted in this thread._"
    )
    message = await _slack_api_post(
        "chat.postMessage",
        {
            "channel": CLICKUP_BRAIN_CHANNEL_ID,
            "text": fallback_summary,
            "unfurl_links": False,
            "unfurl_media": False,
        },
    )
    thread_ts = message.get("ts", "")
    chunks = _chunk_text(transcript_text)
    for index, chunk in enumerate(chunks, 1):
        await _slack_api_post(
            "chat.postMessage",
            {
                "channel": CLICKUP_BRAIN_CHANNEL_ID,
                "thread_ts": thread_ts,
                "text": f"*Transcript part {index}/{len(chunks)}*\n```{chunk}```",
                "unfurl_links": False,
                "unfurl_media": False,
            },
        )


async def send_raw_json_debug(channel_id: str, filename: str, payload: dict | list):
    await upload_text_file_to_slack(channel_id, filename, filename, json.dumps(payload, indent=2, ensure_ascii=False))
