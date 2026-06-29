import json
import os
from dataclasses import replace

from openai import AsyncOpenAI

from services.clickup_brain import assign_participant_names
from services.transcriber import SarvamTranscriptResult, transcribe_audio_batch_mode

openai_key = os.getenv("OPENAI_API_KEY")
openai_client_default = AsyncOpenAI(api_key=openai_key) if openai_key else None

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = AsyncOpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com"
) if deepseek_key else None

HINGLISH_VALIDATION_MODEL = os.getenv("HINGLISH_VALIDATION_MODEL", "gpt-5.1")
HINGLISH_VALIDATION_CHUNK_CHARS = int(os.getenv("HINGLISH_VALIDATION_CHUNK_CHARS", "9000"))


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _reference_text_for_entry(entry: dict, english_entries: list[dict]) -> str:
    start = float(entry.get("start_time_seconds") or 0.0)
    end = float(entry.get("end_time_seconds") or start)
    best_entry = None
    best_overlap = 0.0

    for reference in english_entries:
        ref_start = float(reference.get("start_time_seconds") or 0.0)
        ref_end = float(reference.get("end_time_seconds") or ref_start)
        overlap = _overlap_seconds(start, end, ref_start, ref_end)
        if overlap > best_overlap:
            best_entry = reference
            best_overlap = overlap

    if not best_entry or best_overlap <= 0:
        return ""
    return (best_entry.get("transcript") or best_entry.get("text") or "").strip()


def _validation_chunks(items: list[dict], max_chars: int = HINGLISH_VALIDATION_CHUNK_CHARS):
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


async def validate_hinglish_transcript(
    hinglish_transcript: SarvamTranscriptResult,
    english_transcript: SarvamTranscriptResult,
    client=None,
    model: str = HINGLISH_VALIDATION_MODEL,
) -> SarvamTranscriptResult:
    entries = hinglish_transcript.entries or []
    if not entries:
        return replace(
            hinglish_transcript,
            source="batch/translit + conservative validation",
        )

    items = []
    english_entries = english_transcript.entries or []
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
                "english_reference": _reference_text_for_entry(entry, english_entries),
            }
        )

    if not items:
        return replace(
            hinglish_transcript,
            source="batch/translit + conservative validation",
        )

    openai_client = client or openai_client_default
    corrected_by_id = {}
    system_prompt = (
        "You clean Roman Hinglish ASR transcripts for internal Indian startup meetings. "
        "Correct only obvious speech-recognition, romanization, punctuation, spacing, and repeated-junk errors. "
        "Preserve original meaning, tone, and Hinglish style. Do not translate into formal English. "
        "Keep product/tool/project names exact when clear: ClickUp, Slack, Recall, Sarvam, CIOS, MCP, GPT, API, "
        "frontend, backend, AWS. Use the English reference only to disambiguate unclear words; never add new "
        "content from it. If speech is unclear, keep the closest text and use [unclear] sparingly. "
        "Return JSON only as {\"entries\":[{\"id\":number,\"text\":string}]} with the same ids."
    )

    chunks = list(_validation_chunks(items))
    for chunk_index, chunk in enumerate(chunks, 1):
        print(f"[Hinglish] Validating chunk {chunk_index}/{len(chunks)} ({len(chunk)} segments)")
        user_prompt = (
            "Clean these Roman Hinglish transcript segments conservatively:\n"
            + json.dumps({"entries": chunk}, ensure_ascii=False)
        )
        response = None
        if openai_client and openai_key:
            try:
                response = await openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=12000,
                    timeout=180,
                )
            except Exception as oe:
                print(f"[Hinglish] OpenAI validation failed: {oe}. Trying DeepSeek fallback...")

        if not response:
            if not deepseek_client:
                raise Exception("Neither OpenAI nor DeepSeek client is configured/working.")
            response = await deepseek_client.chat.completions.create(
                model="deepseek-v4-pro",
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
        hinglish_transcript,
        text=text,
        entries=corrected_entries,
        source="batch/translit + conservative validation",
    )


async def build_validated_hinglish_transcript(
    media_path: str,
    english_named_transcript: SarvamTranscriptResult,
    speaker_timeline: list[dict],
    duration_seconds: float | None = None,
) -> SarvamTranscriptResult:
    hinglish_transcript = await transcribe_audio_batch_mode(media_path, "translit")
    named_hinglish = assign_participant_names(
        hinglish_transcript,
        speaker_timeline,
        duration_seconds=duration_seconds,
    )
    return await validate_hinglish_transcript(named_hinglish, english_named_transcript)
