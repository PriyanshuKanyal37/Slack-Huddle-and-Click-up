import asyncio

import httpx

import services.clickup_brain as clickup_brain
from services.clickup_brain import (
    assign_participant_names,
    format_speaker_transcript,
)
from services.transcriber import (
    SarvamTranscriptResult,
    classify_sarvam_batch_error,
    combine_sarvam_batch_outputs,
    normalize_sarvam_batch_payload,
    parse_sarvam_output_content,
)


def test_assign_participant_names_uses_overlap_with_speaker_timeline():
    result = SarvamTranscriptResult(
        text="Hello. Yes.",
        source="batch",
        entries=[
            {
                "speaker_id": "0",
                "start_time_seconds": 0.0,
                "end_time_seconds": 2.0,
                "transcript": "Hello.",
            },
            {
                "speaker_id": "1",
                "start_time_seconds": 2.1,
                "end_time_seconds": 5.0,
                "transcript": "Yes.",
            },
        ],
    )
    speaker_timeline = [
        {"participant": {"name": "Priyanshu"}, "timestamp": {"relative": 0.0}},
        {"participant": {"name": "Shivam"}, "timestamp": {"relative": 2.0}},
        {"participant": {"name": "Priyanshu"}, "timestamp": {"relative": 6.0}},
    ]

    mapped = assign_participant_names(result, speaker_timeline, duration_seconds=8)

    assert mapped.entries[0]["speaker_name"] == "Priyanshu"
    assert mapped.entries[1]["speaker_name"] == "Shivam"


def test_assign_participant_names_uses_recall_speaker_timeline_spans():
    result = SarvamTranscriptResult(
        text="Hello. Yes.",
        source="batch",
        entries=[
            {
                "speaker_id": "S1",
                "start_time_seconds": 14.0,
                "end_time_seconds": 20.0,
                "transcript": "Hello.",
            },
            {
                "speaker_id": "S2",
                "start_time_seconds": 50.0,
                "end_time_seconds": 55.0,
                "transcript": "Yes.",
            },
        ],
    )
    speaker_timeline = [
        {
            "participant": {"name": "Firdosh Ahmad"},
            "start_timestamp": {"relative": 13.239789},
            "end_timestamp": {"relative": 45.39799},
        },
        {
            "participant": {"name": "Shivam"},
            "start_timestamp": {"relative": 45.39799},
            "end_timestamp": {"relative": 169.47809},
        },
    ]

    mapped = assign_participant_names(result, speaker_timeline, duration_seconds=180)

    assert mapped.entries[0]["speaker_name"] == "Firdosh Ahmad"
    assert mapped.entries[0]["speaker_match_confidence"] == 1.0
    assert mapped.entries[1]["speaker_name"] == "Shivam"


def test_assign_participant_names_does_not_force_ambiguous_overlap():
    result = SarvamTranscriptResult(
        text="Mixed segment.",
        source="batch",
        entries=[
            {
                "speaker_id": "S1",
                "start_time_seconds": 0.0,
                "end_time_seconds": 10.0,
                "transcript": "Mixed segment.",
            },
        ],
    )
    speaker_timeline = [
        {
            "participant": {"name": "Firdosh Ahmad"},
            "start_timestamp": {"relative": 0.0},
            "end_timestamp": {"relative": 5.0},
        },
        {
            "participant": {"name": "Shivam"},
            "start_timestamp": {"relative": 5.0},
            "end_timestamp": {"relative": 10.0},
        },
    ]

    mapped = assign_participant_names(result, speaker_timeline, duration_seconds=10)

    assert mapped.entries[0]["speaker_name"] == "Speaker S1"
    assert mapped.entries[0]["possible_speaker_name"] in {"Firdosh Ahmad", "Shivam"}
    assert mapped.entries[0]["speaker_match_ambiguous"] is True


def test_assign_participant_names_normalizes_millisecond_timeline_values():
    result = SarvamTranscriptResult(
        text="Hello.",
        source="batch",
        entries=[
            {
                "speaker_id": "0",
                "start_time_seconds": 2.1,
                "end_time_seconds": 4.0,
                "transcript": "Hello.",
            },
        ],
    )
    speaker_timeline = [
        {"participant": {"name": "Priyanshu"}, "timestamp": {"relative": 0}},
        {"participant": {"name": "Shivam"}, "timestamp": {"relative": 2000}},
    ]

    mapped = assign_participant_names(result, speaker_timeline, duration_seconds=8)

    assert mapped.entries[0]["speaker_name"] == "Shivam"


def test_assign_participant_names_normalizes_long_meeting_millisecond_timeline_values():
    result = SarvamTranscriptResult(
        text="Hello.",
        source="batch",
        entries=[
            {
                "speaker_id": "0",
                "start_time_seconds": 2.1,
                "end_time_seconds": 4.0,
                "transcript": "Hello.",
            },
        ],
    )
    speaker_timeline = [
        {"participant": {"name": "Priyanshu"}, "timestamp": {"relative": 0}},
        {"participant": {"name": "Shivam"}, "timestamp": {"relative": 2000}},
        {"participant": {"name": "Priyanshu"}, "timestamp": {"relative": 2_880_000}},
    ]

    mapped = assign_participant_names(result, speaker_timeline, duration_seconds=2880)

    assert mapped.entries[0]["speaker_name"] == "Shivam"


def test_format_speaker_transcript_includes_metadata_and_timestamps():
    result = SarvamTranscriptResult(
        text="Hello. Yes.",
        source="batch",
        entries=[
            {
                "speaker_id": "0",
                "speaker_name": "Priyanshu",
                "start_time_seconds": 0.25,
                "end_time_seconds": 2.5,
                "transcript": "Hello.",
            },
            {
                "speaker_id": "1",
                "speaker_name": "Shivam",
                "start_time_seconds": 62.0,
                "end_time_seconds": 65.3,
                "transcript": "Yes.",
            },
        ],
    )

    text = format_speaker_transcript(
        result,
        {
            "meeting_id": "bot-1",
            "started_at": "2026-05-14T04:48:59Z",
            "duration_minutes": 48,
            "participants": ["Priyanshu", "Shivam"],
        },
        {"meeting_title": "ClickUp Brain Planning"},
    )

    assert "Meeting: ClickUp Brain Planning" in text
    assert "Participants: Priyanshu, Shivam" in text
    assert "[00:00.250 - 00:02.500] Priyanshu:" in text
    assert "[01:02.000 - 01:05.300] Shivam:" in text


def test_format_speaker_transcript_surfaces_ambiguous_speaker_matches():
    result = SarvamTranscriptResult(
        text="Mixed segment.",
        source="batch",
        entries=[
            {
                "speaker_id": "S1",
                "speaker_name": "Speaker S1",
                "possible_speaker_name": "Shivam",
                "speaker_match_confidence": 0.5,
                "speaker_match_ambiguous": True,
                "start_time_seconds": 0.0,
                "end_time_seconds": 10.0,
                "transcript": "Mixed segment.",
            },
        ],
    )

    text = format_speaker_transcript(
        result,
        {
            "meeting_id": "bot-1",
            "started_at": "2026-05-14T04:48:59Z",
            "duration_minutes": 1,
            "participants": ["Priyanshu", "Shivam"],
        },
        {"meeting_title": "Ambiguous"},
    )

    assert "[00:00.000 - 00:10.000] Speaker S1 (possible Shivam, confidence 0.5):" in text


def test_normalize_sarvam_batch_payload_reads_diarized_segments():
    payload = {
        "transcript": "Hello. Yes.",
        "diarized_transcript": [
            {
                "speaker_id": "S1",
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.75,
                "transcript": "Hello.",
            },
            {
                "speaker_id": "S2",
                "start_time_seconds": 1.75,
                "end_time_seconds": 4.0,
                "transcript": "Yes.",
            },
        ],
    }

    result = normalize_sarvam_batch_payload(payload, offset_seconds=10.0)

    assert result.text == "Hello. Yes."
    assert result.entries == [
        {
            "speaker_id": "S1",
            "start_time_seconds": 10.0,
            "end_time_seconds": 11.75,
            "transcript": "Hello.",
        },
        {
            "speaker_id": "S2",
            "start_time_seconds": 11.75,
            "end_time_seconds": 14.0,
            "transcript": "Yes.",
        },
    ]


def test_parse_sarvam_output_content_accepts_plain_text_outputs():
    assert parse_sarvam_output_content("Plain transcript text.", "result.txt") == {
        "transcript": "Plain transcript text."
    }


def test_combine_sarvam_batch_outputs_uses_input_file_offsets():
    outputs = [
        (
            "part_0000.mp3",
            {
                "transcript": "First.",
                "diarized_transcript": {
                    "entries": [
                        {
                            "speaker_id": "S1",
                            "start_time_seconds": 1.0,
                            "end_time_seconds": 2.0,
                            "transcript": "First.",
                        }
                    ]
                },
            },
        ),
        (
            "part_0001.mp3",
            {
                "transcript": "Second.",
                "diarized_transcript": {
                    "entries": [
                        {
                            "speaker_id": "S2",
                            "start_time_seconds": 1.0,
                            "end_time_seconds": 2.0,
                            "transcript": "Second.",
                        }
                    ]
                },
            },
        ),
    ]

    result = combine_sarvam_batch_outputs(
        outputs,
        {"part_0000.mp3": 0.0, "part_0001.mp3": 3300.0},
    )

    assert result.text == "First. Second."
    assert result.entries[1]["start_time_seconds"] == 3301.0


def test_classify_sarvam_batch_error_distinguishes_quota_and_transient_errors():
    quota_response = httpx.Response(
        429,
        json={"error": {"code": "insufficient_quota_error"}},
        request=httpx.Request("GET", "https://api.sarvam.ai"),
    )
    rate_response = httpx.Response(
        429,
        json={"error": {"code": "rate_limit_exceeded_error"}},
        request=httpx.Request("GET", "https://api.sarvam.ai"),
    )
    overload_response = httpx.Response(
        503,
        json={"error": {"code": "rate_limit_exceeded_error"}},
        request=httpx.Request("GET", "https://api.sarvam.ai"),
    )

    assert classify_sarvam_batch_error(httpx.HTTPStatusError("quota", request=quota_response.request, response=quota_response)) == "credit_exhausted"
    assert classify_sarvam_batch_error(httpx.HTTPStatusError("rate", request=rate_response.request, response=rate_response)) == "retryable"
    assert classify_sarvam_batch_error(httpx.HTTPStatusError("overload", request=overload_response.request, response=overload_response)) == "retryable"
    assert classify_sarvam_batch_error(TimeoutError("timed out")) == "retryable"


def test_clickup_brain_upload_uses_initial_comment_without_separate_message(monkeypatch):
    calls = []

    async def fake_upload(channel_id, filename, title, content, initial_comment="", thread_ts=""):
        calls.append(
            {
                "channel_id": channel_id,
                "filename": filename,
                "content": content,
                "initial_comment": initial_comment,
                "thread_ts": thread_ts,
            }
        )

    async def fail_if_message_posted(method, payload):
        raise AssertionError(f"unexpected Slack method: {method}")

    monkeypatch.setattr(clickup_brain, "CLICKUP_BRAIN_CHANNEL_ID", "C123")
    monkeypatch.setattr(clickup_brain, "upload_text_file_to_slack", fake_upload)
    monkeypatch.setattr(clickup_brain, "_slack_api_post", fail_if_message_posted)

    transcript = SarvamTranscriptResult(text="Hello.", source="batch", entries=[])
    asyncio.run(
        clickup_brain.send_clickup_brain_channel_post(
            {"meeting_title": "Planning"},
            {"meeting_id": "bot-1", "participants": [], "duration_minutes": 1},
            transcript,
            "Hello.",
        )
    )

    assert len(calls) == 1
    assert calls[0]["initial_comment"].startswith("*New Huddle Transcript:* Planning")
    assert calls[0]["thread_ts"] == ""


def test_channel_summary_formats_action_points_without_buttons():
    transcript = SarvamTranscriptResult(
        text="Hello.",
        source="batch",
        entries=[
            {
                "speaker_id": "S1",
                "speaker_name": "Priyanshu",
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.0,
                "transcript": "Hello.",
            }
        ],
    )

    summary = clickup_brain.build_channel_summary(
        {
            "meeting_title": "Planning",
            "overview": "Discussed deployment and QA.",
            "key_takeaways": ["Deployment is ready."],
            "next_steps": [
                {
                    "task": "Finish QA report.",
                    "owner": "Priyanshu",
                    "deadline": "Today",
                    "clickup_task_name": "QA hardening",
                }
            ],
        },
        {"participants": ["Priyanshu"], "duration_minutes": 5},
        transcript,
    )

    assert "*Overview:*" in summary
    assert "Discussed deployment and QA." in summary
    assert "*Action Points:*" in summary
    assert "*Next steps:*" not in summary
    assert "Finish QA report." in summary
    assert "_Owner: Priyanshu_" in summary
    assert "_Deadline: Today_" in summary
    assert "_Suggested ClickUp task: QA hardening_" in summary
    assert "Confirm" not in summary
    assert "Change Task" not in summary
    assert "Create New Task" not in summary


def test_upload_text_file_uses_slack_documented_filename_field(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload=None):
            self.payload = payload or {}
            self.status_code = 200
            self.text = "OK - 5"

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            calls.append({"url": url, **kwargs})
            if url.endswith("files.getUploadURLExternal"):
                return FakeResponse({"ok": True, "upload_url": "https://files.slack.com/upload/v1/test", "file_id": "F1"})
            return FakeResponse()

    async def fake_slack_api_post(method, payload):
        calls.append({"method": method, "payload": payload})
        return {"ok": True}

    monkeypatch.setattr(clickup_brain.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(clickup_brain, "_slack_api_post", fake_slack_api_post)
    monkeypatch.setattr(clickup_brain, "SLACK_BOT_TOKEN", "xoxb-test")

    asyncio.run(
        clickup_brain.upload_text_file_to_slack(
            "C1",
            "meeting.txt",
            "meeting.txt",
            "hello",
            initial_comment="summary",
        )
    )

    upload_call = calls[1]
    assert upload_call["url"] == "https://files.slack.com/upload/v1/test"
    assert "filename" in upload_call["files"]
    assert "file" not in upload_call["files"]
