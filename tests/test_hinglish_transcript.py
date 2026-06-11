import asyncio
import json
from types import SimpleNamespace

from services.hinglish_transcript import validate_hinglish_transcript
from services.transcriber import SarvamTranscriptResult


class FakeChatCompletions:
    def __init__(self):
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        content = json.dumps({"entries": [{"id": 0, "text": "Haan sir, ye ho gaya hai."}]})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_validate_hinglish_transcript_uses_english_reference_without_speaker_changes():
    fake_completions = FakeChatCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    hinglish = SarvamTranscriptResult(
        text="han sir ye ho gya ha",
        source="batch/translit",
        entries=[
            {
                "speaker_id": "0",
                "speaker_name": "Priyanshu",
                "start_time_seconds": 0.0,
                "end_time_seconds": 2.0,
                "transcript": "han sir ye ho gya ha",
            }
        ],
    )
    english = SarvamTranscriptResult(
        text="Yes sir, this is done.",
        source="batch/translate",
        entries=[
            {
                "speaker_id": "0",
                "speaker_name": "Priyanshu",
                "start_time_seconds": 0.0,
                "end_time_seconds": 2.0,
                "transcript": "Yes sir, this is done.",
            }
        ],
    )

    result = asyncio.run(
        validate_hinglish_transcript(
            hinglish,
            english,
            client=fake_client,
            model="fake-model",
        )
    )

    assert result.entries[0]["speaker_name"] == "Priyanshu"
    assert result.entries[0]["transcript"] == "Haan sir, ye ho gaya hai."
    assert result.source == "batch/translit + conservative validation"
    request = fake_completions.requests[0]
    assert request["model"] == "fake-model"
    assert "Yes sir, this is done." in request["messages"][1]["content"]
