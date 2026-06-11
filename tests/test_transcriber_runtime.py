import types
import asyncio

import services.transcriber as transcriber


def test_ffmpeg_executable_falls_back_to_imageio_binary(monkeypatch):
    fake_imageio = types.SimpleNamespace(get_ffmpeg_exe=lambda: "C:/bin/ffmpeg.exe")
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", fake_imageio)

    assert transcriber._ffmpeg_executable() == "C:/bin/ffmpeg.exe"


def test_transcribe_audio_batch_mode_restores_configured_mode(monkeypatch):
    seen_modes = []

    async def fake_transcribe(media_path, batch_mode=None):
        seen_modes.append((media_path, batch_mode, transcriber.SARVAM_BATCH_MODE))
        return "result"

    monkeypatch.setattr(transcriber, "SARVAM_BATCH_MODE", "translate")
    monkeypatch.setattr(transcriber, "_transcribe_audio_batch", fake_transcribe)

    result = asyncio.run(transcriber.transcribe_audio_batch_mode("meeting.mp4", "translit"))

    assert result == "result"
    assert seen_modes == [("meeting.mp4", "translit", "translate")]
    assert transcriber.SARVAM_BATCH_MODE == "translate"
