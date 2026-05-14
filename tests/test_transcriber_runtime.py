import types

import services.transcriber as transcriber


def test_ffmpeg_executable_falls_back_to_imageio_binary(monkeypatch):
    fake_imageio = types.SimpleNamespace(get_ffmpeg_exe=lambda: "C:/bin/ffmpeg.exe")
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", fake_imageio)

    assert transcriber._ffmpeg_executable() == "C:/bin/ffmpeg.exe"
