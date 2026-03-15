import asyncio
import httpx
import os
import subprocess
import tempfile
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
MAX_RETRIES = 5      # max retries on 429 rate limit


def _extract_chunks_sync(media_path: str) -> list[str]:
    """
    Blocking function — runs in thread executor so it doesn't freeze the event loop.
    Converts media file to MP3 chunks of CHUNK_SECONDS each.
    Works for both MP4 (video) and MP3 (audio) input.
    """
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


async def transcribe_audio(media_path: str) -> str:
    """
    Accepts a file path to MP4 or MP3.
    Splits into 25s chunks in a thread executor (non-blocking).
    Sends each chunk to Sarvam AI with rate limit handling.
    Returns full English transcript.
    """
    loop = asyncio.get_event_loop()

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
                headers = {"api-subscription-key": SARVAM_KEYS[_sarvam_key_index]}
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        SARVAM_URL, headers=headers, files=files, data=data
                    )

                # Credits exhausted — switch to next key and retry immediately
                if response.status_code == 402:
                    if _sarvam_key_index + 1 < len(SARVAM_KEYS):
                        _sarvam_key_index += 1
                        print(f"[Sarvam] Key exhausted. Switching to key {_sarvam_key_index + 1}/{len(SARVAM_KEYS)}...")
                        continue
                    else:
                        raise Exception("[Sarvam] All API keys exhausted. Add more credits.")

                if response.status_code == 429:
                    wait = int(response.headers.get("Retry-After", 2 ** attempt))
                    print(f"[Sarvam] Rate limited on chunk {i+1}. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...")
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

    return " ".join(transcripts)
