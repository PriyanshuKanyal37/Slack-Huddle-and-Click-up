"""
Webhook server — Recall.ai fires this when a Slack Huddle recording is done.
No polling needed. Recall.ai calls us, we process immediately.

Run locally (testing):
    uvicorn main:app --reload --port 8000

Run in production (Railway/Render):
    uvicorn main:app --host 0.0.0.0 --port $PORT

Recall.ai webhook URL to register:
    https://your-domain.com/webhook/recall
"""

import asyncio
import hashlib
import hmac
import httpx
import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from dotenv import load_dotenv
from upstash_redis.asyncio import Redis
from services.transcriber import transcribe_audio
from services.summarizer import structure_notes
from services.clickup import create_meeting_task

load_dotenv()

RECALL_API_KEY        = os.getenv("RECALL_API_KEY")
RECALL_BASE_URL       = "https://ap-northeast-1.recall.ai/api/v1"
RECALL_MAX_RETRIES    = 5
RECALL_WEBHOOK_SECRET = os.getenv("RECALL_WEBHOOK_SECRET", "")
SLACK_SIGNING_SECRET  = os.getenv("SLACK_SIGNING_SECRET", "")
BOT_KEY_TTL           = 14 * 24 * 60 * 60   # 14 days in seconds

redis = Redis(
    url=os.getenv("UPSTASH_REDIS_URL"),
    token=os.getenv("UPSTASH_REDIS_TOKEN")
)

app = FastAPI()

POLL_INTERVAL_SECONDS   = 120   # poll every 2 minute as backup
in_progress: set        = set()  # bots currently being processed
failed_bots: set        = set()  # bots that failed — skip until server restarts
active_huddles: set     = set()  # channel IDs where a Recall bot was already sent


# ── STARTUP: run poller in background as fallback ─────────────────────────────

@app.on_event("startup")
async def start_poller():
    asyncio.create_task(poll_loop())


async def poll_loop():
    """Background poller — catches any bots the webhook missed."""
    await asyncio.sleep(30)  # wait 30s for server to fully start
    while True:
        try:
            await poll_once()
        except Exception as e:
            print(f"[Poller] Error: {e}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def poll_once():
    headers = {
        "Authorization": f"Token {RECALL_API_KEY}",
        "Content-Type": "application/json"
    }
    since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    for attempt in range(1, RECALL_MAX_RETRIES + 1):
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{RECALL_BASE_URL}/bot/",
                headers=headers,
                params={"status_filter": "done", "created_at_after": since}
            )
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 2 ** attempt))
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        break
    else:
        return

    all_bots = response.json().get("results", [])
    new_bots = [b for b in all_bots if not await is_processed(b["id"])]

    if new_bots:
        print(f"[Poller] Found {len(new_bots)} unprocessed bot(s). Processing...")
    for bot in new_bots:
        await run_pipeline(bot["id"])


# ── DEDUPLICATION (Redis) ─────────────────────────────────────────────────────
# Each bot ID stored as key "bot:<id>" with 14-day auto-expiry.
# No manual pruning needed — Redis handles it automatically.

async def is_processed(bot_id: str) -> bool:
    try:
        return await redis.exists(f"bot:{bot_id}") == 1
    except Exception as e:
        print(f"[Redis] is_processed check failed: {e} — assuming not processed")
        return False


async def mark_processed(bot_id: str):
    try:
        await redis.set(f"bot:{bot_id}", "1", ex=BOT_KEY_TTL)
    except Exception as e:
        print(f"[Redis] mark_processed failed for {bot_id}: {e}")


# ── WEBHOOK SIGNATURE VERIFICATION ───────────────────────────────────────────

def verify_signature(body: bytes, signature_header: str) -> bool:
    """Verify Recall.ai HMAC-SHA256 webhook signature. Skip if no secret set."""
    if not RECALL_WEBHOOK_SECRET:
        return True
    expected = hmac.new(
        RECALL_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


# ── SLACK SIGNATURE VERIFICATION ─────────────────────────────────────────────

def verify_slack_signature(body: bytes, headers) -> bool:
    """Verify Slack HMAC-SHA256 signing secret. Skip if no secret set."""
    if not SLACK_SIGNING_SECRET:
        return True
    timestamp = headers.get("X-Slack-Request-Timestamp", "")
    slack_sig = headers.get("X-Slack-Signature", "")
    if not timestamp or not slack_sig:
        return False
    # Reject replayed requests older than 5 minutes
    if abs(time.time() - int(timestamp)) > 300:
        return False
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, slack_sig)


# ── RECALL.AI AUTO-JOIN ───────────────────────────────────────────────────────

async def send_recall_bot_to_huddle(huddle_url: str, channel_id: str):
    """Sends a Recall.ai bot to join a Slack Huddle URL."""
    headers = {
        "Authorization": f"Token {RECALL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "meeting_url": huddle_url,
        "bot_name": "Meeting Notes"
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{RECALL_BASE_URL}/bot/",
                json=payload,
                headers=headers
            )
        response.raise_for_status()
        bot_id = response.json().get("id")
        print(f"[AutoJoin] Recall bot sent to huddle in channel {channel_id}. Bot ID: {bot_id}")
    except Exception as e:
        # Remove from active set so next huddle in this channel can retry
        active_huddles.discard(channel_id)
        print(f"[AutoJoin] Failed to send bot to {huddle_url}: {e}")


# ── RECALL.AI API ─────────────────────────────────────────────────────────────

async def get_bot_details(bot_id: str) -> dict:
    headers = {
        "Authorization": f"Token {RECALL_API_KEY}",
        "Content-Type": "application/json"
    }
    for attempt in range(1, RECALL_MAX_RETRIES + 1):
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{RECALL_BASE_URL}/bot/{bot_id}/",
                headers=headers
            )
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 2 ** attempt))
            print(f"[Recall] Rate limited. Waiting {wait}s...")
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        break
    else:
        raise Exception(f"[Recall] get_bot_details failed after {RECALL_MAX_RETRIES} retries.")
    return response.json()


# ── PIPELINE ─────────────────────────────────────────────────────────────────

async def run_pipeline(bot_id: str):
    # Prevent duplicate processing (webhook + poller running at same time)
    if await is_processed(bot_id) or bot_id in in_progress or bot_id in failed_bots:
        print(f"[Pipeline] Bot {bot_id} already processed/in-progress/failed. Skipping.")
        return

    in_progress.add(bot_id)
    print(f"\n[Pipeline] Processing bot: {bot_id}")

    try:
        # Fetch full bot details from Recall.ai
        details = await get_bot_details(bot_id)

        # Extract media download URL
        media_url = None
        for rec in details.get("recordings", []):
            shortcuts = rec.get("media_shortcuts", {})
            audio = shortcuts.get("audio_mixed") or {}
            video = shortcuts.get("video_mixed") or {}
            media_url = (
                (audio.get("data") or {}).get("download_url") or
                (video.get("data") or {}).get("download_url")
            )
            if media_url:
                break

        if not media_url:
            print(f"[Pipeline] No media URL found for bot {bot_id}. Will retry next poll.")
            in_progress.discard(bot_id)
            return

        # Meeting metadata
        # ended_at: use recording's completed_at (real_meeting_ended_at doesn't exist in API)
        recordings = details.get("recordings", [])
        ended_at = recordings[0].get("completed_at", "") if recordings else ""

        # participants: fetch from Recall.ai participants endpoint
        participants = []
        try:
            participant_events = (recordings[0].get("media_shortcuts", {}).get("participant_events") or {})
            participants_url = (participant_events.get("data") or {}).get("participants_download_url", "")
            if participants_url:
                async with httpx.AsyncClient(timeout=30) as client:
                    p_resp = await client.get(participants_url)
                if p_resp.status_code == 200:
                    participants = [p.get("name", "") for p in p_resp.json() if p.get("name")]
        except Exception as e:
            print(f"[Pipeline] Could not fetch participants: {e}")

        metadata = {
            "participants": participants,
            "started_at": details.get("join_at", ""),
            "ended_at": ended_at,
            "slack_channel": details.get("meeting_url", "")
        }

        # Step 1 — Stream media to temp file (safe for 2-hour videos, no RAM crash)
        print("[Step 1] Downloading media from Recall.ai...")
        tmp_fd, tmp_media_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        try:
            downloaded = 0
            async with httpx.AsyncClient(timeout=800) as client:
                async with client.stream("GET", media_url) as dl:
                    dl.raise_for_status()
                    with open(tmp_media_path, "wb") as f:
                        async for chunk in dl.aiter_bytes(1024 * 1024):  # 1MB at a time
                            f.write(chunk)
                            downloaded += len(chunk)
            print(f"[Step 1] Downloaded {downloaded / 1024 / 1024:.1f} MB")

            # Step 2 — Transcribe with Sarvam AI (Hindi → English)
            print("[Step 2] Transcribing with Sarvam AI...")
            transcript = await transcribe_audio(tmp_media_path)
            print(f"[Step 2] Transcript: {transcript[:200]}...")

        finally:
            if os.path.exists(tmp_media_path):
                os.unlink(tmp_media_path)

        # Step 3 — Structure notes with GPT-4o Mini
        print("[Step 3] Structuring notes with GPT-4o Mini...")
        notes = await structure_notes(transcript, metadata["participants"])
        print(f"[Step 3] Title: {notes.get('meeting_title')}")

        # Step 4 — Check if worth logging, then create ClickUp tasks
        if not notes.get("worth_logging", True):
            reason = notes.get("skip_reason", "Not valuable enough")
            print(f"[Step 4] Skipping ClickUp — {reason}")
        else:
            print("[Step 4] Meeting worth logging. Output:")
            print(json.dumps(notes, indent=2, ensure_ascii=False))
            await create_meeting_task(notes, metadata)

        await mark_processed(bot_id)
        in_progress.discard(bot_id)
        print(f"[Pipeline] Done. Bot {bot_id} marked as processed.")

    except Exception as e:
        in_progress.discard(bot_id)
        failed_bots.add(bot_id)
        print(f"[Pipeline] ERROR for bot {bot_id}: {e}")
        print(f"[Pipeline] Bot added to failed list — won't retry until server restarts.")


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "Huddle pipeline running", "time": datetime.utcnow().isoformat()}


@app.post("/webhook/recall")
async def recall_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()

    # Verify webhook signature
    signature = request.headers.get("X-Recall-Signature", "")
    if not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body)
    event = payload.get("event", "")

    # Only process when bot recording is fully done
    if event != "bot.done":
        print(f"[Webhook] Ignored — event: {event}")
        return {"status": "ignored"}

    bot_id = payload["data"]["bot"]["id"]

    # Skip if already processed
    if await is_processed(bot_id):
        print(f"[Webhook] Bot {bot_id} already processed. Skipping.")
        return {"status": "already_processed"}

    print(f"[Webhook] Recording done for bot {bot_id}. Starting pipeline...")

    # Run pipeline in background — webhook returns immediately to Recall.ai
    background_tasks.add_task(run_pipeline, bot_id)

    return {"status": "received", "bot_id": bot_id}


@app.post("/webhook/slack")
async def slack_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()

    # Verify Slack signing secret
    if not verify_slack_signature(body, request.headers):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    payload = json.loads(body)

    # Slack requires this one-time challenge when you first register the URL
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})
    event_type = event.get("type", "")

    # Log all events during initial setup so we can see exact Slack payload format
    print(f"[Slack] Event: {event_type}")
    print(f"[Slack] Payload: {json.dumps(event, indent=2)}")

    # Detect huddle start — fires when first person joins a huddle in a channel
    if event_type == "channel_huddle_updated":
        huddle = event.get("huddle", {})
        channel_id = event.get("channel_id", "")
        attendee_count = huddle.get("attendee_count", 0)

        # Only send bot when huddle first starts (attendee_count = 1 = first person joined)
        # and we haven't already sent a bot to this channel's current huddle
        if attendee_count == 1 and channel_id and channel_id not in active_huddles:
            team_id = payload.get("team_id", "")
            huddle_url = f"https://app.slack.com/huddle/{team_id}/{channel_id}"
            active_huddles.add(channel_id)
            print(f"[AutoJoin] Huddle started in {channel_id}. Sending Recall bot...")
            background_tasks.add_task(send_recall_bot_to_huddle, huddle_url, channel_id)

        # Clear from active set when huddle ends (no more attendees)
        elif attendee_count == 0 and channel_id in active_huddles:
            active_huddles.discard(channel_id)
            print(f"[AutoJoin] Huddle ended in {channel_id}. Channel cleared.")

    return {"status": "ok"}
