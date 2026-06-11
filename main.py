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
import secrets
import tempfile
import time
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from dotenv import load_dotenv
from upstash_redis.asyncio import Redis
from services.transcriber import transcribe_audio_detailed
from services.summarizer import structure_notes, extract_meeting_keywords
from services.clickup import create_meeting_task, search_relevant_tasks
from services.slack_notifier import send_meeting_dms
from services.slack_interact import handle_interaction
from services.clickup_brain import (
    assign_participant_names,
    format_speaker_transcript,
    send_clickup_brain_channel_post,
)
from services.hinglish_transcript import build_validated_hinglish_transcript

load_dotenv()

RECALL_API_KEY        = os.getenv("RECALL_API_KEY")
RECALL_BASE_URL       = "https://ap-northeast-1.recall.ai/api/v1"
RECALL_MAX_RETRIES    = 5
RECALL_WEBHOOK_SECRET = os.getenv("RECALL_WEBHOOK_SECRET", "")
SLACK_SIGNING_SECRET  = os.getenv("SLACK_SIGNING_SECRET", "")
ENABLE_HINGLISH_TRANSCRIPT = os.getenv("ENABLE_HINGLISH_TRANSCRIPT", "true").strip().lower() not in {"0", "false", "no", "off"}

# ── Private-huddle OAuth (per-user Slack token → Recall.ai v2) ────────────────
RECALL_BASE_URL_V2   = "https://ap-northeast-1.recall.ai/api/v2"  # same region as v1
RECALL_SLACK_TEAM_ID = os.getenv("RECALL_SLACK_TEAM_ID", "")
SLACK_CLIENT_ID      = os.getenv("SLACK_CLIENT_ID", "")
SLACK_CLIENT_SECRET  = os.getenv("SLACK_CLIENT_SECRET", "")
SLACK_REDIRECT_URI   = os.getenv("SLACK_REDIRECT_URI", "")

# ── Huddle-start nudge (DM user to add the recorder when no bot auto-joins) ────
RECALL_BOT_SLACK_USER_ID   = os.getenv("RECALL_BOT_SLACK_USER_ID", "U0ALGV44B2R")  # "Huddle Notes Bot" member
APP_BOT_SLACK_USER_ID      = os.getenv("APP_BOT_SLACK_USER_ID", "U0AMDQMKMKK")     # "Haddle Bot" app user
HUDDLE_NUDGE_GRACE_SECONDS = int(os.getenv("HUDDLE_NUDGE_GRACE_SECONDS", "10"))
BOT_KEY_TTL           = 90 * 24 * 60 * 60   # 90 days in seconds
PIPELINE_LOCK_TTL_SECONDS = int(os.getenv("PIPELINE_LOCK_TTL_SECONDS", str(6 * 60 * 60)))

redis = Redis(
    url=os.getenv("UPSTASH_REDIS_URL"),
    token=os.getenv("UPSTASH_REDIS_TOKEN")
)

app = FastAPI()

POLL_INTERVAL_SECONDS   = 1800   # poll every 30 minutes as backup (webhook is primary trigger)
in_progress: set        = set()  # bots currently being processed
failed_bots: dict       = {}     # bot_id → failure count; skip after 3 failures
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
    since = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
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
    cutoff   = datetime.now(timezone.utc) - timedelta(hours=24)

    fresh_bots = []
    for b in all_bots:
        # Skip bots whose meeting started more than 24 hours ago — stale, never process
        join_at_str = b.get("join_at") or b.get("created_at", "")
        try:
            join_at = datetime.fromisoformat(join_at_str.replace("Z", "+00:00"))
            if join_at < cutoff:
                print(f"[Poller] Bot {b['id']} is stale (join_at={join_at_str}). Marking processed and skipping.")
                await mark_processed(b["id"])
                continue
        except Exception:
            pass  # If we can't parse the date, let is_processed decide
        fresh_bots.append(b)

    new_bots = [b for b in fresh_bots if not await is_processed(b["id"])]

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


async def acquire_pipeline_lock(bot_id: str, redis_client=redis) -> bool:
    try:
        result = await redis_client.set(
            f"bot_lock:{bot_id}",
            "1",
            nx=True,
            ex=PIPELINE_LOCK_TTL_SECONDS,
        )
        return result in (True, "OK", 1)
    except Exception as e:
        print(f"[Redis] acquire_pipeline_lock failed for {bot_id}: {e} — using local in-process guard only")
        return True


async def release_pipeline_lock(bot_id: str, redis_client=redis):
    try:
        await redis_client.delete(f"bot_lock:{bot_id}")
    except Exception as e:
        print(f"[Redis] release_pipeline_lock failed for {bot_id}: {e}")


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
    try:
        request_time = int(timestamp)
    except ValueError:
        return False
    # Reject replayed requests older than 5 minutes
    if abs(time.time() - request_time) > 300:
        return False
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, slack_sig)


def _recall_done_bot_id(payload: dict) -> str:
    event = payload.get("event", "")
    data = payload.get("data") or {}
    bot = data.get("bot") or {}
    status = data.get("status") or {}

    if event == "bot.done":
        return bot.get("id") or data.get("bot_id") or ""
    if event == "recording.done":
        return bot.get("id") or data.get("bot_id") or ""
    if event == "bot.status_change" and status.get("code") == "done":
        return bot.get("id") or data.get("bot_id") or ""
    return ""


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
    if await is_processed(bot_id) or bot_id in in_progress or failed_bots.get(bot_id, 0) >= 3:
        print(f"[Pipeline] Bot {bot_id} already processed/in-progress/failed. Skipping.")
        return

    lock_acquired = await acquire_pipeline_lock(bot_id)
    if not lock_acquired:
        print(f"[Pipeline] Bot {bot_id} is locked by another worker. Skipping.")
        return

    in_progress.add(bot_id)
    print(f"\n[Pipeline] Processing bot: {bot_id}")
    tmp_media_path = ""

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
            return

        # Meeting metadata
        # ended_at: use recording's completed_at (real_meeting_ended_at doesn't exist in API)
        recordings = details.get("recordings", [])
        ended_at = recordings[0].get("completed_at", "") if recordings else ""

        # participants + speaker timeline: fetch Recall.ai participant artifacts.
        # Sarvam owns transcription/diarization; Recall timeline is used only to map speakers to names.
        participants = []
        speaker_timeline = []
        try:
            participant_events = (recordings[0].get("media_shortcuts", {}).get("participant_events") or {})
            participant_data = participant_events.get("data") or {}
            participants_url = participant_data.get("participants_download_url", "")
            if participants_url:
                async with httpx.AsyncClient(timeout=30) as client:
                    p_resp = await client.get(participants_url)
                if p_resp.status_code == 200:
                    participants = [p.get("name", "") for p in p_resp.json() if p.get("name")]
            speaker_timeline_url = (
                participant_data.get("speaker_timeline_download_url")
                or participant_data.get("speakerTimeline_download_url")
                or participant_data.get("speaker_timeline_url")
            )
            if speaker_timeline_url:
                async with httpx.AsyncClient(timeout=30) as client:
                    st_resp = await client.get(speaker_timeline_url)
                if st_resp.status_code == 200:
                    timeline_payload = st_resp.json()
                    if isinstance(timeline_payload, list):
                        speaker_timeline = timeline_payload
                    elif isinstance(timeline_payload, dict):
                        for key in ("speaker_timeline", "timeline", "events", "results", "data"):
                            value = timeline_payload.get(key)
                            if isinstance(value, list):
                                speaker_timeline = value
                                break
        except Exception as e:
            print(f"[Pipeline] Could not fetch Recall participant artifacts: {e}")

        # Calculate duration in minutes
        duration_minutes = 0
        duration_seconds = 0.0
        try:
            if recordings:
                rec0 = recordings[0]
                t_start = datetime.fromisoformat(rec0.get("started_at", "").replace("Z", "+00:00"))
                t_end   = datetime.fromisoformat(rec0.get("completed_at", "").replace("Z", "+00:00"))
                duration_seconds = max(0.0, (t_end - t_start).total_seconds())
                duration_minutes = int(duration_seconds / 60)
        except Exception:
            pass

        metadata = {
            "meeting_id": bot_id,
            "participants": participants,
            "started_at": details.get("join_at", ""),
            "ended_at": ended_at,
            "duration_minutes": duration_minutes,
            "duration_seconds": duration_seconds,
            "slack_channel": details.get("meeting_url", "")
        }

        # Step 1 — Stream media to temp file (safe for 2-hour videos, no RAM crash)
        print("[Step 1] Downloading media from Recall.ai...")
        tmp_fd, tmp_media_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
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
        transcript_result = await transcribe_audio_detailed(tmp_media_path)
        transcript = transcript_result.text
        print(f"[Step 2] Transcript: {transcript[:200]}...")

        # Step 3 — Extract keywords from transcript → search ClickUp workspace
        relevant_tasks = []
        try:
            keywords = await extract_meeting_keywords(transcript)
            if keywords:
                relevant_tasks = await search_relevant_tasks(keywords)
        except Exception as e:
            print(f"[Step 3] Could not get relevant tasks: {e}")

        # Step 4 — Structure notes with GPT
        print("[Step 4] Structuring notes with GPT...")
        notes = await structure_notes(
            transcript,
            metadata["participants"],
            metadata["duration_minutes"],
            relevant_tasks or None
        )
        print(f"[Step 4] Title: {notes.get('meeting_title')}")

        # Step 5 — Check if worth logging, then create ClickUp task
        if not notes.get("worth_logging", True):
            reason = notes.get("skip_reason", "Not valuable enough")
            print(f"[Step 5] Skipping ClickUp — {reason}")
        else:
            print("[Step 5] Creating ClickUp task...")
            await create_meeting_task(notes, metadata)

        # Step 6 — Send Slack DMs to each participant with action points
        print("[Step 6] Sending Slack DMs...")
        try:
            await send_meeting_dms(notes, metadata)
        except Exception as e:
            print(f"[Step 6] Slack DM failed (non-fatal): {e}")

        # Step 7 — Send full speaker transcript to ClickUp Brain Slack channel
        print("[Step 7] Sending ClickUp Brain channel transcript...")
        try:
            named_transcript = assign_participant_names(
                transcript_result,
                speaker_timeline,
                duration_seconds=metadata.get("duration_seconds") or None,
            )
            transcript_text = format_speaker_transcript(named_transcript, metadata, notes)
            hinglish_transcript_text = None
            if ENABLE_HINGLISH_TRANSCRIPT and tmp_media_path and os.path.exists(tmp_media_path):
                try:
                    print("[Step 7] Generating validated Roman-Hinglish transcript...")
                    hinglish_transcript = await build_validated_hinglish_transcript(
                        tmp_media_path,
                        named_transcript,
                        speaker_timeline,
                        duration_seconds=metadata.get("duration_seconds") or None,
                    )
                    hinglish_transcript_text = format_speaker_transcript(
                        hinglish_transcript,
                        metadata,
                        notes,
                        heading="Validated Roman-Hinglish Speaker Transcript",
                    )
                except Exception as e:
                    print(f"[Step 7] Hinglish transcript generation failed (non-fatal): {e}")
            await send_clickup_brain_channel_post(
                notes,
                metadata,
                named_transcript,
                transcript_text,
                hinglish_transcript_text=hinglish_transcript_text,
            )
        except Exception as e:
            print(f"[Step 7] ClickUp Brain channel post failed (non-fatal): {e}")

        await mark_processed(bot_id)
        print(f"[Pipeline] Done. Bot {bot_id} marked as processed.")

    except Exception as e:
        count = failed_bots.get(bot_id, 0) + 1
        failed_bots[bot_id] = count
        print(f"[Pipeline] ERROR for bot {bot_id} (attempt {count}/3): {e}")
        if count >= 3:
            print(f"[Pipeline] Bot {bot_id} failed 3 times — skipping permanently.")
    finally:
        if tmp_media_path and os.path.exists(tmp_media_path):
            os.unlink(tmp_media_path)
        in_progress.discard(bot_id)
        await release_pipeline_lock(bot_id)


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

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None
    event = payload.get("event", "")

    # Only process when bot recording is fully done
    bot_id = _recall_done_bot_id(payload)
    if not bot_id:
        print(f"[Webhook] Ignored — event: {event}")
        return {"status": "ignored"}

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

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None

    # Slack requires this one-time challenge when you first register the URL
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})
    event_type = event.get("type", "")

    print(f"[Slack] Event: {event_type}")

    # Handle DM messages — used for API key updates (user pastes pk_xxx in bot DM)
    if event_type == "message" and event.get("channel_type") == "im":
        # Ignore bot's own messages
        if not event.get("bot_id") and not event.get("subtype"):
            background_tasks.add_task(_handle_dm_message, event)
        return {"status": "ok"}

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

    # App Home opened — show enable button (or confirmation if already authorized)
    if event_type == "app_home_opened":
        user_id = event.get("user", "")
        if user_id:
            already_auth = await redis.exists(f"slack_oauth:{user_id}") == 1
            background_tasks.add_task(_publish_app_home, user_id, already_auth)

    # Huddle started — nudge the user to add the recorder, unless a bot auto-joins (public/member channel)
    if event_type == "user_huddle_changed":
        u            = event.get("user", {}) or {}
        user_id      = u.get("id", "")
        profile      = u.get("profile", {}) or {}
        huddle_state = profile.get("huddle_state", "")
        call_id      = profile.get("huddle_state_call_id", "")
        if (
            huddle_state == "in_a_huddle"
            and user_id
            and call_id
            and user_id not in (RECALL_BOT_SLACK_USER_ID, APP_BOT_SLACK_USER_ID)
        ):
            started_ts = datetime.now(timezone.utc).isoformat()
            background_tasks.add_task(_huddle_nudge_flow, user_id, call_id, started_ts)

    return {"status": "ok"}


async def _send_dm(slack_user_id: str, text: str):
    """Open a DM channel with a user and post a plain-text message."""
    headers = {
        "Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        open_resp = await client.post(
            "https://slack.com/api/conversations.open",
            headers=headers,
            json={"users": slack_user_id},
        )
    channel_id = open_resp.json().get("channel", {}).get("id")
    if not channel_id:
        return
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json={"channel": channel_id, "text": text},
        )


async def _publish_app_home(user_id: str, authorized: bool):
    """
    Publish the App Home tab view.
    Authorized users see a confirmation; others see the enable button.
    """
    base_url = SLACK_REDIRECT_URI.replace("/auth/slack/callback", "")
    auth_url = f"{base_url}/auth/slack?user_id={user_id}"
    headers = {
        "Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}",
        "Content-Type": "application/json",
    }
    if authorized:
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":white_check_mark: *Private huddle recording is enabled.*\nThe bot will join your DM and private channel huddles when one starts.",
                },
            }
        ]
    else:
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        ":microphone: *Enable private call recording*\n"
                        "The bot already joins public channel huddles. "
                        "Authorize once below so it can also join your private channel and DM huddles."
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Enable Private Recording"},
                        "style": "primary",
                        "url": auth_url,
                        "action_id": "enable_private_recording",
                    }
                ],
            },
        ]
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            "https://slack.com/api/views.publish",
            headers=headers,
            json={"user_id": user_id, "view": {"type": "home", "blocks": blocks}},
        )


async def _recall_bot_joined_since(cutoff: datetime) -> bool:
    """True if any Recall bot began joining a call at/after cutoff (i.e. a huddle is being covered)."""
    headers = {"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{RECALL_BASE_URL}/bot/",
                headers=headers,
                params={"ordering": "-created_at", "page_size": 5},
            )
        resp.raise_for_status()
        for bot in resp.json().get("results", []):
            changes = bot.get("status_changes") or []
            if not changes:
                continue
            first_ts = changes[0].get("created_at", "")
            try:
                t = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if t >= cutoff:
                return True
    except Exception as e:
        print(f"[Huddle Nudge] Recall check failed: {e}")
    return False


async def _huddle_nudge_flow(user_id: str, call_id: str, started_ts: str):
    """
    After an authorized user starts a huddle: claim it (dedup), wait a grace
    window, and nudge to add the recorder ONLY if no bot auto-joined.
    """
    # 1. Claim — exactly one nudge per huddle, atomic across racing participants/retries
    try:
        claimed = await redis.set(f"huddle_nudge:{call_id}", user_id, nx=True, ex=2 * 60 * 60)
    except Exception as e:
        print(f"[Huddle Nudge] claim failed for {call_id}: {e} — proceeding")
        claimed = True  # fail open — a rare dup beats silent failure
    if claimed not in (True, "OK", 1):
        return

    # 2. Grace window — give auto-join (public / member channel) or a manual add time to appear
    await asyncio.sleep(HUDDLE_NUDGE_GRACE_SECONDS)

    # 3. Suppress if a bot showed up (public auto-join, member channel, already recording)
    try:
        cutoff = datetime.fromisoformat(started_ts) - timedelta(seconds=5)
    except Exception:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=HUDDLE_NUDGE_GRACE_SECONDS + 5)
    if await _recall_bot_joined_since(cutoff):
        print(f"[Huddle Nudge] Bot already covering huddle {call_id} — no nudge.")
        return

    # 4. Nudge — sent by the app (Haddle Bot), telling the user to add the recorder member
    await _send_dm(
        user_id,
        ":red_circle: *Add \"Huddle Notes Bot\" to record this huddle.*\n"
        "Tap *Invite people* (person+ icon) in the huddle → pick *Huddle Notes Bot*.\n"
        "Notes post to ClickUp automatically.",
    )
    print(f"[Huddle Nudge] Sent to {user_id} for huddle {call_id}.")


async def _handle_dm_message(event: dict):
    """
    Handles DMs sent to the bot.
    - If user types 'apikey', 'api key', or 'api' → send instructions
    - If message starts with pk_ → validate and save as ClickUp API key
    """
    from services.clickup import validate_clickup_api_key
    from upstash_redis.asyncio import Redis as _Redis

    user_id  = event.get("user", "")
    text     = (event.get("text") or "").strip()
    channel  = event.get("channel", "")

    slack_headers = {
        "Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}",
        "Content-Type": "application/json"
    }

    # Keyword trigger — send instructions
    if text.lower() in ("apikey", "api key", "api"):
        instructions = (
            "*How to get your ClickUp API key:*\n"
            "1. Open ClickUp\n"
            "2. Right-click your avatar (top-right corner)\n"
            "3. Click *Settings*\n"
            "4. Click *ClickUp API*\n"
            "5. Generate your token and paste it here"
        )
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                "https://slack.com/api/chat.postMessage",
                headers=slack_headers,
                json={"channel": channel, "text": instructions}
            )
        return

    # API key submission
    if not text.startswith("pk_"):
        return  # ignore everything else

    _redis = _Redis(url=os.getenv("UPSTASH_REDIS_URL"), token=os.getenv("UPSTASH_REDIS_TOKEN"))
    valid, name = await validate_clickup_api_key(text)

    if valid:
        await _redis.set(f"clickup_key:{user_id}", text)
        reply = f":white_check_mark: Verified and saved — connected as *{name}*"
        print(f"[Slack DM] API key saved for {user_id} ({name})")
    else:
        reply = ":x: Invalid API key. Please check and try again."

    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            "https://slack.com/api/chat.postMessage",
            headers=slack_headers,
            json={"channel": channel, "text": reply}
        )



@app.get("/auth/slack")
async def slack_oauth_start(request: Request):
    """OAuth entry — generate CSRF state, redirect user to Slack consent screen."""
    state = secrets.token_urlsafe(16)
    user_id = request.query_params.get("user_id", "")
    await redis.set(f"slack_oauth_state:{state}", user_id or "unknown", ex=300)
    params = urlencode({
        "client_id": SLACK_CLIENT_ID,
        "user_scope": "channels:read groups:read im:read mpim:read team:read users:read",
        "redirect_uri": SLACK_REDIRECT_URI,
        "state": state,
    })
    return RedirectResponse(f"https://slack.com/oauth/v2/authorize?{params}")


@app.get("/auth/slack/callback")
async def slack_oauth_callback(code: str, state: str):
    """Slack redirect target — exchange code for user token, register it with Recall.ai."""
    # 1. Verify CSRF state
    stored = await redis.get(f"slack_oauth_state:{state}")
    if not stored:
        raise HTTPException(status_code=400, detail="Invalid or expired state")
    await redis.delete(f"slack_oauth_state:{state}")

    # 2. Exchange code → user token (xoxp- prefix)
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://slack.com/api/oauth.v2.access",
            data={
                "client_id": SLACK_CLIENT_ID,
                "client_secret": SLACK_CLIENT_SECRET,
                "code": code,
                "redirect_uri": SLACK_REDIRECT_URI,
            },
        )
    data = resp.json()
    if not data.get("ok"):
        raise HTTPException(status_code=400, detail=f"Slack OAuth failed: {data.get('error')}")

    authed_user = data.get("authed_user") or {}
    user_token    = authed_user.get("access_token", "")  # xoxp- user token, NOT bot token
    slack_user_id = authed_user.get("id", "")
    if not user_token or not slack_user_id:
        raise HTTPException(status_code=400, detail="Slack did not return a user token")

    # 3. Register user token with Recall.ai v2
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"{RECALL_BASE_URL_V2}/slack-teams/{RECALL_SLACK_TEAM_ID}/oauth-tokens/",
            headers={
                "Authorization": f"Token {RECALL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"token": user_token},
        )
    if r.status_code not in (200, 201):
        print(f"[OAuth] Recall token POST failed ({r.status_code}): {r.text}")
        raise HTTPException(status_code=500, detail="Failed to register token with Recall")
    print(f"[OAuth] Registered Recall token for user {slack_user_id} ({r.status_code})")

    # 4. Mark user authorized + send confirm DM + refresh App Home
    await redis.set(f"slack_oauth:{slack_user_id}", "1")
    await _send_dm(
        slack_user_id,
        ":white_check_mark: Private huddle recording enabled. The bot will now join your DM and private channel huddles.",
    )
    try:
        await _publish_app_home(slack_user_id, True)
    except Exception as e:
        print(f"[OAuth] App Home refresh failed: {e}")

    return HTMLResponse("<html><body><h2>Done! You can close this tab.</h2></body></html>")


@app.post("/webhook/slack-options")
async def slack_options(request: Request):
    """
    Called by Slack for external_select dropdowns — when modal opens (empty query)
    or user types a search string. Must respond within 3 seconds.
    Set this URL in: Slack App → Interactivity & Shortcuts → Select Menus → Options Load URL.
    """
    body = await request.body()

    if not verify_slack_signature(body, request.headers):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    form    = await request.form()
    payload    = json.loads(form.get("payload", "{}"))
    query      = payload.get("value", "").strip()
    action_id  = payload.get("action_id", "")
    # Slack can send state in different shapes for options requests.
    state_vals = (
        payload.get("view", {}).get("state", {}).get("values", {}) or
        payload.get("state", {}).get("values", {}) or
        {}
    )

    from services.clickup import (
        get_backlog_tasks_cached,
        search_backlog_by_query,
        get_parent_tasks_for_options,
        get_targets_for_parent,
        search_subtasks_global,
    )

    all_tasks = await get_backlog_tasks_cached()

    # New flow: parent task selector
    if action_id == "selected_parent":
        parent_tasks = get_parent_tasks_for_options(query, all_tasks)
        print(f"[Slack Options] parent query='{query}' → {len(parent_tasks)} matches")
        options = []
        for t in parent_tasks:
            name      = t.get("name", "")[:75]
            assignees = t.get("assignees", "")
            option    = {"text": {"type": "plain_text", "text": name}, "value": f"p:{t['id']}"}
            if assignees:
                option["description"] = {"type": "plain_text", "text": assignees[:75]}
            options.append(option)
        if not options:
            options = [{"text": {"type": "plain_text", "text": "No parent tasks found"}, "value": "none"}]
        return {"options": options}

    # New flow: target selector (parent activity OR one of its subtasks)
    if action_id.startswith("selected_target"):
        action_parent_id = ""
        if "__" in action_id:
            action_parent_id = action_id.split("__", 1)[1].strip()

        selected_parent_val = (
            state_vals.get("parent_select", {})
            .get("selected_parent", {})
            .get("selected_option", {})
            .get("value", "")
        )

        # Strongest source: action_id carries current parent id after modal refresh.
        if action_parent_id:
            selected_parent_val = f"p:{action_parent_id}"

        # Fallback from modal private metadata if present.
        if not selected_parent_val:
            try:
                private_meta_raw = payload.get("view", {}).get("private_metadata", "{}")
                private_meta = json.loads(private_meta_raw) if private_meta_raw else {}
                if isinstance(private_meta, dict):
                    selected_parent_val = private_meta.get("selected_parent_value", "") or ""
            except Exception:
                pass

        # Fallback: scan all state entries for any selected_option value prefixed with p:
        if not selected_parent_val:
            for block_data in state_vals.values():
                if not isinstance(block_data, dict):
                    continue
                for action_data in block_data.values():
                    if not isinstance(action_data, dict):
                        continue
                    candidate = (action_data.get("selected_option", {}) or {}).get("value", "")
                    if isinstance(candidate, str) and candidate.startswith("p:"):
                        selected_parent_val = candidate
                        break
                if selected_parent_val:
                    break

        # Final fallback: use value cached from selected_parent block_action.
        if not selected_parent_val:
            view_id = payload.get("view", {}).get("id", "")
            user_id = payload.get("user", {}).get("id", "")
            if view_id and user_id:
                try:
                    cached_val = await redis.get(f"parent_pick:{view_id}:{user_id}")
                    if isinstance(cached_val, bytes):
                        cached_val = cached_val.decode("utf-8", errors="ignore")
                    if cached_val:
                        selected_parent_val = cached_val
                except Exception as e:
                    print(f"[Slack Options] parent cache lookup failed: {e}")

        parent_id = selected_parent_val[2:] if isinstance(selected_parent_val, str) and selected_parent_val.startswith("p:") else selected_parent_val
        parent_task, subtasks = get_targets_for_parent(parent_id, query, all_tasks)
        print(f"[Slack Options] target parent='{parent_id}' query='{query}' → {len(subtasks)} subtasks")

        options = []
        if parent_task:
            parent_name = parent_task.get("name", "")[:67]
            options.append({
                "text": {"type": "plain_text", "text": f"Parent: {parent_name}"},
                "value": f"p:{parent_task['id']}",
                "description": {"type": "plain_text", "text": "Post to parent activity"}
            })

            seen_subtask_ids = set()
            for st in subtasks:
                sub_name   = st.get("name", "")[:65]
                assignees  = st.get("assignees", "")
                seen_subtask_ids.add(str(st.get("id")))
                sub_option = {
                    "text":  {"type": "plain_text", "text": f"Subtask: {sub_name}"},
                    "value": f"s:{st['id']}:{parent_task['id']}"
                }
                if assignees:
                    sub_option["description"] = {"type": "plain_text", "text": assignees[:75]}
                options.append(sub_option)

            if query:
                global_subtasks = search_subtasks_global(query, all_tasks)
                global_subtasks = [
                    st for st in global_subtasks
                    if str(st.get("id")) not in seen_subtask_ids
                ]
                print(f"[Slack Options] global subtask query='{query}' with parent='{parent_id}' -> {len(global_subtasks)} extra matches")
                for st in global_subtasks:
                    sub_name = st.get("name", "")[:65]
                    parent_name_for_desc = st.get("parent_name", "")
                    parent_id_for_value = st.get("parent_id", "")
                    description = f"Parent: {parent_name_for_desc}" if parent_name_for_desc else "Parent not found in cache"
                    options.append({
                        "text": {"type": "plain_text", "text": f"Subtask: {sub_name}"},
                        "value": f"s:{st['id']}:{parent_id_for_value}",
                        "description": {"type": "plain_text", "text": description[:75]}
                    })
        elif query:
            global_subtasks = search_subtasks_global(query, all_tasks)
            print(f"[Slack Options] global subtask query='{query}' -> {len(global_subtasks)} matches")
            for st in global_subtasks:
                sub_name = st.get("name", "")[:65]
                parent_name = st.get("parent_name", "")
                parent_id_for_value = st.get("parent_id", "")
                description = f"Parent: {parent_name}" if parent_name else "Parent not found in cache"
                options.append({
                    "text": {"type": "plain_text", "text": f"Subtask: {sub_name}"},
                    "value": f"s:{st['id']}:{parent_id_for_value}",
                    "description": {"type": "plain_text", "text": description[:75]}
                })
        else:
            options = [{
                "text": {"type": "plain_text", "text": "Pick parent task or search subtask"},
                "value": "none"
            }]

        if not options:
            options = [{"text": {"type": "plain_text", "text": "No targets found"}, "value": "none"}]
        return {"options": options[:100]}

    # Legacy flow (backward compatibility for old modal messages)
    if query:
        tasks = search_backlog_by_query(query, all_tasks)
        print(f"[Slack Options] legacy query='{query}' → {len(tasks)} matches")
    else:
        assigned   = [t for t in all_tasks if t.get("assignees")]
        unassigned = [t for t in all_tasks if not t.get("assignees")]
        tasks      = (assigned + unassigned)[:100]
        print(f"[Slack Options] legacy default load: {len(tasks)} tasks")

    options = []
    for t in tasks:
        name      = t.get("name", "")[:75]
        assignees = t.get("assignees", "")
        option    = {"text": {"type": "plain_text", "text": name}, "value": t["id"]}
        if assignees:
            option["description"] = {"type": "plain_text", "text": assignees[:75]}
        options.append(option)

    if not options:
        options = [{"text": {"type": "plain_text", "text": "No tasks found"}, "value": "none"}]

    return {"options": options}


@app.post("/webhook/slack-interact")
async def slack_interact(request: Request, background_tasks: BackgroundTasks):
    """
    Receives all Slack button clicks and modal submissions.
    Must respond within 3 seconds — heavy work runs in background.
    """
    body = await request.body()

    if not verify_slack_signature(body, request.headers):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    # Slack sends interactions as URL-encoded form: payload=<json>
    form    = await request.form()
    payload = json.loads(form.get("payload", "{}"))

    interaction_type = payload.get("type")

    # Slack url_verification (shouldn't hit this route but handle just in case)
    if interaction_type == "url_verification":
        return {"challenge": payload.get("challenge")}

    # Modal select changes must be processed synchronously so Slack sees the
    # refreshed view before the user continues inside the same modal.
    if interaction_type == "block_actions":
        actions = payload.get("actions", []) or []
        first_action_id = (actions[0].get("action_id", "") if actions else "")
        if first_action_id == "selected_parent" or first_action_id.startswith("selected_target"):
            await handle_interaction(payload)
            return {"status": "ok"}

    # Process in background — return 200 immediately to Slack
    background_tasks.add_task(handle_interaction, payload)

    # For view_submission (modal submit), return {} to close the modal
    if interaction_type == "view_submission":
        return {}

    return {"status": "ok"}
