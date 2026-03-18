import httpx
import json
import os
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
    "Content-Type": "application/json"
}

# Cache of all Slack workspace users — refreshed every hour
_slack_users_cache: list = []        # [{id, real_name, display_name}]
_slack_users_fetched_at: float = 0   # unix timestamp


async def _load_slack_users():
    """
    Fetch all non-bot, non-deleted Slack workspace members.
    Cached for 1 hour. Handles cursor-based pagination.
    """
    global _slack_users_cache, _slack_users_fetched_at

    if _slack_users_fetched_at and (time.time() - _slack_users_fetched_at) < 3600:
        return  # cache still fresh

    members = []
    cursor  = None

    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor

            r = await client.get(
                "https://slack.com/api/users.list",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                params=params
            )
            data = r.json()

            if not data.get("ok"):
                print(f"[Slack] users.list failed: {data.get('error')}")
                break

            for m in data.get("members", []):
                if m.get("deleted") or m.get("is_bot"):
                    continue
                members.append({
                    "id":           m["id"],
                    "real_name":    m.get("real_name", ""),
                    "display_name": m.get("profile", {}).get("display_name", "")
                })

            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break

    _slack_users_cache     = members
    _slack_users_fetched_at = time.time()
    print(f"[Slack] Users loaded ({len(members)}): {[u['real_name'] for u in members]}")


def _match_slack_user(name: str) -> str | None:
    """
    Fuzzy match a ClickUp display name → Slack user ID.
    Order: exact real_name → exact display_name → first name → last name → any word.
    """
    if not name:
        return None

    name_lower = name.lower().strip()
    name_words = name_lower.split()

    # 1. Exact match on real_name
    for u in _slack_users_cache:
        if u["real_name"].lower() == name_lower:
            return u["id"]

    # 2. Exact match on display_name
    for u in _slack_users_cache:
        if u["display_name"].lower() == name_lower:
            return u["id"]

    # 3. First name match
    if name_words:
        for u in _slack_users_cache:
            real_parts    = u["real_name"].lower().split()
            display_parts = u["display_name"].lower().split()
            if (real_parts and real_parts[0] == name_words[0]) or \
               (display_parts and display_parts[0] == name_words[0]):
                return u["id"]

    # 4. Last name match
    if len(name_words) > 1:
        for u in _slack_users_cache:
            real_parts = u["real_name"].lower().split()
            if len(real_parts) > 1 and real_parts[-1] == name_words[-1]:
                return u["id"]

    # 5. Any word overlap on real_name
    for u in _slack_users_cache:
        real_lower = u["real_name"].lower()
        if any(w in real_lower for w in name_words):
            return u["id"]

    return None


async def send_meeting_dms(notes: dict, metadata: dict):
    """
    Send meeting summary DMs to all participants.
    Matches participant names directly to Slack users — no email needed.
    """
    if not notes.get("worth_logging", True):
        return

    participants = metadata.get("participants", [])
    participant_names = []
    for p in participants:
        if isinstance(p, dict):
            participant_names.append(p.get("name") or p.get("display_name") or "")
        else:
            participant_names.append(str(p))
    participant_names = [n for n in participant_names if n]

    if not participant_names:
        print("[Slack DM] No participants to DM.")
        return

    await _load_slack_users()

    blocks       = _build_dm_blocks(notes, metadata)
    started = metadata.get("started_at", "")
    try:
        dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        dt_ist = dt + timedelta(hours=5, minutes=30)
        dt_str = dt_ist.strftime("%d %b, %I:%M %p")
    except Exception:
        dt_str = datetime.now().strftime("%d %b, %I:%M %p")
    title = notes.get("meeting_title", "Huddle Meeting")
    short_title = title[:40] + "…" if len(title) > 40 else title
    first_names = ", ".join(n.split()[0] for n in participant_names if n)
    fallback_text = f"🎙️ {dt_str} IST · {short_title} · {first_names}"

    for name in participant_names:
        slack_user_id = _match_slack_user(name)
        if not slack_user_id:
            print(f"[Slack DM] No Slack user matched for '{name}' — skipping.")
            continue

        try:
            # Open DM channel
            async with httpx.AsyncClient(timeout=10) as client:
                open_resp = await client.post(
                    "https://slack.com/api/conversations.open",
                    headers=HEADERS,
                    json={"users": slack_user_id}
                )
            channel_id = open_resp.json().get("channel", {}).get("id")
            if not channel_id:
                print(f"[Slack DM] Could not open DM channel for '{name}'.")
                continue

            # Send message
            async with httpx.AsyncClient(timeout=10) as client:
                msg_resp = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers=HEADERS,
                    json={
                        "channel": channel_id,
                        "blocks":  blocks,
                        "text":    fallback_text
                    }
                )
            resp_data = msg_resp.json()
            if resp_data.get("ok"):
                print(f"[Slack DM] Sent to '{name}' (id: {slack_user_id})")
            else:
                print(f"[Slack DM] Failed for '{name}': {resp_data.get('error')}")

        except Exception as e:
            print(f"[Slack DM] Error sending to '{name}': {e}")


def _build_dm_blocks(notes: dict, metadata: dict) -> list:
    """
    Build Slack Block Kit blocks for the meeting DM.
    Contains: header, overview, action points with Confirm / Change buttons.
    """
    title      = notes.get("meeting_title", "Huddle Meeting")
    overview   = notes.get("overview", "")
    next_steps = notes.get("next_steps", [])
    duration   = metadata.get("duration_minutes", 0)
    participants = metadata.get("participants", [])

    parts = []
    for p in participants:
        if isinstance(p, dict):
            parts.append(p.get("name") or p.get("display_name") or "")
        else:
            parts.append(str(p))
    participants_str = " · ".join(p for p in parts if p) or "Unknown"

    started = metadata.get("started_at", "")
    try:
        dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        dt_ist = dt + timedelta(hours=5, minutes=30)
        date_str = dt_ist.strftime("%d %b %Y")
    except Exception:
        date_str = datetime.now().strftime("%d %b %Y")

    meta_for_buttons = {
        "meeting_title":    title,
        "participants_str": participants_str[:200],
        "overview":         overview[:500]
    }

    blocks = [
        {"type": "divider"},
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Meeting Notes  |  {date_str}"}
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{title}*\n:timer_clock: {duration} min   :busts_in_silhouette: {participants_str}"
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*:pushpin: Overview*\n{overview}"}
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:zap: Action Points*\nMap each point to the right ClickUp task."
            }
        }
    ]

    for i, step in enumerate(next_steps, 1):
        if not isinstance(step, dict):
            continue

        task_text  = step.get("task", "")
        task_id    = step.get("clickup_task_id")
        task_name  = step.get("clickup_task_name")
        deadline   = step.get("deadline")

        deadline_str = f"\n_Deadline: {deadline}_" if deadline else ""
        suggestion = (
            f":bulb: Suggested \u2192 *{task_name}*"
            if task_id else
            ":bulb: No existing task matched"
        )

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{i}.* {task_text}{deadline_str}\n{suggestion}"
            }
        })

        button_value = json.dumps({
            "task_text":         task_text[:300],
            "clickup_task_id":   task_id,
            "clickup_task_name": task_name or "",
            "deadline":          deadline or "",
            "meta":              meta_for_buttons
        })

        create_value = json.dumps({
            "task_text": task_text[:300],
            "deadline":  deadline or "",
            "meta":      meta_for_buttons
        })

        create_btn = {
            "type":      "button",
            "text":      {"type": "plain_text", "text": "Create New Task"},
            "action_id": f"create_{i}",
            "value":     create_value
        }

        if task_id:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type":      "button",
                        "text":      {"type": "plain_text", "text": "Confirm"},
                        "style":     "primary",
                        "action_id": f"confirm_{i}",
                        "value":     button_value
                    },
                    {
                        "type":      "button",
                        "text":      {"type": "plain_text", "text": "Change Task"},
                        "action_id": f"change_{i}",
                        "value":     button_value
                    },
                    create_btn
                ]
            })
        else:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type":      "button",
                        "text":      {"type": "plain_text", "text": "Pick a Task"},
                        "action_id": f"pick_{i}",
                        "value":     button_value
                    },
                    create_btn
                ]
            })

    return blocks
