import httpx
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

CLICKUP_API_KEY = os.getenv("CLICKUP_API_KEY")
CLICKUP_LIST_ID = os.getenv("CLICKUP_LIST_ID")
BASE_URL = "https://api.clickup.com/api/v2"

HEADERS = {
    "Authorization": CLICKUP_API_KEY,
    "Content-Type": "application/json"
}


def _format_participants(participants: list) -> str:
    """Handles both string list and dict list from Recall.ai."""
    if not participants:
        return "Unknown"
    parts = []
    for p in participants:
        if isinstance(p, dict):
            parts.append(p.get("name") or p.get("display_name") or "Unknown")
        else:
            parts.append(str(p))
    return ", ".join(filter(None, parts)) or "Unknown"


def _parse_datetime(iso_str: str):
    """Parses ISO datetime string and converts to IST (UTC+5:30)."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt + timedelta(hours=5, minutes=30)
    except Exception:
        return datetime.now()


def _build_task_description(notes: dict, metadata: dict) -> str:
    participants = _format_participants(metadata.get("participants", []))

    started = metadata.get("started_at", "")
    ended = metadata.get("ended_at", "")

    start_dt = _parse_datetime(started)
    start_time = start_dt.strftime("%I:%M %p")

    if ended:
        end_dt = _parse_datetime(ended)
        duration = f"{start_time} – {end_dt.strftime('%I:%M %p')} IST"
    else:
        duration = f"{start_time} IST"

    decisions = notes.get("decisions", [])
    decisions_text = "\n".join(f"- {d}" for d in decisions) if decisions else "_None recorded_"

    return f"""**Participants:** {participants}
**Time:** {duration}

## Overview
{notes.get("overview", "")}

## Decisions
{decisions_text}"""


async def create_meeting_task(notes: dict, metadata: dict):
    """
    Creates one main task in the Huddle Meetings list.
    Each key_point becomes a subtask inside the main task.
    """
    started = metadata.get("started_at", "")
    start_dt = _parse_datetime(started)

    date_str = start_dt.strftime("%d %b %Y")
    time_str = start_dt.strftime("%I:%M %p")
    title = notes.get("meeting_title", "Huddle Meeting")

    task_name = f"{date_str}, {time_str} — {title}"

    # Step 1 — Create main task
    payload = {
        "name": task_name,
        "markdown_description": _build_task_description(notes, metadata),
        "tags": ["huddle-notes", "auto-generated"]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{BASE_URL}/list/{CLICKUP_LIST_ID}/task",
            json=payload,
            headers=HEADERS
        )
        response.raise_for_status()

    task_id = response.json()["id"]
    print(f"[ClickUp] Main task created: {task_name} (id: {task_id})")

    # Step 2 — Create subtask for each key discussion point
    for point in notes.get("key_points", []):
        subtask_payload = {
            "name": point.get("title", "Discussion point"),
            "markdown_description": point.get("detail", ""),
            "parent": task_id
        }

        async with httpx.AsyncClient(timeout=30) as client:
            sub_response = await client.post(
                f"{BASE_URL}/list/{CLICKUP_LIST_ID}/task",
                json=subtask_payload,
                headers=HEADERS
            )
            sub_response.raise_for_status()

        print(f"[ClickUp] Subtask created: {point.get('title')}")
