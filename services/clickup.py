import httpx
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

CLICKUP_API_KEY  = os.getenv("CLICKUP_API_KEY")
CLICKUP_LIST_ID  = os.getenv("CLICKUP_LIST_ID")
BASE_URL         = "https://api.clickup.com/api/v2"
DATE_FIELD_ID    = "00a57e74-5b15-42da-8e6d-7163977c66ce"  # custom "Date" field
MEMBERS_TTL_SECS = 3600  # re-fetch ClickUp members every 1 hour

HEADERS = {
    "Authorization": CLICKUP_API_KEY,
    "Content-Type": "application/json"
}

# Cache of ClickUp members: {username_lower: (id, display_name)}
# Refreshed every hour so new team members are picked up automatically
_members_cache: dict = {}
_members_fetched_at: datetime | None = None


async def _load_members():
    """
    Fetch list members from ClickUp API.
    Refreshes every hour — new members added to ClickUp are picked up automatically.
    """
    global _members_cache, _members_fetched_at

    now = datetime.now(timezone.utc)
    if _members_fetched_at and (now - _members_fetched_at).total_seconds() < MEMBERS_TTL_SECS:
        return  # cache still fresh

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{BASE_URL}/list/{CLICKUP_LIST_ID}/member",
            headers=HEADERS
        )
        r.raise_for_status()

    _members_cache = {}
    for m in r.json().get("members", []):
        name = m.get("username", "").strip()
        uid  = m.get("id")
        if name and uid:
            _members_cache[name.lower()] = (uid, name)

    _members_fetched_at = now
    print(f"[ClickUp] Members refreshed ({len(_members_cache)}): {[v[1] for v in _members_cache.values()]}")


def _match_participants(participants: list) -> tuple[list[int], list[str]]:
    """
    Match Recall.ai participant names → ClickUp user IDs.
    Returns (matched_ids, unmatched_names).
    Matching order: exact → first name → last name → any word.
    """
    matched_ids   = []
    unmatched     = []
    seen_ids      = set()

    for participant in participants:
        p = participant.strip()
        if not p:
            continue

        p_lower  = p.lower()
        p_words  = p_lower.split()
        found_id = None
        found_name = None

        # 1. Exact match
        if p_lower in _members_cache:
            found_id, found_name = _members_cache[p_lower]

        # 2. First name match
        if not found_id and p_words:
            for name_lower, (uid, display) in _members_cache.items():
                if name_lower.split()[0] == p_words[0]:
                    found_id, found_name = uid, display
                    break

        # 3. Last name match
        if not found_id and len(p_words) > 1:
            for name_lower, (uid, display) in _members_cache.items():
                name_parts = name_lower.split()
                if len(name_parts) > 1 and name_parts[-1] == p_words[-1]:
                    found_id, found_name = uid, display
                    break

        # 4. Any word overlap
        if not found_id:
            for name_lower, (uid, display) in _members_cache.items():
                if any(w in name_lower for w in p_words):
                    found_id, found_name = uid, display
                    break

        if found_id and found_id not in seen_ids:
            matched_ids.append(found_id)
            seen_ids.add(found_id)
            print(f"[ClickUp] Matched '{p}' -> '{found_name}' (id: {found_id})")
        else:
            unmatched.append(p)
            print(f"[ClickUp] No ClickUp match for participant '{p}' — will note in description")

    return matched_ids, unmatched


def _parse_datetime(iso_str: str):
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt + timedelta(hours=5, minutes=30)
    except Exception:
        return datetime.now()


def _build_task_description(notes: dict, metadata: dict, unmatched_participants: list) -> str:
    participants  = metadata.get("participants", [])
    duration_min  = metadata.get("duration_minutes", 0)
    started       = metadata.get("started_at", "")
    ended         = metadata.get("ended_at", "")

    start_dt   = _parse_datetime(started)
    start_time = start_dt.strftime("%I:%M %p")
    time_str   = f"{start_time} – {_parse_datetime(ended).strftime('%I:%M %p')} IST" if ended else f"{start_time} IST"

    def fmt_participants(plist):
        parts = []
        for p in plist:
            if isinstance(p, dict):
                parts.append(p.get("name") or p.get("display_name") or "Unknown")
            else:
                parts.append(str(p))
        return ", ".join(filter(None, parts)) or "Unknown"

    lines = []

    # Header info
    lines.append(f"👥 **Participants:** {fmt_participants(participants)}")
    if unmatched_participants:
        lines.append(f"⚠️ **Not in ClickUp:** {', '.join(unmatched_participants)}")
    lines.append(f"⏱️ **Duration:** {duration_min} min | {time_str}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Meeting Purpose
    purpose = notes.get("meeting_purpose", "")
    if purpose:
        lines.append("## 🎯 Meeting Purpose")
        lines.append(purpose)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Overview
    overview = notes.get("overview", "")
    if overview:
        lines.append("## 📋 Overview")
        lines.append(overview)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Key Takeaways
    takeaways = notes.get("key_takeaways", [])
    if takeaways:
        lines.append("## ⚡ Key Takeaways")
        for t in takeaways:
            lines.append(f"- {t}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Topics Discussed
    topics = notes.get("topics", [])
    if topics:
        lines.append("## 🗂️ Topics Discussed")
        lines.append("")
        for i, topic in enumerate(topics, 1):
            lines.append(f"### {i}. {topic.get('title', 'Topic')}")
            detail = topic.get("detail", "")
            if isinstance(detail, list):
                lines.extend(detail)
            else:
                lines.append(detail)
            lines.append("")
        lines.append("---")
        lines.append("")

    # Decisions Made
    decisions = notes.get("decisions", [])
    if decisions:
        lines.append("## ✅ Decisions Made")
        for d in decisions:
            if isinstance(d, dict):
                decision  = d.get("decision", "")
                rationale = d.get("rationale", "")
                if rationale:
                    lines.append(f"- **{decision}**  \n  _Why: {rationale}_")
                else:
                    lines.append(f"- {decision}")
            else:
                lines.append(f"- {d}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Implementation Plan
    impl = notes.get("implementation_plan", [])
    if impl:
        lines.append("## 🔧 Implementation Plan")
        for i, step in enumerate(impl, 1):
            lines.append(f"{i}. {step}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Next Steps
    next_steps = notes.get("next_steps", [])
    if next_steps:
        lines.append("## 📌 Next Steps")
        for ns in next_steps:
            if isinstance(ns, dict):
                task     = ns.get("task", "")
                owner    = ns.get("owner", "")
                deadline = ns.get("deadline", "")
                suffix   = ""
                if owner:
                    suffix += f" — **{owner}**"
                if deadline:
                    suffix += f" _(by {deadline})_"
                lines.append(f"- [ ] {task}{suffix}")
            else:
                lines.append(f"- [ ] {ns}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Blockers & Risks
    blockers = notes.get("blockers", [])
    if blockers:
        lines.append("## 🚧 Blockers & Risks")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    return "\n".join(lines)


async def create_meeting_task(notes: dict, metadata: dict):
    """
    Creates one main ClickUp task with:
    - Title: just the meeting title (no date prefix)
    - start_date + custom Date field = meeting start time
    - assignees = all matched participants
    - Full structured description
    - One [TOPIC] subtask per topic
    """
    # Load members if not cached
    await _load_members()

    # Match participants to ClickUp users
    raw_participants = metadata.get("participants", [])
    participant_names = []
    for p in raw_participants:
        if isinstance(p, dict):
            participant_names.append(p.get("name") or p.get("display_name") or "")
        else:
            participant_names.append(str(p))
    participant_names = [n for n in participant_names if n]

    matched_ids, unmatched = _match_participants(participant_names)

    # Parse meeting start time → Unix ms for ClickUp
    started    = metadata.get("started_at", "")
    start_dt   = _parse_datetime(started)
    start_ms   = int(start_dt.timestamp() * 1000)

    # Task name = just the meeting title
    title     = notes.get("meeting_title", "Huddle Meeting")
    task_name = title

    payload = {
        "name": task_name,
        "markdown_description": _build_task_description(notes, metadata, unmatched),
        "assignees": matched_ids,
        "start_date": start_ms,
        "custom_fields": [
            {
                "id": DATE_FIELD_ID,
                "value": start_ms
            }
        ],
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
    print(f"[ClickUp] Task created: '{task_name}' (id: {task_id})")
    print(f"[ClickUp] Assigned: {matched_ids} | Unmatched: {unmatched}")

    # One subtask per topic (no assignees, no [TOPIC] prefix)
    for topic in notes.get("topics", []):
        detail = topic.get("detail", "")
        if isinstance(detail, list):
            detail = "\n".join(detail)
        subtask_payload = {
            "name": topic.get("title", "Discussion point"),
            "markdown_description": detail,
            "parent": task_id
        }
        async with httpx.AsyncClient(timeout=30) as client:
            sub = await client.post(
                f"{BASE_URL}/list/{CLICKUP_LIST_ID}/task",
                json=subtask_payload,
                headers=HEADERS
            )
            sub.raise_for_status()
        print(f"[ClickUp] Subtask: {topic.get('title')}")

