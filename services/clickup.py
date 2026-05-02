import httpx
import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

CLICKUP_API_KEY       = os.getenv("CLICKUP_API_KEY")
CLICKUP_LIST_ID       = os.getenv("CLICKUP_LIST_ID")        # Huddle Meetings list — where notes are posted
CLICKUP_BACKLOG_LIST_ID = os.getenv("CLICKUP_BACKLOG_LIST_ID") # Backlog list — where work tasks live
CLICKUP_TEAM_ID       = "9017847781"
BASE_URL         = "https://api.clickup.com/api/v2"
DATE_FIELD_ID    = "00a57e74-5b15-42da-8e6d-7163977c66ce"  # custom "Date" field
MEMBERS_TTL_SECS = 60    # re-fetch ClickUp members every 1 minute
BACKLOG_TTL_SECS = 300   # re-fetch backlog tasks every 5 minutes

HEADERS = {
    "Authorization": CLICKUP_API_KEY,
    "Content-Type": "application/json"
}

# Cache of ClickUp members: {username_lower: (id, display_name)}
# Refreshed every hour so new team members are picked up automatically
_members_cache: dict = {}
_members_by_id: dict = {}          # {clickup_user_id: {"name": ..., "email": ...}}
_members_fetched_at: datetime | None = None

# Cache of all Backlog tasks — refreshed every 5 minutes
_backlog_cache: list = []
_backlog_fetched_at: float = 0


async def _load_members():
    """
    Fetch list members from ClickUp API.
    Refreshes every hour — new members added to ClickUp are picked up automatically.
    """
    global _members_cache, _members_by_id, _members_fetched_at

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
    _members_by_id = {}
    for m in r.json().get("members", []):
        name  = m.get("username", "").strip()
        uid   = m.get("id")
        email = m.get("email", "")
        if name and uid:
            _members_cache[name.lower()] = (uid, name)
            _members_by_id[uid] = {"name": name, "email": email}

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
    """Returns IST datetime (UTC+5:30) — for display purposes only."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt + timedelta(hours=5, minutes=30)
    except Exception:
        return datetime.now()


def _to_epoch_ms(iso_str: str) -> int:
    """Converts UTC ISO string to Unix epoch milliseconds — for ClickUp API."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(datetime.now().timestamp() * 1000)


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

    # Parse meeting start time → Unix ms for ClickUp (UTC epoch, no IST offset)
    started  = metadata.get("started_at", "")
    start_ms = _to_epoch_ms(started)

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


async def search_relevant_tasks(keywords: list[str]) -> list[dict]:
    """
    Searches ClickUp workspace for tasks matching each keyword (parallel).
    Deduplicates by task ID. Returns flat list of {id, name, status, list}.
    Scales to any number of tasks — ClickUp's search handles the heavy lifting.
    """
    import asyncio

    async def search_one(keyword: str) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{BASE_URL}/team/{CLICKUP_TEAM_ID}/task",
                    headers=HEADERS,
                    params={
                        "search":         keyword,
                        "include_closed": "false",
                        "list_ids[]":     CLICKUP_BACKLOG_LIST_ID,
                        "page":           0
                    }
                )
                r.raise_for_status()
            return r.json().get("tasks", [])
        except Exception as e:
            print(f"[ClickUp Search] '{keyword}' failed: {e}")
            return []

    # Run all keyword searches in parallel
    results = await asyncio.gather(*[search_one(kw) for kw in keywords])

    # Deduplicate by task ID
    seen = set()
    tasks = []
    for batch in results:
        for t in batch:
            tid = t.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                tasks.append({
                    "id":     tid,
                    "name":   t.get("name", ""),
                    "status": t.get("status", {}).get("status", ""),
                    "list":   t.get("list", {}).get("name", ""),
                })

    print(f"[ClickUp Search] {len(keywords)} keywords -> {len(tasks)} unique tasks found")
    return tasks


async def match_and_get_emails(participant_names: list[str]) -> list[tuple]:
    """
    Matches participant names to ClickUp members and returns their emails.
    Returns list of (clickup_id, display_name, email) for each matched participant.
    Used by slack_notifier to look up Slack user IDs.
    """
    await _load_members()
    matched_ids, _ = _match_participants(participant_names)
    result = []
    for uid in matched_ids:
        info = _members_by_id.get(uid, {})
        result.append((uid, info.get("name", ""), info.get("email", "")))
    return result


async def get_backlog_tasks() -> list[dict]:
    """
    Fetch all open tasks from the Backlog list (paginated).
    Used to populate the modal dropdown when user clicks Change / Pick a Task.
    """
    tasks = []
    page  = 0
    while True:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{BASE_URL}/list/{CLICKUP_BACKLOG_LIST_ID}/task",
                headers=HEADERS,
                params={"include_closed": "false", "subtasks": "true", "page": page}
            )
            r.raise_for_status()
        batch = r.json().get("tasks", [])
        if not batch:
            break
        for t in batch:
            parent_raw = t.get("parent")
            if isinstance(parent_raw, dict):
                parent_id = str(parent_raw.get("id") or "")
            elif parent_raw:
                parent_id = str(parent_raw)
            else:
                parent_id = ""
            assignees = t.get("assignees", [])
            assignee_names = ", ".join(
                a.get("username", "").split()[0]   # first name only
                for a in assignees if a.get("username")
            )
            tasks.append({
                "id":        t.get("id"),
                "name":      t.get("name", ""),
                "status":    t.get("status", {}).get("status", ""),
                "assignees": assignee_names,
                "parent_id": parent_id,
                "is_subtask": bool(parent_id)
            })
        if len(batch) < 100:   # ClickUp returns max 100 per page
            break
        page += 1
    return tasks


async def get_backlog_tasks_cached() -> list[dict]:
    """
    Returns all Backlog tasks from cache. Refreshes every 5 minutes.
    Used by /webhook/slack-options so search responses are fast.
    """
    global _backlog_cache, _backlog_fetched_at
    if _backlog_fetched_at and (time.time() - _backlog_fetched_at) < BACKLOG_TTL_SECS:
        return _backlog_cache
    _backlog_cache = await get_backlog_tasks()
    _backlog_fetched_at = time.time()
    print(f"[ClickUp] Backlog cache refreshed ({len(_backlog_cache)} tasks)")
    return _backlog_cache


def search_backlog_by_query(query: str, all_tasks: list[dict]) -> list[dict]:
    """
    Filter backlog tasks by query — matches task name OR assignee name (case-insensitive).
    Called client-side from cached task list so it's instant regardless of task count.
    """
    q = query.lower()
    matched = [
        t for t in all_tasks
        if q in t.get("name", "").lower() or q in t.get("assignees", "").lower()
    ]
    # Assigned tasks first within matches
    assigned   = [t for t in matched if t.get("assignees")]
    unassigned = [t for t in matched if not t.get("assignees")]
    return (assigned + unassigned)[:100]


def get_parent_tasks_for_options(query: str, all_tasks: list[dict]) -> list[dict]:
    """
    Return only parent tasks for Slack parent selector.
    Assigned tasks are ranked first, then unassigned.
    """
    q = (query or "").lower().strip()
    parents = [t for t in all_tasks if not t.get("is_subtask")]
    if q:
        parents = [
            t for t in parents
            if q in t.get("name", "").lower() or q in t.get("assignees", "").lower()
        ]
    assigned = [t for t in parents if t.get("assignees")]
    unassigned = [t for t in parents if not t.get("assignees")]
    return (assigned + unassigned)[:100]


def get_targets_for_parent(parent_id: str, query: str, all_tasks: list[dict]) -> tuple[dict | None, list[dict]]:
    """
    For a selected parent task, return:
    - parent task object
    - list of its direct subtasks (optionally filtered by query)
    """
    if not parent_id:
        return None, []

    parent = next((t for t in all_tasks if t.get("id") == parent_id), None)
    if not parent:
        return None, []

    q = (query or "").lower().strip()
    subtasks = [t for t in all_tasks if t.get("parent_id") == parent_id]
    if q:
        subtasks = [
            t for t in subtasks
            if q in t.get("name", "").lower() or q in t.get("assignees", "").lower()
        ]

    assigned = [t for t in subtasks if t.get("assignees")]
    unassigned = [t for t in subtasks if not t.get("assignees")]
    return parent, (assigned + unassigned)[:100]


def search_subtasks_global(query: str, all_tasks: list[dict]) -> list[dict]:
    """
    Search all subtasks when the user knows the subtask name but not the parent.
    Returns subtask records with parent metadata used by Slack option rendering.
    """
    q = (query or "").lower().strip()
    if not q:
        return []

    tasks_by_id = {str(t.get("id")): t for t in all_tasks if t.get("id")}
    matches = []

    for task in all_tasks:
        if not task.get("is_subtask"):
            continue
        if q not in task.get("name", "").lower() and q not in task.get("assignees", "").lower():
            continue

        parent_id = str(task.get("parent_id") or "")
        parent = tasks_by_id.get(parent_id)
        matches.append({
            **task,
            "parent_name": parent.get("name", "") if parent else "",
            "parent_assignees": parent.get("assignees", "") if parent else "",
        })

    assigned = [t for t in matches if t.get("assignees")]
    unassigned = [t for t in matches if not t.get("assignees")]
    return (assigned + unassigned)[:100]


async def validate_clickup_api_key(api_key: str) -> tuple[bool, str]:
    """
    Validates a ClickUp API key by calling GET /user.
    Returns (True, display_name) if valid, (False, "") if not.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{BASE_URL}/user",
                headers={"Authorization": api_key, "Content-Type": "application/json"}
            )
        if r.status_code == 200:
            user = r.json().get("user", {})
            name = user.get("username", "") or user.get("email", "Unknown")
            return True, name
        return False, ""
    except Exception:
        return False, ""


CUSTOM_FIELD_BRAND        = "2243c0ae-20c7-4ae5-80e1-9a71567f4013"
CUSTOM_FIELD_PROJECT_TYPE = "1996e5f4-d942-4f43-8f28-6ecf6a3ac52e"
CUSTOM_FIELD_THEME        = "fc9aafb6-d93b-4794-99e2-0672c4ecb10c"
CUSTOM_FIELDS_TTL_SECS    = 60    # re-fetch custom field options every 1 minute

_custom_fields_cache:      dict  = {}
_custom_fields_fetched_at: float = 0

_backlog_members_cache:      list  = []   # [{"id": int, "name": str}]
_backlog_members_fetched_at: float = 0


async def get_backlog_members() -> list[dict]:
    """
    Fetch all members of the Backlog list from ClickUp.
    Cached for 1 hour — auto-refreshes when team changes.
    Returns: [{"id": int, "name": str}]
    """
    global _backlog_members_cache, _backlog_members_fetched_at

    if _backlog_members_fetched_at and (time.time() - _backlog_members_fetched_at) < MEMBERS_TTL_SECS:
        return _backlog_members_cache

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{BASE_URL}/list/{CLICKUP_BACKLOG_LIST_ID}/member",
                headers=HEADERS
            )
            r.raise_for_status()
        _backlog_members_cache = [
            {"id": m["id"], "name": m.get("username", "Unknown")}
            for m in r.json().get("members", [])
            if m.get("id") and m.get("username")
        ]
        _backlog_members_fetched_at = time.time()
        print(f"[ClickUp] Backlog members refreshed ({len(_backlog_members_cache)})")
    except Exception as e:
        print(f"[ClickUp] get_backlog_members failed: {e} — using cached")

    return _backlog_members_cache


async def get_backlog_custom_fields() -> dict:
    """
    Fetch Brand, Project Type, and Theme dropdown options from ClickUp.
    Cached for 1 hour — auto-refreshes if options change in ClickUp.
    Returns: {field_id: [{"id": option_id, "name": option_name}]}
    """
    global _custom_fields_cache, _custom_fields_fetched_at

    if _custom_fields_fetched_at and (time.time() - _custom_fields_fetched_at) < CUSTOM_FIELDS_TTL_SECS:
        return _custom_fields_cache

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{BASE_URL}/list/{CLICKUP_BACKLOG_LIST_ID}/field",
                headers=HEADERS
            )
            r.raise_for_status()
        result = {}
        for field in r.json().get("fields", []):
            fid = field.get("id")
            if fid in (CUSTOM_FIELD_BRAND, CUSTOM_FIELD_PROJECT_TYPE, CUSTOM_FIELD_THEME):
                options = field.get("type_config", {}).get("options", [])
                result[fid] = [{"id": o["id"], "name": o["name"]} for o in options]
        _custom_fields_cache     = result
        _custom_fields_fetched_at = time.time()
        print(f"[ClickUp] Custom fields refreshed from API")
    except Exception as e:
        print(f"[ClickUp] get_backlog_custom_fields failed: {e} — using cached")

    return _custom_fields_cache


async def create_backlog_task(
    name: str,
    description: str,
    due_date: str = "",
    api_key: str = None,
    assignees: list = None,
    priority: int = None,
    brand_option_id: str = None,
    project_type_option_id: str = None,
    theme_option_id: str = None
) -> dict:
    """
    Creates a new task in the Backlog list with optional assignees, priority,
    and custom fields (Brand, Project Type, Theme).
    """
    global _backlog_fetched_at

    headers = {
        "Authorization": api_key or CLICKUP_API_KEY,
        "Content-Type":  "application/json"
    }
    payload: dict = {
        "name":                 name,
        "markdown_description": description
    }
    if due_date:
        try:
            dt = datetime.strptime(due_date.strip(), "%Y-%m-%d")
            payload["due_date"] = int(dt.timestamp() * 1000)
        except Exception:
            pass
    if assignees:
        payload["assignees"] = assignees
    if priority:
        payload["priority"] = priority

    custom_fields = []
    if brand_option_id:
        custom_fields.append({"id": CUSTOM_FIELD_BRAND, "value": brand_option_id})
    if project_type_option_id:
        custom_fields.append({"id": CUSTOM_FIELD_PROJECT_TYPE, "value": project_type_option_id})
    if theme_option_id:
        custom_fields.append({"id": CUSTOM_FIELD_THEME, "value": theme_option_id})
    if custom_fields:
        payload["custom_fields"] = custom_fields

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{BASE_URL}/list/{CLICKUP_BACKLOG_LIST_ID}/task",
            headers=headers,
            json=payload
        )
        r.raise_for_status()
    result = r.json()
    _backlog_fetched_at = 0  # invalidate cache so new task appears immediately
    print(f"[ClickUp] New Backlog task created: '{name}' (id: {result.get('id')})")
    return result


async def comment_already_exists(task_id: str, action_text: str) -> bool:
    """
    Check if a comment containing this action point already exists on the task.
    Used to prevent duplicate posts when multiple participants confirm the same point.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{BASE_URL}/task/{task_id}/comment",
            headers=HEADERS
        )
        r.raise_for_status()
    for comment in r.json().get("comments", []):
        existing = comment.get("comment_text", "")
        if action_text.strip() and action_text.strip() in existing:
            return True
    return False


async def post_task_comment(task_id: str, comment_text: str, api_key: str = None):
    """
    Posts a comment to a ClickUp task's Activity section.
    Uses the provided api_key so the comment shows under that user's name in ClickUp.
    Falls back to default CLICKUP_API_KEY if no key provided.
    """
    headers = {
        "Authorization": api_key or CLICKUP_API_KEY,
        "Content-Type":  "application/json"
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{BASE_URL}/task/{task_id}/comment",
            headers=headers,
            json={"comment_text": comment_text, "notify_all": False}
        )
        r.raise_for_status()
    print(f"[ClickUp] Comment posted to task {task_id}")
