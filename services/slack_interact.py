import asyncio
import httpx
import json
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from upstash_redis.asyncio import Redis

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
    "Content-Type": "application/json"
}

redis_client = Redis(
    url=os.getenv("UPSTASH_REDIS_URL"),
    token=os.getenv("UPSTASH_REDIS_TOKEN")
)

ACTION_STATE_TTL = 7 * 24 * 60 * 60   # 7 days
PARENT_PICK_TTL  = 60 * 60            # 1 hour


# ── Redis helpers ─────────────────────────────────────────────────────────────

async def _get_user_clickup_key(slack_user_id: str) -> str | None:
    try:
        return await redis_client.get(f"clickup_key:{slack_user_id}")
    except Exception:
        return None


async def _get_step_data(meeting_id: str, step_index: int) -> dict | None:
    """
    Look up a single step's full data from the stored meeting session.
    Returns dict with: task_text, task_context, clickup_task_id,
                       clickup_task_name, deadline, participants_str.
    Returns None if session missing or step index out of range.
    """
    if not meeting_id or not step_index:
        return None
    try:
        raw = await redis_client.get(f"dm_session:{meeting_id}")
        if not raw:
            return None
        session = json.loads(raw)
        steps   = session.get("notes", {}).get("next_steps", [])
        if step_index < 1 or step_index > len(steps):
            return None
        step             = steps[step_index - 1]   # si is 1-indexed
        participants_str = session.get("metadata", {}).get("participants_str", "")
        return {
            "task_text":         step.get("task", ""),
            "task_context":      step.get("context", ""),
            "clickup_task_id":   step.get("clickup_task_id"),
            "clickup_task_name": step.get("clickup_task_name") or "",
            "deadline":          step.get("deadline") or "",
            "participants_str":  participants_str
        }
    except Exception as e:
        print(f"[Slack Interact] _get_step_data({meeting_id}, {step_index}) failed: {e}")
        return None


async def _get_slack_display_name(slack_user_id: str) -> str:
    """Return the first name of a Slack user ID. Falls back to 'Someone'."""
    from services.slack_notifier import _slack_users_cache, _load_slack_users
    await _load_slack_users()
    for u in _slack_users_cache:
        if u.get("id") == slack_user_id:
            name = u.get("real_name") or u.get("display_name") or ""
            return name.split()[0] if name else "Someone"
    return "Someone"


async def _get_all_action_states(meeting_id: str, num_steps: int) -> dict:
    """
    Fetch all confirmed action states for a meeting in parallel.
    Returns {str(step_index): {"confirmed_by_name": ..., "task_name": ...}}
    """
    if not meeting_id or num_steps < 1:
        return {}
    try:
        keys    = [f"action_state:{meeting_id}:{i}" for i in range(1, num_steps + 1)]
        results = await asyncio.gather(
            *[redis_client.get(k) for k in keys],
            return_exceptions=True
        )
        states = {}
        for i, result in enumerate(results, 1):
            if result and not isinstance(result, Exception):
                states[str(i)] = json.loads(result)
        return states
    except Exception as e:
        print(f"[Slack Interact] _get_all_action_states failed: {e}")
        return {}


async def _update_all_dm_threads(meeting_id: str, action_states: dict, session: dict):
    """
    Rebuild blocks and call chat.update on every participant's DM message.
    Silent — no new notification fired, just the existing message is updated.
    """
    from services.slack_notifier import _build_dm_blocks

    try:
        participant_ids = await redis_client.lrange(f"dm_participants:{meeting_id}", 0, -1)
    except Exception as e:
        print(f"[Slack Interact] Failed to fetch participants list for {meeting_id}: {e}")
        return

    if not participant_ids:
        print(f"[Slack Interact] No DM participants found for meeting {meeting_id}")
        return

    notes_part    = session.get("notes", {})
    metadata_part = session.get("metadata", {})
    blocks        = _build_dm_blocks(
        notes_part, metadata_part,
        meeting_id=meeting_id,
        action_states=action_states
    )
    fallback_text = "Meeting notes updated."

    async def _update_one(user_id: str):
        try:
            dm_raw = await redis_client.get(f"dm_msg:{meeting_id}:{user_id}")
            if not dm_raw:
                return
            dm_info = json.loads(dm_raw)
            ts      = dm_info.get("ts")
            channel = dm_info.get("channel")
            if not ts or not channel:
                return
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://slack.com/api/chat.update",
                    headers=HEADERS,
                    json={"channel": channel, "ts": ts, "blocks": blocks, "text": fallback_text}
                )
            result = resp.json()
            if not result.get("ok"):
                print(f"[Slack Interact] chat.update failed for {user_id}: {result.get('error')}")
            else:
                print(f"[Slack Interact] DM thread updated for {user_id}")
        except Exception as e:
            print(f"[Slack Interact] Failed to update DM for {user_id}: {e}")

    await asyncio.gather(*[_update_one(uid) for uid in participant_ids], return_exceptions=True)


async def _store_and_update_dm_state(
    meeting_id: str,
    step_index: int,
    actor_name: str,
    actor_id: str,
    task_name: str
):
    """
    Store confirmed state for one action item, then silently update all participants' DM threads.
    Called after every successful confirm / pick / create action.
    """
    if not meeting_id or not step_index:
        return

    try:
        state = {
            "confirmed_by_name": actor_name,
            "confirmed_by_id":   actor_id,
            "task_name":         task_name
        }
        await redis_client.set(
            f"action_state:{meeting_id}:{step_index}",
            json.dumps(state),
            ex=ACTION_STATE_TTL
        )
    except Exception as e:
        print(f"[Slack Interact] Failed to store action state: {e}")
        return

    try:
        raw = await redis_client.get(f"dm_session:{meeting_id}")
        if not raw:
            print(f"[Slack Interact] No session found for meeting {meeting_id} — skipping thread update")
            return
        session   = json.loads(raw)
        num_steps = len(session.get("notes", {}).get("next_steps", []))
        states    = await _get_all_action_states(meeting_id, num_steps)
        await _update_all_dm_threads(meeting_id, states, session)
    except Exception as e:
        print(f"[Slack Interact] _store_and_update_dm_state failed during thread update: {e}")


# ── Main interaction router ───────────────────────────────────────────────────

async def handle_interaction(payload: dict):
    """
    Main router for all Slack interactions.
    Called in a background task — Slack already got its 200 OK.
    """
    interaction_type = payload.get("type")

    if interaction_type == "block_actions":
        actions = payload.get("actions", [])
        if not actions:
            return
        action        = actions[0]
        action_id     = action.get("action_id", "")
        response_url  = payload.get("response_url", "")
        trigger_id    = payload.get("trigger_id", "")
        slack_user_id = payload.get("user", {}).get("id", "")

        if action_id == "selected_parent":
            await _handle_parent_selected(payload, action)
        elif action_id.startswith("confirm_"):
            await _handle_confirm(action, response_url, slack_user_id, trigger_id)
        elif action_id.startswith("change_") or action_id.startswith("pick_"):
            await _handle_change_or_pick(action, trigger_id, response_url, slack_user_id)
        elif action_id.startswith("create_"):
            await _handle_create(action, trigger_id, response_url, slack_user_id)
        elif action_id.startswith("edit_"):
            await _handle_edit(action, slack_user_id)

    elif interaction_type == "view_submission":
        callback_id = payload.get("view", {}).get("callback_id", "")
        if callback_id == "api_key_modal":
            await _handle_api_key_submit(payload)
        elif callback_id == "create_task_modal":
            await _handle_create_task_submit(payload)
        else:
            await _handle_modal_submit(payload)


# ── Block-action handlers ─────────────────────────────────────────────────────

async def _handle_confirm(action: dict, response_url: str, slack_user_id: str, trigger_id: str):
    """User clicked Confirm — post comment to the suggested ClickUp task."""
    from services.clickup import post_task_comment, comment_already_exists

    value      = json.loads(action.get("value", "{}"))
    meeting_id = value.get("mid", "")
    step_index = value.get("si", 0)

    # New format: look up step data from Redis
    if meeting_id and step_index:
        step_data = await _get_step_data(meeting_id, step_index)
        if step_data is None:
            await _respond_url(response_url, ":x: Meeting data not found — this link may have expired.")
            return
        task_id      = step_data["clickup_task_id"]
        task_name    = step_data["clickup_task_name"]
        task_text    = step_data["task_text"]
        task_context = step_data["task_context"]
        meta         = {"participants_str": step_data["participants_str"]}
    else:
        # Backward compat: full data embedded in button value
        task_id      = value.get("clickup_task_id")
        task_name    = value.get("clickup_task_name", "")
        task_text    = value.get("task_text", "")
        task_context = value.get("task_context", "")
        meta         = value.get("meta", {})

    if not task_id:
        await _respond_url(response_url, ":x: No ClickUp task ID found — could not post comment.")
        return

    api_key = await _get_user_clickup_key(slack_user_id)
    if not api_key:
        await _open_api_key_modal(trigger_id, "confirm", value, response_url)
        return

    if await comment_already_exists(task_id, task_text):
        await _respond_url(response_url, f":information_source: Already posted to *{task_name}* by someone else.")
        return

    comment = _build_comment(task_text, meta, task_context)
    try:
        await post_task_comment(task_id, comment, api_key=api_key)
        await _respond_url(response_url, f":white_check_mark: Posted to *{task_name}*")
        if meeting_id and step_index:
            actor_name = await _get_slack_display_name(slack_user_id)
            await _store_and_update_dm_state(meeting_id, step_index, actor_name, slack_user_id, task_name)
    except Exception as e:
        print(f"[Slack Interact] post_task_comment failed: {e}")
        await _respond_url(response_url, f":x: Failed to post comment: {e}")


async def _handle_change_or_pick(action: dict, trigger_id: str, response_url: str, slack_user_id: str):
    """User clicked Change Task or Pick a Task — check API key, then open modal."""
    value = json.loads(action.get("value", "{}"))

    api_key = await _get_user_clickup_key(slack_user_id)
    if not api_key:
        await _open_api_key_modal(trigger_id, "pick", value, response_url)
        return

    await _open_pick_task_modal(trigger_id, value, response_url)


async def _handle_create(action: dict, trigger_id: str, response_url: str, slack_user_id: str):
    """User clicked Create New Task — check API key, then open modal."""
    value = json.loads(action.get("value", "{}"))

    api_key = await _get_user_clickup_key(slack_user_id)
    if not api_key:
        await _open_api_key_modal(trigger_id, "create", value, response_url)
        return

    await _open_create_task_modal(trigger_id, value, response_url)


async def _handle_edit(action: dict, slack_user_id: str):
    """
    User clicked Edit — clear the confirmed state for this action item
    and restore original buttons across all participants' threads.
    """
    value      = json.loads(action.get("value", "{}"))
    meeting_id = value.get("mid", "")
    step_index = value.get("si", 0)

    if not meeting_id or not step_index:
        print("[Slack Interact] Edit clicked with missing mid/si — ignoring")
        return

    # Clear the action state for this step
    try:
        await redis_client.delete(f"action_state:{meeting_id}:{step_index}")
    except Exception as e:
        print(f"[Slack Interact] Failed to delete action state for edit: {e}")
        return

    # Rebuild with remaining states and update all threads
    try:
        raw = await redis_client.get(f"dm_session:{meeting_id}")
        if not raw:
            return
        session   = json.loads(raw)
        num_steps = len(session.get("notes", {}).get("next_steps", []))
        states    = await _get_all_action_states(meeting_id, num_steps)
        await _update_all_dm_threads(meeting_id, states, session)
    except Exception as e:
        print(f"[Slack Interact] _handle_edit failed: {e}")


async def _handle_parent_selected(payload: dict, action: dict):
    """
    Persist selected parent value for the current modal view.
    Slack options requests for the second selector may omit view.state in some payloads,
    so we keep a short-lived fallback keyed by view_id + user_id.
    """
    view_obj    = payload.get("view", {}) or {}
    view_id     = view_obj.get("id", "")
    view_hash   = view_obj.get("hash", "")
    slack_user  = payload.get("user", {}).get("id", "")
    selected_opt = action.get("selected_option", {}) or {}
    parent_val   = selected_opt.get("value", "")
    parent_text  = (selected_opt.get("text", {}) or {}).get("text", "")

    if not view_id or not slack_user or not parent_val:
        return

    try:
        await redis_client.set(
            f"parent_pick:{view_id}:{slack_user}",
            parent_val,
            ex=PARENT_PICK_TTL
        )
        print(f"[Slack Interact] parent selected cached view={view_id} user={slack_user} val={parent_val}")
    except Exception as e:
        print(f"[Slack Interact] Failed to cache parent selection: {e}")

    # Rebuild modal with a new target selector identity tied to parent ID.
    # This forces Slack to fetch fresh options and avoids stale subtask lists.
    try:
        private_metadata = json.loads(view_obj.get("private_metadata", "{}"))
        private_metadata["selected_parent_value"] = parent_val
        if parent_text:
            private_metadata["selected_parent_text"] = parent_text

        display_text = private_metadata.get("display_text", "(see meeting notes)")
        updated_view = _build_pick_task_modal_view(
            private_metadata=private_metadata,
            display_text=display_text,
            selected_parent_value=parent_val,
            selected_parent_text=parent_text
        )

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://slack.com/api/views.update",
                headers=HEADERS,
                json={
                    "view_id": view_id,
                    "hash": view_hash,
                    "view": updated_view
                }
            )
        resp = r.json()
        if not resp.get("ok"):
            print(f"[Slack Interact] views.update failed after parent select: {resp.get('error')}")
    except Exception as e:
        print(f"[Slack Interact] Failed to refresh modal after parent select: {e}")


# ── Modal openers ─────────────────────────────────────────────────────────────

async def _open_api_key_modal(trigger_id: str, pending_action: str, action_value: dict, response_url: str):
    """
    Opens a modal asking the user for their ClickUp API key.
    Stores pending action in private_metadata so it can be re-executed after key is saved.
    """
    private_metadata = json.dumps({
        "pending_action": pending_action,
        "action_value":   action_value,
        "response_url":   response_url
    })

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*How to get your ClickUp API key:*\n"
                    "1. Open ClickUp\n"
                    "2. Right-click your *avatar* (top-right corner)\n"
                    "3. Click *Settings*\n"
                    "4. Click *ClickUp API*\n"
                    "5. Generate your token and paste it below"
                )
            }
        },
        {"type": "divider"},
        {
            "type":     "input",
            "block_id": "api_key_block",
            "element": {
                "type":        "plain_text_input",
                "action_id":   "key_value",
                "placeholder": {"type": "plain_text", "text": "pk_xxxxxxxxxx..."}
            },
            "label": {"type": "plain_text", "text": "Paste your API Key"}
        }
    ]

    modal = {
        "type":             "modal",
        "callback_id":      "api_key_modal",
        "title":            {"type": "plain_text", "text": "Connect ClickUp"},
        "submit":           {"type": "plain_text", "text": "Save & Continue"},
        "close":            {"type": "plain_text", "text": "Cancel"},
        "private_metadata": private_metadata,
        "blocks":           blocks
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://slack.com/api/views.open",
            headers=HEADERS,
            json={"trigger_id": trigger_id, "view": modal}
        )
    resp = r.json()
    if not resp.get("ok"):
        print(f"[Slack Interact] api_key modal views.open failed: {resp.get('error')}")


async def _open_pick_task_modal(trigger_id: str, action_value: dict, response_url: str):
    """
    Opens a Slack modal with an external_select dropdown.
    Options load dynamically from /webhook/slack-options.
    """
    meeting_id = action_value.get("mid", "")
    step_index = action_value.get("si", 0)

    # Look up task_text for modal display (new format has no task_text in button value)
    display_text = action_value.get("task_text", "")
    if meeting_id and step_index and not display_text:
        step_data    = await _get_step_data(meeting_id, step_index)
        display_text = step_data["task_text"] if step_data else "(see meeting notes)"

    private_metadata = {
        **action_value,
        "response_url": response_url,
        "display_text": display_text
    }
    modal = _build_pick_task_modal_view(private_metadata=private_metadata, display_text=display_text)

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://slack.com/api/views.open",
            headers=HEADERS,
            json={"trigger_id": trigger_id, "view": modal}
        )
    resp = r.json()
    if not resp.get("ok"):
        print(f"[Slack Interact] pick modal views.open failed: {resp.get('error')}")


def _build_pick_task_modal_view(
    private_metadata: dict,
    display_text: str,
    selected_parent_value: str = "",
    selected_parent_text: str = ""
) -> dict:
    """
    Build pick-task modal view.
    target_select/action ids are versioned by parent id to force Slack to refetch
    external_select options after parent changes.
    """
    parent_id = ""
    if isinstance(selected_parent_value, str) and selected_parent_value.startswith("p:"):
        parent_id = selected_parent_value[2:]

    target_block_id = f"target_select_{parent_id}" if parent_id else "target_select"
    target_action_id = f"selected_target__{parent_id}" if parent_id else "selected_target"

    parent_element = {
        "type":             "external_select",
        "placeholder":      {"type": "plain_text", "text": "Search parent tasks..."},
        "action_id":        "selected_parent",
        "min_query_length": 0
    }
    if selected_parent_value and selected_parent_text:
        parent_element["initial_option"] = {
            "text": {"type": "plain_text", "text": selected_parent_text[:75]},
            "value": selected_parent_value
        }

    return {
        "type":             "modal",
        "callback_id":      "pick_task_modal",
        "title":            {"type": "plain_text", "text": "Pick a Task"},
        "submit":           {"type": "plain_text", "text": "Post Comment"},
        "close":            {"type": "plain_text", "text": "Cancel"},
        "private_metadata": json.dumps(private_metadata),
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Action Point:*\n{display_text}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "1) Pick a parent task\n"
                        "2) Optional: choose subtask (leave empty to post on parent)"
                    )
                }
            },
            {"type": "divider"},
            {
                "type":     "input",
                "block_id": "parent_select",
                "dispatch_action": True,
                "element": parent_element,
                "label": {"type": "plain_text", "text": "Parent Task"}
            },
            {
                "type":     "input",
                "block_id": target_block_id,
                "optional": True,
                "element": {
                    "type":             "external_select",
                    "placeholder":      {"type": "plain_text", "text": "Choose subtask (optional)..."},
                    "action_id":        target_action_id,
                    "min_query_length": 0
                },
                "label": {"type": "plain_text", "text": "Choose Subtask (Optional)"}
            }
        ]
    }


async def _open_create_task_modal(trigger_id: str, action_value: dict, response_url: str):
    """
    Opens a modal pre-filled with the action point as task name.
    Required: Brand, Project Type, Theme (fetched live from ClickUp, 1hr cache).
    Optional: Due Date, Assignee, Priority.
    """
    from services.clickup import get_backlog_custom_fields, get_backlog_members, CUSTOM_FIELD_BRAND, CUSTOM_FIELD_PROJECT_TYPE, CUSTOM_FIELD_THEME

    meeting_id = action_value.get("mid", "")
    step_index = action_value.get("si", 0)

    # Look up step data for pre-fill values
    if meeting_id and step_index:
        step_data = await _get_step_data(meeting_id, step_index)
        task_text = step_data["task_text"] if step_data else ""
        deadline  = step_data["deadline"]  if step_data else ""
    else:
        task_text = action_value.get("task_text", "")
        deadline  = action_value.get("deadline", "")

    private_metadata = json.dumps({
        "mid":          meeting_id,
        "si":           step_index,
        "task_text":    task_text,
        "response_url": response_url,
        # Keep old-format fields for backward compat when mid is absent
        "task_context": action_value.get("task_context", "") if not meeting_id else "",
        "meta":         action_value.get("meta", {}) if not meeting_id else {}
    })

    # Fetch all dynamic data live from ClickUp (1hr cache each)
    cf, members = await asyncio.gather(get_backlog_custom_fields(), get_backlog_members())

    def _to_slack_options(field_id: str) -> list:
        return [
            {"text": {"type": "plain_text", "text": o["name"]}, "value": o["id"]}
            for o in cf.get(field_id, [])
        ]

    brand_options  = _to_slack_options(CUSTOM_FIELD_BRAND)
    pt_options     = _to_slack_options(CUSTOM_FIELD_PROJECT_TYPE)
    theme_options  = _to_slack_options(CUSTOM_FIELD_THEME)

    if not brand_options or not pt_options or not theme_options:
        await _respond_url(response_url, ":x: Could not load ClickUp fields — please try again in a moment.")
        return

    member_options = [
        {"text": {"type": "plain_text", "text": m["name"]}, "value": str(m["id"])}
        for m in members
    ]

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Action Point:*\n{task_text}"}
        },
        {"type": "divider"},
        {
            "type":     "input",
            "block_id": "task_name",
            "element": {
                "type":          "plain_text_input",
                "action_id":     "name_input",
                "initial_value": task_text[:150],
                "placeholder":   {"type": "plain_text", "text": "Task name"}
            },
            "label": {"type": "plain_text", "text": "Task Name"}
        },
        {
            "type":     "input",
            "block_id": "brand",
            "element": {
                "type":        "static_select",
                "action_id":   "brand_input",
                "placeholder": {"type": "plain_text", "text": "Select brand"},
                "options":     brand_options
            },
            "label": {"type": "plain_text", "text": "Brand"}
        },
        {
            "type":     "input",
            "block_id": "project_type",
            "element": {
                "type":        "static_select",
                "action_id":   "project_type_input",
                "placeholder": {"type": "plain_text", "text": "Select project type"},
                "options":     pt_options
            },
            "label": {"type": "plain_text", "text": "Project Type"}
        },
        {
            "type":     "input",
            "block_id": "theme",
            "element": {
                "type":        "static_select",
                "action_id":   "theme_input",
                "placeholder": {"type": "plain_text", "text": "Select theme"},
                "options":     theme_options
            },
            "label": {"type": "plain_text", "text": "Theme"}
        },
        {
            "type":     "input",
            "block_id": "due_date",
            "optional": True,
            "element": {
                "type":          "plain_text_input",
                "action_id":     "due_date_input",
                "initial_value": deadline,
                "placeholder":   {"type": "plain_text", "text": "YYYY-MM-DD  e.g. 2026-03-25"}
            },
            "label": {"type": "plain_text", "text": "Due Date (optional)"}
        },
        {
            "type":     "input",
            "block_id": "assignee",
            "optional": True,
            "element": {
                "type":        "multi_static_select",
                "action_id":   "assignee_input",
                "placeholder": {"type": "plain_text", "text": "Select assignee(s)"},
                "options": member_options or [{"text": {"type": "plain_text", "text": "No members found"}, "value": "0"}]
            },
            "label": {"type": "plain_text", "text": "Assignee (optional)"}
        },
        {
            "type":     "input",
            "block_id": "priority",
            "optional": True,
            "element": {
                "type":        "static_select",
                "action_id":   "priority_input",
                "placeholder": {"type": "plain_text", "text": "Select priority"},
                "options": [
                    {"text": {"type": "plain_text", "text": "Urgent"}, "value": "1"},
                    {"text": {"type": "plain_text", "text": "High"},   "value": "2"},
                    {"text": {"type": "plain_text", "text": "Normal"}, "value": "3"},
                    {"text": {"type": "plain_text", "text": "Low"},    "value": "4"}
                ]
            },
            "label": {"type": "plain_text", "text": "Priority (optional)"}
        }
    ]

    modal = {
        "type":             "modal",
        "callback_id":      "create_task_modal",
        "title":            {"type": "plain_text", "text": "Create New Task"},
        "submit":           {"type": "plain_text", "text": "Create Task"},
        "close":            {"type": "plain_text", "text": "Cancel"},
        "private_metadata": private_metadata,
        "blocks":           blocks
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://slack.com/api/views.open",
            headers=HEADERS,
            json={"trigger_id": trigger_id, "view": modal}
        )
    resp = r.json()
    if not resp.get("ok"):
        print(f"[Slack Interact] create modal views.open failed: {resp.get('error')}")


# ── Modal submission handlers ─────────────────────────────────────────────────

async def _handle_api_key_submit(payload: dict):
    """
    User submitted the Connect ClickUp modal.
    Validates key → saves to Redis → re-executes the original pending action.
    """
    from services.clickup import validate_clickup_api_key, post_task_comment, comment_already_exists

    view             = payload.get("view", {})
    private_metadata = json.loads(view.get("private_metadata", "{}"))
    state            = view.get("state", {}).get("values", {})
    slack_user_id    = payload.get("user", {}).get("id", "")
    trigger_id       = payload.get("trigger_id", "")

    api_key        = (state.get("api_key_block", {}).get("key_value", {}).get("value") or "").strip()
    pending_action = private_metadata.get("pending_action", "")
    action_value   = private_metadata.get("action_value", {})
    response_url   = private_metadata.get("response_url", "")

    valid, clickup_name = await validate_clickup_api_key(api_key)
    if not valid:
        await _respond_url(response_url, ":x: Invalid ClickUp API key — please click the button again and enter the correct key.")
        return

    await redis_client.set(f"clickup_key:{slack_user_id}", api_key)
    print(f"[Slack Interact] ClickUp key saved for Slack user {slack_user_id} ({clickup_name})")

    if pending_action == "confirm":
        meeting_id = action_value.get("mid", "")
        step_index = action_value.get("si", 0)

        # New format: look up from Redis
        if meeting_id and step_index:
            step_data = await _get_step_data(meeting_id, step_index)
            if not step_data:
                await _respond_url(response_url, f":white_check_mark: ClickUp connected as *{clickup_name}*! Meeting data expired — no comment posted.")
                return
            task_id      = step_data["clickup_task_id"]
            task_name    = step_data["clickup_task_name"]
            task_text    = step_data["task_text"]
            task_context = step_data["task_context"]
            meta         = {"participants_str": step_data["participants_str"]}
        else:
            # Backward compat
            task_id      = action_value.get("clickup_task_id")
            task_name    = action_value.get("clickup_task_name", "")
            task_text    = action_value.get("task_text", "")
            task_context = action_value.get("task_context", "")
            meta         = action_value.get("meta", {})

        if not task_id:
            await _respond_url(response_url, f":white_check_mark: ClickUp connected as *{clickup_name}*!")
            return

        if await comment_already_exists(task_id, task_text):
            await _respond_url(response_url, f":information_source: Already posted to *{task_name}*.")
            return

        comment = _build_comment(task_text, meta, task_context)
        try:
            await post_task_comment(task_id, comment, api_key=api_key)
            await _respond_url(response_url, f":white_check_mark: Connected as *{clickup_name}* · Posted to *{task_name}*")
            if meeting_id and step_index:
                actor_name = await _get_slack_display_name(slack_user_id)
                await _store_and_update_dm_state(meeting_id, step_index, actor_name, slack_user_id, task_name)
        except Exception as e:
            await _respond_url(response_url, f":x: Key saved but failed to post: {e}")

    elif pending_action == "pick":
        await _respond_url(response_url, f":white_check_mark: ClickUp connected as *{clickup_name}*! Please click *Change Task / Pick a Task* again.")

    elif pending_action == "create":
        await _respond_url(response_url, f":white_check_mark: ClickUp connected as *{clickup_name}*! Please click *Create New Task* again.")


async def _handle_modal_submit(payload: dict):
    """User submitted the pick-task modal — post comment to the selected ClickUp task."""
    from services.clickup import post_task_comment, comment_already_exists

    view             = payload.get("view", {})
    private_metadata = json.loads(view.get("private_metadata", "{}"))
    state            = view.get("state", {}).get("values", {})
    slack_user_id    = payload.get("user", {}).get("id", "")

    selected_opt_new = {}
    for block_id, actions in state.items():
        if not (isinstance(block_id, str) and block_id.startswith("target_select")):
            continue
        if not isinstance(actions, dict):
            continue
        for action_data in actions.values():
            if not isinstance(action_data, dict):
                continue
            opt = (action_data.get("selected_option", {}) or {})
            if opt:
                selected_opt_new = opt
                break
        if selected_opt_new:
            break

    selected_opt_parent = state.get("parent_select", {}).get("selected_parent", {}).get("selected_option", {}) or {}
    selected_opt_old = state.get("task_select", {}).get("selected_task", {}).get("selected_option", {}) or {}

    selected_opt  = selected_opt_new or selected_opt_parent or selected_opt_old
    selected_val  = selected_opt.get("value", "")
    selected_name = selected_opt.get("text", {}).get("text", "")

    # New value format:
    #   p:<parent_id>            -> post on parent
    #   s:<subtask_id>:<parent>  -> post on subtask
    # Legacy format:
    #   <task_id>
    if isinstance(selected_val, str) and selected_val.startswith("p:"):
        selected_id = selected_val[2:]
    elif isinstance(selected_val, str) and selected_val.startswith("s:"):
        parts = selected_val.split(":")
        selected_id = parts[1] if len(parts) >= 2 else ""
    else:
        selected_id = selected_val

    meeting_id   = private_metadata.get("mid", "")
    step_index   = private_metadata.get("si", 0)
    response_url = private_metadata.get("response_url", "")

    # New format: look up from Redis
    if meeting_id and step_index:
        step_data    = await _get_step_data(meeting_id, step_index)
        task_text    = step_data["task_text"]    if step_data else ""
        task_context = step_data["task_context"] if step_data else ""
        meta         = {"participants_str": step_data["participants_str"]} if step_data else {}
    else:
        # Backward compat
        task_text    = private_metadata.get("task_text", "")
        task_context = private_metadata.get("task_context", "")
        meta         = private_metadata.get("meta", {})

    if not selected_id or selected_id == "none":
        await _respond_url(response_url, ":x: No task selected — comment not posted.")
        return

    if await comment_already_exists(selected_id, task_text):
        await _respond_url(response_url, f":information_source: Already posted to *{selected_name}* by someone else.")
        return

    api_key = await _get_user_clickup_key(slack_user_id)
    comment = _build_comment(task_text, meta, task_context)
    try:
        await post_task_comment(selected_id, comment, api_key=api_key)
        await _respond_url(response_url, f":white_check_mark: Posted to *{selected_name}*")
        if meeting_id and step_index:
            actor_name = await _get_slack_display_name(slack_user_id)
            await _store_and_update_dm_state(meeting_id, step_index, actor_name, slack_user_id, selected_name)
    except Exception as e:
        print(f"[Slack Interact] modal post_task_comment failed: {e}")
        await _respond_url(response_url, f":x: Failed to post comment: {e}")


async def _handle_create_task_submit(payload: dict):
    """User submitted the create-task modal — create task in ClickUp Backlog + post comment."""
    from services.clickup import create_backlog_task, post_task_comment

    view             = payload.get("view", {})
    private_metadata = json.loads(view.get("private_metadata", "{}"))
    state            = view.get("state", {}).get("values", {})
    slack_user_id    = payload.get("user", {}).get("id", "")

    task_name    = state.get("task_name", {}).get("name_input", {}).get("value", "").strip()
    due_date     = (state.get("due_date", {}).get("due_date_input", {}).get("value") or "").strip()
    meeting_id   = private_metadata.get("mid", "")
    step_index   = private_metadata.get("si", 0)
    response_url = private_metadata.get("response_url", "")

    # New format: look up from Redis
    if meeting_id and step_index:
        step_data    = await _get_step_data(meeting_id, step_index)
        task_text    = step_data["task_text"]    if step_data else private_metadata.get("task_text", "")
        task_context = step_data["task_context"] if step_data else ""
        meta         = {"participants_str": step_data["participants_str"]} if step_data else {}
    else:
        # Backward compat
        task_text    = private_metadata.get("task_text", "")
        task_context = private_metadata.get("task_context", "")
        meta         = private_metadata.get("meta", {})

    # Assignee (multi-select)
    assignee_opts = state.get("assignee", {}).get("assignee_input", {}).get("selected_options", []) or []
    assignees     = [int(o["value"]) for o in assignee_opts] if assignee_opts else []

    # Priority
    priority_opt = state.get("priority", {}).get("priority_input", {}).get("selected_option")
    priority     = int(priority_opt["value"]) if priority_opt else None

    # Brand (required)
    brand_opt = state.get("brand", {}).get("brand_input", {}).get("selected_option")
    brand_id  = brand_opt["value"] if brand_opt else None

    # Project Type (required)
    pt_opt          = state.get("project_type", {}).get("project_type_input", {}).get("selected_option")
    project_type_id = pt_opt["value"] if pt_opt else None

    # Theme (required)
    theme_opt = state.get("theme", {}).get("theme_input", {}).get("selected_option")
    theme_id  = theme_opt["value"] if theme_opt else None

    if not task_name:
        await _respond_url(response_url, ":x: Task name is empty — not created.")
        return

    api_key = await _get_user_clickup_key(slack_user_id)
    comment = _build_comment(task_text, meta, task_context)
    try:
        result = await create_backlog_task(
            task_name, "", due_date,
            api_key=api_key,
            assignees=assignees or None,
            priority=priority,
            brand_option_id=brand_id,
            project_type_option_id=project_type_id,
            theme_option_id=theme_id
        )
        created_name = result.get("name", task_name)
        created_id   = result.get("id")
        if created_id:
            await post_task_comment(created_id, comment, api_key=api_key)
        await _respond_url(response_url, f":white_check_mark: New task created in Backlog: *{created_name}*")
        if meeting_id and step_index:
            actor_name = await _get_slack_display_name(slack_user_id)
            await _store_and_update_dm_state(meeting_id, step_index, actor_name, slack_user_id, created_name)
    except Exception as e:
        print(f"[Slack Interact] create_backlog_task failed: {e}")
        await _respond_url(response_url, f":x: Failed to create task: {e}")


# ── Comment builder ───────────────────────────────────────────────────────────

def _build_comment(task_text: str, meta: dict, task_context: str = "") -> str:
    """
    Build the comment text posted to ClickUp task Activity.
    Format: date · what was discussed · action point · participants
    """
    now_ist  = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    date_str = now_ist.strftime("%d %b %Y, %I:%M %p IST")

    participants_str = meta.get("participants_str", "")

    lines = [f"🗓️ {date_str}"]

    if task_context:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("💬  What Was Discussed")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(task_context)

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("✅  Action Point")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(task_text)

    if participants_str:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"👥  Participants: {participants_str}")

    return "\n".join(lines)


# ── Utility ───────────────────────────────────────────────────────────────────

async def _respond_url(response_url: str, text: str):
    """Post acknowledgment back to Slack via response_url."""
    if not response_url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                response_url,
                json={"text": text, "replace_original": False}
            )
    except Exception as e:
        print(f"[Slack Interact] _respond_url failed: {e}")
