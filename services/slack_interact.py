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


async def _get_user_clickup_key(slack_user_id: str) -> str | None:
    try:
        return await redis_client.get(f"clickup_key:{slack_user_id}")
    except Exception:
        return None


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

        if action_id.startswith("confirm_"):
            await _handle_confirm(action, response_url, slack_user_id, trigger_id)
        elif action_id.startswith("change_") or action_id.startswith("pick_"):
            await _handle_change_or_pick(action, trigger_id, response_url, slack_user_id)
        elif action_id.startswith("create_"):
            await _handle_create(action, trigger_id, response_url, slack_user_id)

    elif interaction_type == "view_submission":
        callback_id = payload.get("view", {}).get("callback_id", "")
        if callback_id == "api_key_modal":
            await _handle_api_key_submit(payload)
        elif callback_id == "create_task_modal":
            await _handle_create_task_submit(payload)
        else:
            await _handle_modal_submit(payload)


async def _handle_confirm(action: dict, response_url: str, slack_user_id: str, trigger_id: str):
    """
    User clicked Confirm — check API key first, then post comment to suggested ClickUp task.
    """
    from services.clickup import post_task_comment, comment_already_exists

    value        = json.loads(action.get("value", "{}"))
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
    except Exception as e:
        print(f"[Slack Interact] post_task_comment failed: {e}")
        await _respond_url(response_url, f":x: Failed to post comment: {e}")


async def _handle_change_or_pick(action: dict, trigger_id: str, response_url: str, slack_user_id: str):
    """User clicked Change Task or Pick a Task — check API key first, then open modal."""
    value = json.loads(action.get("value", "{}"))

    api_key = await _get_user_clickup_key(slack_user_id)
    if not api_key:
        await _open_api_key_modal(trigger_id, "pick", value, response_url)
        return

    await _open_pick_task_modal(trigger_id, value, response_url)


async def _handle_create(action: dict, trigger_id: str, response_url: str, slack_user_id: str):
    """User clicked Create New Task — check API key first, then open modal."""
    value = json.loads(action.get("value", "{}"))

    api_key = await _get_user_clickup_key(slack_user_id)
    if not api_key:
        await _open_api_key_modal(trigger_id, "create", value, response_url)
        return

    await _open_create_task_modal(trigger_id, value, response_url)


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

    # Validate the key
    valid, clickup_name = await validate_clickup_api_key(api_key)
    if not valid:
        await _respond_url(response_url, ":x: Invalid ClickUp API key — please click the button again and enter the correct key.")
        return

    # Save to Redis permanently
    await redis_client.set(f"clickup_key:{slack_user_id}", api_key)
    print(f"[Slack Interact] ClickUp key saved for Slack user {slack_user_id} ({clickup_name})")

    # Re-execute original action
    if pending_action == "confirm":
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
        except Exception as e:
            await _respond_url(response_url, f":x: Key saved but failed to post: {e}")

    elif pending_action == "pick":
        await _respond_url(response_url, f":white_check_mark: ClickUp connected as *{clickup_name}*! Please click *Change Task / Pick a Task* again.")

    elif pending_action == "create":
        await _respond_url(response_url, f":white_check_mark: ClickUp connected as *{clickup_name}*! Please click *Create New Task* again.")


async def _open_create_task_modal(trigger_id: str, action_value: dict, response_url: str):
    """
    Opens a modal pre-filled with the action point as task name.
    Required: Brand, Project Type, Theme (fetched live from ClickUp, 1hr cache).
    Optional: Due Date, Assignee, Priority.
    """
    from services.clickup import get_backlog_custom_fields, get_backlog_members, CUSTOM_FIELD_BRAND, CUSTOM_FIELD_PROJECT_TYPE, CUSTOM_FIELD_THEME

    task_text = action_value.get("task_text", "")
    deadline  = action_value.get("deadline", "") or ""

    private_metadata = json.dumps({
        "task_text":    task_text,
        "task_context": action_value.get("task_context", ""),
        "meta":         action_value.get("meta", {}),
        "response_url": response_url
    })

    # Fetch all dynamic data live from ClickUp (1hr cache each)
    cf, members = await asyncio.gather(get_backlog_custom_fields(), get_backlog_members())

    def _to_slack_options(field_id: str) -> list:
        return [
            {"text": {"type": "plain_text", "text": o["name"]}, "value": o["id"]}
            for o in cf.get(field_id, [])
        ]

    brand_options   = _to_slack_options(CUSTOM_FIELD_BRAND)
    pt_options      = _to_slack_options(CUSTOM_FIELD_PROJECT_TYPE)
    theme_options   = _to_slack_options(CUSTOM_FIELD_THEME)

    # If ClickUp API failed and options are empty, show error and abort
    if not brand_options or not pt_options or not theme_options:
        await _respond_url(response_url, ":x: Could not load ClickUp fields — please try again in a moment.")
        return
    member_options  = [
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


async def _handle_create_task_submit(payload: dict):
    """User submitted the create-task modal — create task in ClickUp Backlog + post comment to activity."""
    from services.clickup import create_backlog_task, post_task_comment

    view             = payload.get("view", {})
    private_metadata = json.loads(view.get("private_metadata", "{}"))
    state            = view.get("state", {}).get("values", {})
    slack_user_id    = payload.get("user", {}).get("id", "")

    task_name    = state.get("task_name", {}).get("name_input", {}).get("value", "").strip()
    due_date     = (state.get("due_date", {}).get("due_date_input", {}).get("value") or "").strip()
    task_text    = private_metadata.get("task_text", "")
    task_context = private_metadata.get("task_context", "")
    meta         = private_metadata.get("meta", {})
    response_url = private_metadata.get("response_url", "")

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
        task_id = result.get("id")
        if task_id:
            await post_task_comment(task_id, comment, api_key=api_key)
        await _respond_url(response_url, f":white_check_mark: New task created in Backlog: *{created_name}*")
    except Exception as e:
        print(f"[Slack Interact] create_backlog_task failed: {e}")
        await _respond_url(response_url, f":x: Failed to create task: {e}")


async def _open_pick_task_modal(trigger_id: str, action_value: dict, response_url: str):
    """
    Opens a Slack modal with an external_select dropdown.
    Options load dynamically from /webhook/slack-options.
    """
    private_metadata = json.dumps({
        **action_value,
        "response_url": response_url
    })

    modal = {
        "type":             "modal",
        "callback_id":      "pick_task_modal",
        "title":            {"type": "plain_text", "text": "Pick a Task"},
        "submit":           {"type": "plain_text", "text": "Post Comment"},
        "close":            {"type": "plain_text", "text": "Cancel"},
        "private_metadata": private_metadata,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Action Point:*\n{action_value.get('task_text', '')}"
                }
            },
            {"type": "divider"},
            {
                "type":     "input",
                "block_id": "task_select",
                "element": {
                    "type":             "external_select",
                    "placeholder":      {"type": "plain_text", "text": "Search tasks..."},
                    "action_id":        "selected_task",
                    "min_query_length": 0
                },
                "label": {"type": "plain_text", "text": "Select a ClickUp Task"}
            }
        ]
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://slack.com/api/views.open",
            headers=HEADERS,
            json={"trigger_id": trigger_id, "view": modal}
        )
    resp = r.json()
    if not resp.get("ok"):
        print(f"[Slack Interact] views.open failed: {resp.get('error')}")


async def _handle_modal_submit(payload: dict):
    """User submitted the pick-task modal — post comment to the selected ClickUp task."""
    from services.clickup import post_task_comment, comment_already_exists

    view             = payload.get("view", {})
    private_metadata = json.loads(view.get("private_metadata", "{}"))
    state            = view.get("state", {}).get("values", {})
    slack_user_id    = payload.get("user", {}).get("id", "")

    selected_opt  = state.get("task_select", {}).get("selected_task", {}).get("selected_option", {})
    selected_id   = selected_opt.get("value", "")
    selected_name = selected_opt.get("text", {}).get("text", "")

    task_text    = private_metadata.get("task_text", "")
    task_context = private_metadata.get("task_context", "")
    meta         = private_metadata.get("meta", {})
    response_url = private_metadata.get("response_url", "")

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
    except Exception as e:
        print(f"[Slack Interact] modal post_task_comment failed: {e}")
        await _respond_url(response_url, f":x: Failed to post comment: {e}")


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
