import httpx
import json
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
    "Content-Type": "application/json"
}


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
        action       = actions[0]
        action_id    = action.get("action_id", "")
        response_url = payload.get("response_url", "")
        trigger_id   = payload.get("trigger_id", "")

        if action_id.startswith("confirm_"):
            await _handle_confirm(action, response_url)
        elif action_id.startswith("change_") or action_id.startswith("pick_"):
            await _handle_change_or_pick(action, trigger_id, response_url)
        elif action_id.startswith("create_"):
            await _handle_create(action, trigger_id, response_url)

    elif interaction_type == "view_submission":
        callback_id = payload.get("view", {}).get("callback_id", "")
        if callback_id == "create_task_modal":
            await _handle_create_task_submit(payload)
        else:
            await _handle_modal_submit(payload)


async def _handle_confirm(action: dict, response_url: str):
    """
    User clicked Confirm — post comment to the suggested ClickUp task immediately.
    Checks for duplicate first — skips if same action point already posted.
    """
    from services.clickup import post_task_comment, comment_already_exists

    value     = json.loads(action.get("value", "{}"))
    task_id   = value.get("clickup_task_id")
    task_name = value.get("clickup_task_name", "")
    task_text = value.get("task_text", "")
    meta      = value.get("meta", {})

    if not task_id:
        await _respond_url(response_url, ":x: No ClickUp task ID found — could not post comment.")
        return

    if await comment_already_exists(task_id, task_text):
        await _respond_url(response_url, f":information_source: Already posted to *{task_name}* by someone else.")
        return

    comment = _build_comment(task_text, meta)
    try:
        await post_task_comment(task_id, comment)
        await _respond_url(response_url, f":white_check_mark: Posted to *{task_name}*")
    except Exception as e:
        print(f"[Slack Interact] post_task_comment failed: {e}")
        await _respond_url(response_url, f":x: Failed to post comment: {e}")


async def _handle_change_or_pick(action: dict, trigger_id: str, response_url: str):
    """
    User clicked Change Task or Pick a Task — open a modal with Backlog task dropdown.
    """
    value = json.loads(action.get("value", "{}"))
    await _open_pick_task_modal(trigger_id, value, response_url)


async def _handle_create(action: dict, trigger_id: str, response_url: str):
    """User clicked Create New Task — open a modal to confirm/edit task name and due date."""
    value = json.loads(action.get("value", "{}"))
    await _open_create_task_modal(trigger_id, value, response_url)


async def _open_create_task_modal(trigger_id: str, action_value: dict, response_url: str):
    """
    Opens a modal pre-filled with the action point as task name + optional due date.
    User can edit before creating in ClickUp Backlog.
    """
    task_text = action_value.get("task_text", "")
    deadline  = action_value.get("deadline", "") or ""

    private_metadata = json.dumps({
        "task_text":    task_text,
        "meta":         action_value.get("meta", {}),
        "response_url": response_url
    })

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
            "block_id": "due_date",
            "optional": True,
            "element": {
                "type":          "plain_text_input",
                "action_id":     "due_date_input",
                "initial_value": deadline,
                "placeholder":   {"type": "plain_text", "text": "YYYY-MM-DD  e.g. 2026-03-25"}
            },
            "label": {"type": "plain_text", "text": "Due Date (optional)"}
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
    """User submitted the create-task modal — create a new task in ClickUp Backlog."""
    from services.clickup import create_backlog_task

    view             = payload.get("view", {})
    private_metadata = json.loads(view.get("private_metadata", "{}"))
    state            = view.get("state", {}).get("values", {})

    task_name    = state.get("task_name", {}).get("name_input", {}).get("value", "").strip()
    due_date     = (state.get("due_date", {}).get("due_date_input", {}).get("value") or "").strip()
    task_text    = private_metadata.get("task_text", "")
    meta         = private_metadata.get("meta", {})
    response_url = private_metadata.get("response_url", "")

    if not task_name:
        await _respond_url(response_url, ":x: Task name is empty — not created.")
        return

    comment = _build_comment(task_text, meta)
    try:
        from services.clickup import post_task_comment
        result = await create_backlog_task(task_name, "", due_date)
        created_name = result.get("name", task_name)
        task_id = result.get("id")
        if task_id:
            await post_task_comment(task_id, comment)
        await _respond_url(response_url, f":white_check_mark: New task created in Backlog: *{created_name}*")
    except Exception as e:
        print(f"[Slack Interact] create_backlog_task failed: {e}")
        await _respond_url(response_url, f":x: Failed to create task: {e}")


async def _open_pick_task_modal(trigger_id: str, action_value: dict, response_url: str):
    """
    Opens a Slack modal with an external_select dropdown.
    Options load dynamically from /webhook/slack-options — works across all tasks, no cap.
    User can type to search any task by name. Assigned tasks shown first by default.
    User selects a task and submits — handled by _handle_modal_submit.
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
    """
    User submitted the pick-task modal — post comment to the selected ClickUp task.
    """
    from services.clickup import post_task_comment, comment_already_exists

    view             = payload.get("view", {})
    private_metadata = json.loads(view.get("private_metadata", "{}"))

    # Extract selected task from modal state
    state         = view.get("state", {}).get("values", {})
    selected_opt  = state.get("task_select", {}).get("selected_task", {}).get("selected_option", {})
    selected_id   = selected_opt.get("value", "")
    selected_name = selected_opt.get("text", {}).get("text", "")

    task_text    = private_metadata.get("task_text", "")
    meta         = private_metadata.get("meta", {})
    response_url = private_metadata.get("response_url", "")

    if not selected_id or selected_id == "none":
        await _respond_url(response_url, ":x: No task selected — comment not posted.")
        return

    if await comment_already_exists(selected_id, task_text):
        await _respond_url(response_url, f":information_source: Already posted to *{selected_name}* by someone else.")
        return

    comment = _build_comment(task_text, meta)
    try:
        await post_task_comment(selected_id, comment)
        await _respond_url(response_url, f":white_check_mark: Posted to *{selected_name}*")
    except Exception as e:
        print(f"[Slack Interact] modal post_task_comment failed: {e}")
        await _respond_url(response_url, f":x: Failed to post comment: {e}")


def _build_comment(task_text: str, meta: dict) -> str:
    """
    Build the comment text that gets posted to ClickUp task Activity.
    Format: date/time IST · participants · discussion overview · action point
    """
    now_ist  = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    date_str = now_ist.strftime("%d %b %Y, %I:%M %p IST")

    participants_str = meta.get("participants_str", "")
    overview         = meta.get("overview", "")

    lines = [date_str]
    if participants_str:
        lines.append(f"Participants: {participants_str}")
    if overview:
        lines.append("")
        lines.append("Discussion:")
        lines.append(overview)
    lines.append("")
    lines.append("Action Point:")
    lines.append(task_text)

    return "\n".join(lines)


async def _respond_url(response_url: str, text: str):
    """
    Post a text acknowledgment back to Slack using the interaction's response_url.
    Does not replace the original message — appends a new message below it.
    """
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
