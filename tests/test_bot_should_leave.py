"""Guards the predicate that decides when to evict a Recall bot from a huddle.

Getting this wrong in either direction is expensive:
  - too eager  → leave_call lands mid-meeting and truncates a live recording
  - too late   → bot shuts down uncommandable and holds the Slack huddle open
"""
from main import _bot_should_leave


def test_join_transition_does_not_leave():
    # 'in_call_not_recording' also fires on join, before anything is recorded.
    assert not _bot_should_leave(["joining_call", "in_call_not_recording"])


def test_live_recording_is_never_cut_short():
    assert not _bot_should_leave(
        ["joining_call", "in_call_not_recording", "in_call_recording"]
    )


def test_recording_finished_but_still_in_call_leaves():
    # The one window where leave_call still works.
    assert _bot_should_leave(
        ["joining_call", "in_call_not_recording", "in_call_recording",
         "in_call_not_recording"]
    )


def test_already_shut_down_is_not_retried():
    # Real 2026-07-31 ghost: recording_done with no call_ended/done. Recall
    # rejects commands here (400 bot_command_error), so don't bother.
    assert not _bot_should_leave(
        ["joining_call", "in_call_not_recording", "in_call_recording",
         "in_call_not_recording", "recording_done"]
    )


def test_healthy_completed_bot_is_left_alone():
    assert not _bot_should_leave(
        ["joining_call", "in_call_not_recording", "in_call_recording",
         "call_ended", "recording_done", "done"]
    )


def test_empty_history():
    assert not _bot_should_leave([])
