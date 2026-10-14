"""Guards the two decisions that decide whether a long meeting is transcribed completely on the free tier:
where audio is cut into parts, and which Gemini errors mean wait / fall back / give up for the day."""
from datetime import datetime, timezone

from services.gemini_transcriber import _classify_error, _parse_ts, _plan_parts, _seconds_until_quota_reset


def test_short_meeting_is_one_part():
    assert _plan_parts(832, [100, 500]) == [(0.0, 832)]


def test_one_hour_meeting_cuts_at_nearest_pause():
    parts = _plan_parts(3600, [880, 905, 1790, 2750], part_seconds=900)
    assert parts == [(0.0, 905), (905, 1790), (1790, 2750), (2750, 3600)]


def test_no_pause_nearby_falls_back_to_hard_cut():
    assert _plan_parts(2000, [], part_seconds=900) == [(0.0, 900), (900, 1800), (1800, 2000)]
    assert _plan_parts(1900, [], part_seconds=900) == [(0.0, 900), (900, 1900)]  # 1000s tail <= 1.2x: kept whole


def test_parts_cover_whole_audio_without_gaps():
    parts = _plan_parts(7300, [905.5, 1810.2, 2700, 3610, 4499, 5420], part_seconds=900)
    assert parts[0][0] == 0.0 and parts[-1][1] == 7300
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))


def test_error_classification():
    daily = {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}
    per_minute = {"details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                              {"retryDelay": "37s"}]}
    assert _classify_error(429, daily) == ("daily", None)
    assert _classify_error(429, per_minute) == ("retry", 37.0)
    assert _classify_error(503, {}) == ("retry", None)
    assert _classify_error(404, {}) == ("next_model", None)


def test_timestamps_and_quota_reset():
    assert _parse_ts("01:43") == 103 and _parse_ts("1:02:03") == 3723 and _parse_ts("bad") is None
    # 06:00 UTC -> next reset 08:00 UTC the same day (+60s margin)
    assert _seconds_until_quota_reset(datetime(2026, 10, 14, 6, 0, tzinfo=timezone.utc)) == 2 * 3600 + 60
