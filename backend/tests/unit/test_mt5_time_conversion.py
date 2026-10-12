"""Unit tests for MT5 Time Conversion, Broker Timezone Verification, and Timestamp Staleness Protection."""
from datetime import datetime, timezone, timedelta
import pytest

from backend.app.services.market_data.time_conversion import (
    MT5TimeConverter,
    TimestampValidationResult,
)


def test_explicit_offset_conversion():
    """Verifies that an explicitly known broker offset (+3h) converts raw MT5 time to true UTC."""
    converter = MT5TimeConverter(explicit_offset_seconds=3 * 3600)
    assert converter.server_offset_seconds == 10800
    assert converter.offset_source == "EXPLICIT"

    # Raw value from Phase Q2.2 report: 1791817800 (15:10:00 broker server time)
    raw_time = 1791817800
    dt_utc, reason = converter.raw_to_utc(raw_time)

    assert dt_utc is not None
    assert dt_utc == datetime(2026, 10, 12, 12, 10, 0, tzinfo=timezone.utc)
    assert "CONVERTED_WITH_OFFSET_10800S" in reason


def test_explicit_offset_conversion_different_broker():
    """Verifies that converter works with other broker timezones (e.g. London GMT+0 or Winter EET GMT+2)."""
    # Winter EET (+2h)
    converter_eet = MT5TimeConverter(explicit_offset_seconds=2 * 3600)
    raw_time = 1791817800  # 15:10 server time in GMT+2 means 13:10 UTC
    dt_utc, _ = converter_eet.raw_to_utc(raw_time)
    assert dt_utc == datetime(2026, 10, 12, 13, 10, 0, tzinfo=timezone.utc)

    # GMT+0
    converter_gmt = MT5TimeConverter(explicit_offset_seconds=0)
    dt_utc, _ = converter_gmt.raw_to_utc(raw_time)
    assert dt_utc == datetime(2026, 10, 12, 15, 10, 0, tzinfo=timezone.utc)


def test_adaptive_offset_detection_from_live_tick():
    """Verifies adaptive offset detection from a live quote with minor clock skew."""
    converter = MT5TimeConverter()
    assert converter.server_offset_seconds is None

    # Simulated live scenario:
    # True UTC time is 12:00:00
    # Broker clock is UTC+3 (15:00:00) with 1.5s network/tick age delay -> raw_tick_time = 15:00:00 - 1.5s
    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
    utc_epoch = now_utc.timestamp()
    raw_tick_time = int(utc_epoch + 10800 - 2)  # 2s lag behind server 15:00:00

    ok, offset, reason = converter.detect_and_set_offset_from_live_tick(
        raw_tick_time=raw_tick_time,
        now_utc=now_utc,
        max_clock_skew_seconds=10.0
    )

    assert ok is True
    assert offset == 10800  # Exactly 3 hours (10800s)
    assert converter.server_offset_seconds == 10800
    assert converter.offset_source == "DYNAMICALLY_DETECTED"
    assert "UTC+3.0h" in reason


def test_adaptive_offset_detection_rejects_excessive_skew():
    """Verifies fail-closed behavior: rejects offset detection if skew does not match standard timezones."""
    converter = MT5TimeConverter()
    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
    utc_epoch = now_utc.timestamp()

    # Raw tick diff is 10800 + 400 seconds (too much skew to be a real standard timezone clock)
    raw_tick_time = int(utc_epoch + 11200)

    ok, offset, reason = converter.detect_and_set_offset_from_live_tick(
        raw_tick_time=raw_tick_time,
        now_utc=now_utc,
        max_clock_skew_seconds=60.0
    )

    assert ok is False
    assert offset is None
    assert converter.server_offset_seconds is None
    assert "exceeds threshold" in reason


def test_fail_closed_without_verified_offset():
    """Verifies that without verified offset, strict mode refuses to guess and fails closed."""
    converter = MT5TimeConverter()
    raw_time = 1791817800

    dt_utc, reason = converter.raw_to_utc(raw_time, require_verified_offset=True)
    assert dt_utc is None
    assert "UNVERIFIED_SERVER_TIMEZONE" in reason


def test_future_timestamp_rejected():
    """Verifies that any quote with a future timestamp is detected and rejected."""
    converter = MT5TimeConverter(max_future_tolerance_seconds=2.0)
    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)

    # Future timestamp: 10 seconds in the future
    future_time = now_utc + timedelta(seconds=10)
    res: TimestampValidationResult = converter.validate_timestamp(future_time, now_utc=now_utc)

    assert res.is_valid is False
    assert res.is_future is True
    assert res.is_stale is False
    assert res.age_seconds == -10.0
    assert "FUTURE_TIMESTAMP_DETECTED" in res.reason


def test_future_timestamp_within_skew_tolerance_accepted():
    """Verifies that tiny clock jitter within max_future_tolerance is tolerated."""
    converter = MT5TimeConverter(max_future_tolerance_seconds=3.0)
    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)

    # 1.5 seconds in the future (within 3.0s tolerance)
    jitter_time = now_utc + timedelta(seconds=1.5)
    res: TimestampValidationResult = converter.validate_timestamp(jitter_time, now_utc=now_utc)

    assert res.is_valid is True
    assert res.is_future is False
    assert res.age_seconds == -1.5


def test_stale_timestamp_rejected():
    """Verifies that quotes exceeding max_staleness_seconds are marked as stale."""
    converter = MT5TimeConverter(max_staleness_seconds=120.0)
    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)

    # 130 seconds old
    stale_time = now_utc - timedelta(seconds=130)
    res: TimestampValidationResult = converter.validate_timestamp(stale_time, now_utc=now_utc)

    assert res.is_valid is False
    assert res.is_future is False
    assert res.is_stale is True
    assert res.age_seconds == 130.0
    assert "STALE_TIMESTAMP" in res.reason


def test_fresh_valid_timestamp_accepted():
    """Verifies that fresh timestamps within bounds are approved."""
    converter = MT5TimeConverter(max_staleness_seconds=120.0, max_future_tolerance_seconds=5.0)
    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)

    # Fresh timestamp: 4.5 seconds old
    fresh_time = now_utc - timedelta(seconds=4.5)
    res: TimestampValidationResult = converter.validate_timestamp(fresh_time, now_utc=now_utc)

    assert res.is_valid is True
    assert res.is_future is False
    assert res.is_stale is False
    assert res.age_seconds == 4.5
    assert "TIMESTAMP_VALID" in res.reason


def test_bar_key_generation_matches_utc():
    """Verifies that bar_key generated from converted UTC matches actual UTC time, resolving Q2.2 discrepancy."""
    converter = MT5TimeConverter(explicit_offset_seconds=3 * 3600)
    raw_broker_time = 1791817800  # 15:10 broker time

    dt_utc, _ = converter.raw_to_utc(raw_broker_time)
    assert dt_utc is not None

    bar_key = f"XAUUSD_M1_{dt_utc.strftime('%Y%m%d%H%M')}"
    # Must be 12:10 UTC, NOT 15:10
    assert bar_key == "XAUUSD_M1_202610121210"
