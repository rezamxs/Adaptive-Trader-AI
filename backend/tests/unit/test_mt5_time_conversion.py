"""Unit tests for MT5 Time Conversion, Broker Timezone Verification, and Timestamp Staleness Protection."""
from datetime import datetime, timezone, timedelta
import pytest

from backend.app.services.market_data.time_conversion import (
    MT5TimeConverter,
    TimestampValidationResult,
)


def test_explicit_offset_conversion():
    """Verifies that MT5TimeConverter reports MT5_UNIX_EPOCH_UTC and server_offset_seconds=0."""
    converter = MT5TimeConverter(explicit_offset_seconds=3 * 3600)
    assert converter.server_offset_seconds == 0
    assert converter.offset_source == "MT5_UNIX_EPOCH_UTC"

    # Raw value 1791817800 corresponds to 2026-10-12 15:10:00 UTC
    raw_time = 1791817800
    dt_utc, reason = converter.raw_to_utc(raw_time)

    assert dt_utc is not None
    assert dt_utc == datetime(2026, 10, 12, 15, 10, 0, tzinfo=timezone.utc)
    assert reason == "MT5_UNIX_EPOCH_UTC"


def test_explicit_offset_conversion_different_broker():
    """Verifies that epoch seconds convert consistently to UTC without artificial shifting."""
    converter_eet = MT5TimeConverter(explicit_offset_seconds=2 * 3600)
    raw_time = 1791817800
    dt_utc, _ = converter_eet.raw_to_utc(raw_time)
    assert dt_utc == datetime(2026, 10, 12, 15, 10, 0, tzinfo=timezone.utc)

    # GMT+0
    converter_gmt = MT5TimeConverter(explicit_offset_seconds=0)
    dt_utc, _ = converter_gmt.raw_to_utc(raw_time)
    assert dt_utc == datetime(2026, 10, 12, 15, 10, 0, tzinfo=timezone.utc)


def test_adaptive_offset_detection_from_live_tick():
    """Verifies live tick validation confirms epoch format and reports zero offset."""
    converter = MT5TimeConverter()
    assert converter.server_offset_seconds == 0

    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
    raw_tick_time = int(now_utc.timestamp())

    ok, offset, reason = converter.detect_and_set_offset_from_live_tick(
        raw_tick_time=raw_tick_time,
        now_utc=now_utc,
    )

    assert ok is True
    assert offset == 0
    assert converter.server_offset_seconds == 0
    assert "no broker offset applied" in reason


def test_adaptive_offset_detection_rejects_excessive_skew():
    """Verifies fail-closed behavior on invalid epoch values."""
    converter = MT5TimeConverter()
    now_utc = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)

    ok, offset, reason = converter.detect_and_set_offset_from_live_tick(
        raw_tick_time="invalid-epoch-value",
        now_utc=now_utc,
    )

    assert ok is False
    assert offset is None
    assert converter.server_offset_seconds == 0
    assert "invalid_unix_epoch_timestamp" in reason.lower()


def test_fail_closed_without_verified_offset():
    """Verifies that non-convertible raw timestamps fail closed."""
    converter = MT5TimeConverter()
    raw_time = "non-numeric-timestamp"

    dt_utc, reason = converter.raw_to_utc(raw_time, require_verified_offset=True)
    assert dt_utc is None
    assert reason.startswith("CONVERSION_ERROR:")


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
    """Verifies that bar_key generated from converted UTC matches actual UTC time."""
    converter = MT5TimeConverter()
    raw_broker_time = 1791817800  # 15:10:00 UTC epoch

    dt_utc, _ = converter.raw_to_utc(raw_broker_time)
    assert dt_utc is not None

    bar_key = f"XAUUSD_M1_{dt_utc.strftime('%Y%m%d%H%M')}"
    assert bar_key == "XAUUSD_M1_202610121510"
