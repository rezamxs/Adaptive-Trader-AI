"""MT5 epoch timestamps are UTC and must not receive broker-local offsets."""
from datetime import datetime, timezone
import numpy as np

from backend.app.services.market_data.time_conversion import MT5TimeConverter


def test_epoch_seconds_are_utc_even_if_legacy_offset_is_configured():
    timestamp = 1_797_000_000
    converter = MT5TimeConverter(explicit_offset_seconds=3 * 60 * 60)

    converted, reason = converter.raw_to_utc(timestamp)

    assert converted == datetime.fromtimestamp(timestamp, tz=timezone.utc)
    assert converted.utcoffset().total_seconds() == 0
    assert reason == "MT5_UNIX_EPOCH_UTC"


def test_live_tick_does_not_infer_a_broker_offset():
    converter = MT5TimeConverter()
    timestamp = 1_797_000_000

    verified, offset, reason = converter.detect_and_set_offset_from_live_tick(timestamp)

    assert verified
    assert offset == 0
    assert converter.server_offset_seconds == 0
    assert "no broker offset applied" in reason


def test_invalid_epoch_value_is_rejected():
    converted, reason = MT5TimeConverter().raw_to_utc("not-a-timestamp")

    assert converted is None
    assert reason.startswith("CONVERSION_ERROR:")



def test_numpy_integer_rate_timestamp_is_converted_as_utc():
    timestamp = np.int64(1_797_000_000)
    converted, reason = MT5TimeConverter().raw_to_utc(timestamp)
    assert converted == datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    assert reason == "MT5_UNIX_EPOCH_UTC"
