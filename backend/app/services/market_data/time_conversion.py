"""UTC conversion and freshness validation for MetaTrader 5 Unix timestamps.

MT5's Python API exposes quote and rate times as Unix epoch seconds. These are
already UTC and must not be shifted by a broker's chart/display timezone.
"""
from datetime import datetime, timezone
from typing import Optional, Tuple
from dataclasses import dataclass


class TimestampValidationResult:
    """Structured result of timestamp validation."""
    is_valid: bool
    age_seconds: float
    is_future: bool
    is_stale: bool
    reason: str


class MT5TimeConverter:
    """Converts MT5 Unix epoch seconds directly to timezone-aware UTC datetimes.

    The offset helpers remain for backward compatibility. They always report zero
    and never modify epoch timestamps.
    """

    DEFAULT_MAX_STALENESS_SECONDS: float = 120.0
    DEFAULT_MAX_FUTURE_TOLERANCE_SECONDS: float = 5.0

    def __init__(
        self,
        explicit_offset_seconds: Optional[int] = None,
        max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
        max_future_tolerance_seconds: float = DEFAULT_MAX_FUTURE_TOLERANCE_SECONDS,
    ):
        # Explicit offsets are intentionally ignored; epoch seconds are UTC.
        self._explicit_offset_seconds = None
        self._detected_offset_seconds: Optional[int] = 0
        self._offset_source = "MT5_UNIX_EPOCH_UTC"
        self.max_staleness_seconds = float(max_staleness_seconds)
        self.max_future_tolerance_seconds = float(max_future_tolerance_seconds)

    @property
    def server_offset_seconds(self) -> Optional[int]:
        """Returns zero because MT5 Unix epoch timestamps are already UTC."""
        return 0

    @property
    def offset_source(self) -> str:
        return "MT5_UNIX_EPOCH_UTC"

    def set_explicit_offset_hours(self, hours: float) -> None:
        """Reject legacy offset settings that could shift UTC epoch timestamps."""
        raise ValueError(
            "MT5 Unix epoch timestamps are already UTC; broker display offsets must not be applied."
        )

    def clear_offset(self) -> None:
        """Reset legacy state while preserving UTC epoch semantics."""
        self._explicit_offset_seconds = None
        self._detected_offset_seconds = 0
        self._offset_source = "MT5_UNIX_EPOCH_UTC"

    def detect_and_set_offset_from_live_tick(
        self,
        raw_tick_time: int,
        now_utc: Optional[datetime] = None,
        max_clock_skew_seconds: float = 60.0,
    ) -> Tuple[bool, Optional[int], str]:
        """Legacy API: validate that a supplied value is a valid Unix epoch."""
        try:
            datetime.fromtimestamp(raw_tick_time, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            return False, None, f"INVALID_UNIX_EPOCH_TIMESTAMP: {exc}"
        self._detected_offset_seconds = 0
        self._offset_source = "MT5_UNIX_EPOCH_UTC"
        return True, 0, "MT5 Unix epoch timestamp interpreted as UTC; no broker offset applied."

    def raw_to_utc(
        self,
        raw_seconds: int,
        require_verified_offset: bool = True,
    ) -> Tuple[Optional[datetime], str]:
        """Convert Unix epoch seconds to UTC without any broker-time adjustment."""
        try:
            dt = datetime.fromtimestamp(raw_seconds, tz=timezone.utc)
            return dt, "MT5_UNIX_EPOCH_UTC"
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            return None, f"CONVERSION_ERROR: {exc}"

    def validate_timestamp(
        self,
        dt_utc: datetime,
        now_utc: Optional[datetime] = None,
        max_staleness_seconds: Optional[float] = None,
        max_future_tolerance_seconds: Optional[float] = None
    ) -> TimestampValidationResult:
        """Validates UTC timestamp against current UTC clock for future skew and staleness."""
        now = now_utc or datetime.now(timezone.utc)
        if dt_utc.tzinfo is None:
            # Assume UTC if naive
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

        max_stale = max_staleness_seconds if max_staleness_seconds is not None else self.max_staleness_seconds
        max_future = max_future_tolerance_seconds if max_future_tolerance_seconds is not None else self.max_future_tolerance_seconds

        age_seconds = (now - dt_utc).total_seconds()

        # Check future timestamp (negative age beyond tolerance)
        if age_seconds < -max_future:
            return TimestampValidationResult(
                is_valid=False,
                age_seconds=age_seconds,
                is_future=True,
                is_stale=False,
                reason=f"FUTURE_TIMESTAMP_DETECTED: Timestamp is {-age_seconds:.1f}s in the future (tolerance {max_future}s).",
            )

        # Check staleness
        if age_seconds > max_stale:
            return TimestampValidationResult(
                is_valid=False,
                age_seconds=age_seconds,
                is_future=False,
                is_stale=True,
                reason=f"STALE_TIMESTAMP: Age {age_seconds:.1f}s exceeds maximum threshold {max_stale}s.",
            )

        return TimestampValidationResult(
            is_valid=True,
            age_seconds=age_seconds,
            is_future=False,
            is_stale=False,
            reason=f"TIMESTAMP_VALID: Age {age_seconds:.1f}s is fresh and within bounds.",
        )
