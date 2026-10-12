"""Time conversion and validation utilities for MetaTrader 5 market data.
Provides verifiable server timezone conversion, future timestamp protection, and tick staleness auditing.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TimestampValidationResult:
    """Structured result of timestamp validation."""
    is_valid: bool
    age_seconds: float
    is_future: bool
    is_stale: bool
    reason: str


class MT5TimeConverter:
    """Manages time conversion between MT5 broker server time and UTC with strict validation.
    
    Guarantees:
    - Never uses an unverified arbitrary offset without proof.
    - Strictly rejects future timestamps (negative age beyond skew tolerance).
    - Detects broker server timezone offset adaptively from verified live ticks if not explicitly configured.
    - Fails closed when timezone contract cannot be verified.
    """

    DEFAULT_MAX_STALENESS_SECONDS: float = 120.0
    DEFAULT_MAX_FUTURE_TOLERANCE_SECONDS: float = 5.0

    def __init__(
        self,
        explicit_offset_seconds: Optional[int] = None,
        max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
        max_future_tolerance_seconds: float = DEFAULT_MAX_FUTURE_TOLERANCE_SECONDS,
    ):
        self._explicit_offset_seconds = explicit_offset_seconds
        self._detected_offset_seconds: Optional[int] = None
        self._offset_source: str = "EXPLICIT" if explicit_offset_seconds is not None else "UNSET"
        self.max_staleness_seconds = max_staleness_seconds
        self.max_future_tolerance_seconds = max_future_tolerance_seconds

    @property
    def server_offset_seconds(self) -> Optional[int]:
        """Returns verified offset in seconds, or None if unverified."""
        if self._explicit_offset_seconds is not None:
            return self._explicit_offset_seconds
        return self._detected_offset_seconds

    @property
    def offset_source(self) -> str:
        """Indicates whether offset is EXPLICIT, DYNAMICALLY_DETECTED, or UNVERIFIED."""
        if self._explicit_offset_seconds is not None:
            return "EXPLICIT"
        if self._detected_offset_seconds is not None:
            return "DYNAMICALLY_DETECTED"
        return "UNVERIFIED"

    def set_explicit_offset_hours(self, hours: float) -> None:
        """Sets explicit broker timezone offset in hours (e.g. 3.0 for UTC+3)."""
        self._explicit_offset_seconds = int(round(hours * 3600))
        self._offset_source = "EXPLICIT"

    def clear_offset(self) -> None:
        """Clears explicit and detected offsets."""
        self._explicit_offset_seconds = None
        self._detected_offset_seconds = None
        self._offset_source = "UNSET"

    def detect_and_set_offset_from_live_tick(
        self,
        raw_tick_time: int,
        now_utc: Optional[datetime] = None,
        max_clock_skew_seconds: float = 60.0
    ) -> Tuple[bool, Optional[int], str]:
        """Dynamically detects and validates broker server offset using a known live quote.
        
        Logic:
        1. Compares raw_tick_time to UTC epoch.
        2. Broker offsets in standard financial venues are integer or half-hour multiples of 1800s.
        3. Validates that the clock skew to the nearest standard timezone bucket is within max_clock_skew_seconds.
        4. If validated, stores as detected offset. Otherwise, rejects without guessing.
        """
        now = now_utc or datetime.now(timezone.utc)
        utc_epoch = now.timestamp()
        raw_diff = raw_tick_time - utc_epoch

        # Find nearest standard half-hour offset (1800 seconds)
        candidate_offset = int(round(raw_diff / 1800.0) * 1800)
        skew = abs(raw_diff - candidate_offset)

        # Standard broker timezones typically fall between UTC-12 and UTC+14
        if candidate_offset < -43200 or candidate_offset > 50400:
            reason = f"Calculated offset {candidate_offset}s outside valid global timezone boundaries [-12h, +14h]."
            return False, None, reason

        if skew > max_clock_skew_seconds:
            reason = f"Clock skew {skew:.1f}s exceeds threshold {max_clock_skew_seconds}s for offset {candidate_offset/3600:.1f}h."
            return False, None, reason

        self._detected_offset_seconds = candidate_offset
        self._offset_source = "DYNAMICALLY_DETECTED"
        offset_hours = candidate_offset / 3600.0
        reason = f"Broker offset verified as UTC{offset_hours:+.1f}h (skew={skew:.2f}s)."
        return True, candidate_offset, reason

    def raw_to_utc(
        self,
        raw_seconds: int,
        require_verified_offset: bool = True
    ) -> Tuple[Optional[datetime], str]:
        """Converts raw MT5 epoch integer to true UTC datetime.
        
        If require_verified_offset is True and no offset is verified:
        Fails closed, returns (None, 'UNVERIFIED_SERVER_TIMEZONE').
        """
        offset = self.server_offset_seconds
        if offset is None:
            if require_verified_offset:
                return None, "UNVERIFIED_SERVER_TIMEZONE: Cannot convert raw timestamp without verified broker offset."
            # Fallback without adjustment (treats raw as UTC - vulnerable to skew)
            dt = datetime.fromtimestamp(raw_seconds, tz=timezone.utc)
            return dt, "UNADJUSTED_RAW_FALLBACK"

        # Correct calculation: true_utc_seconds = raw_broker_seconds - broker_offset_seconds
        utc_seconds = raw_seconds - offset
        try:
            dt = datetime.fromtimestamp(utc_seconds, tz=timezone.utc)
            return dt, f"CONVERTED_WITH_OFFSET_{offset}S"
        except (ValueError, OverflowError, OSError) as e:
            return None, f"CONVERSION_ERROR: {e}"

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
