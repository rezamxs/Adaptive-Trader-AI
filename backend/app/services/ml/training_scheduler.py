"""Step 3.4: scheduled automatic candidate training (owner request 2026-11-01).

A daemon thread wakes once per UTC day at TRAINING_SCHEDULE_HOUR_UTC and
trains ONE candidate model on real MT5 data via the exact same shared flow
the /ml/train endpoint uses — so the scheduled path and the manual path can
never drift. It NEVER promotes: activation stays a human POST /promote.

Failure semantics: any failure (MT5 down, market closed, bad data, internal
bug) is logged and recorded in last_training_status; the next scheduled day
retries. Nothing trading-related is touched — the worker's signal loop keeps
using whatever active model the human last promoted.
"""
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger

# Module-level last-run status, surfaced in /system/status for observability.
last_training_status: Dict[str, Any] = {
    "ran_at": None,
    "outcome": None,
    "model_id": None,
    "detail": None,
}


def compute_next_run(now_utc: datetime, hour_utc: int) -> datetime:
    """Next occurrence of `hour_utc` (today if still ahead, else tomorrow)."""
    candidate = now_utc.replace(minute=0, second=0, microsecond=0)
    candidate = candidate.replace(hour=hour_utc)
    if candidate <= now_utc:
        candidate = candidate + timedelta(days=1)
    return candidate


class TrainingScheduler:
    """Daily candidate-training scheduler (never promotes anything)."""

    def __init__(
        self,
        db_factory: Callable,
        train_fn: Optional[Callable] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        bars_count: Optional[int] = None,
        horizon_bars: Optional[int] = None,
        algorithm: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        from backend.app.services.ml.training import train_candidate_from_market

        self.db_factory = db_factory
        self._train_fn = train_fn or train_candidate_from_market
        self.symbol = symbol or settings.WORKER_SYMBOL          # gold only: XAUUSD
        self.timeframe = timeframe or "M1"                       # gold only: M1
        self.bars_count = bars_count or 5000
        self.horizon_bars = horizon_bars or 15
        self.algorithm = algorithm or "logistic_regression"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._trained_on_date: Optional[str] = None

    def start(self) -> None:
        if not settings.TRAINING_SCHEDULE_ENABLED:
            logger.info("Training scheduler disabled (TRAINING_SCHEDULE_ENABLED=False).")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="training-scheduler", daemon=True)
        self._thread.start()
        logger.info(
            "Training scheduler started (daily at %02d:00 UTC, symbol=%s timeframe=%s; candidates only, never promotes).",
            settings.TRAINING_SCHEDULE_HOUR_UTC, self.symbol, self.timeframe,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = self._clock()
            next_run = compute_next_run(now, settings.TRAINING_SCHEDULE_HOUR_UTC)
            wait_seconds = max(0.0, (next_run - now).total_seconds())
            # Wake at most hourly so a stop request is always honored quickly;
            # _due() re-checks the schedule after every wake.
            if self._stop_event.wait(timeout=min(wait_seconds, 3600.0)):
                return  # stopped
            now = self._clock()
            self._maybe_train(now)

    def _due(self, now_utc: datetime) -> bool:
        """Runs once per UTC calendar day at/after the scheduled hour."""
        if now_utc.hour < settings.TRAINING_SCHEDULE_HOUR_UTC:
            return False
        today = now_utc.strftime("%Y-%m-%d")
        if self._trained_on_date == today:
            return False
        return True

    def _maybe_train(self, now_utc: datetime) -> None:
        if not self._due(now_utc):
            return
        self._trained_on_date = now_utc.strftime("%Y-%m-%d")
        self.run_training_once()

    def run_training_once(self) -> Optional[Dict[str, Any]]:
        """Trains one candidate now (used by the scheduler and by tests)."""
        started = datetime.now(timezone.utc).isoformat()
        try:
            with self.db_factory() as db:
                train_res = self._train_fn(
                    db,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    bars_count=self.bars_count,
                    horizon_bars=self.horizon_bars,
                    algorithm=self.algorithm,
                )
            last_training_status.update({
                "ran_at": started,
                "outcome": "trained",
                "model_id": train_res.get("model_id"),
                "detail": None,
            })
            logger.info(
                "Scheduled training finished: candidate %s created (NOT promoted).",
                train_res.get("model_id"),
            )
            return train_res
        except Exception as e:
            last_training_status.update({
                "ran_at": started,
                "outcome": "failed",
                "model_id": None,
                "detail": f"{type(e).__name__}: {e}",
            })
            logger.error("Scheduled training failed (retries next scheduled day): %s", e)
            return None
