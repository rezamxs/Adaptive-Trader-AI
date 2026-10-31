"""Application Configuration with strict Fail-Closed security.
Live trading is architecturally excluded and rejected at configuration validation.
"""
import math
from enum import Enum
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from backend.app.core.errors import LiveTradingProhibitedError, InvalidExecutionModeError


class ExecutionMode(str, Enum):
    """Permitted execution modes for Adaptive Trader AI.
    LIVE_TRADING is intentionally absent from this enumeration.
    """
    BACKTEST = "BACKTEST"
    SIGNAL_ONLY = "SIGNAL_ONLY"
    DEMO_TRADING = "DEMO_TRADING"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Core Application Info
    APP_NAME: str = "Adaptive Trader AI"
    APP_ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Execution Mode (Defaults strictly to SIGNAL_ONLY)
    EXECUTION_MODE: ExecutionMode = ExecutionMode.SIGNAL_ONLY

    # API Server
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    # Shared-secret token for state-changing endpoints. Fail-closed: if unset, a
    # process-start random secret is generated so no predictable default exists.
    # Read only from environment/.env; never hardcode a usable credential here.
    API_TOKEN: str = ""
    # HMAC key for metric/model-evidence digests (H1 + M8). Empty = plain
    # SHA-256 integrity digests (corruption detection only, NOT tamper-evident).
    # Settings-carried so a .env-provided value is honored too — metrics_integrity
    # reads os.environ first, then falls back to this field.
    METRIC_SIGNING_KEY: str = ""
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Database Settings
    # Absolute default: sqlite:///./ follows pytest's CWD and forks the database.
    MT5_PATH: str = r"C:\Program Files\<YourBroker> MT5\terminal64.exe"
    MT5_SERVER: str = "YourBroker-MT5"
    # (Removed dead keys: MT5_LOGIN / MT5_TIMEOUT_MS / MAX_CONSECUTIVE_EXECUTION_ERRORS
    # were referenced nowhere outside this file — an operator tuning
    # MAX_CONSECUTIVE_EXECUTION_ERRORS would have been configuring an inert
    # value. Re-add a key ONLY together with the code that consumes it.)
    # Measured clock offset of the CONFIGURED MT5_SERVER: YourBroker-MT5 stamps
    # broker server-time as epoch (verified live 2026-10-28 — ticks and M1
    # bars exactly +3h ahead of system UTC while updating every second).
    # Applied ONLY when the connected account's server equals MT5_SERVER; any
    # other server keeps raw-epoch (UTC) semantics. Live-consistency
    # validation fails closed on wrong-offset or ambiguous stamps.
    MT5_SERVER_TIME_OFFSET_HOURS: float = 3.0

    # Demo Trading Hard Gates
    DEMO_TRADING_ENABLED: bool = False
    MAX_SLIPPAGE_POINTS: int = 10

    # Optical Character Recognition (OCR) Feature Flag (Disabled by default)
    OCR_ENABLED: bool = False

    # Risk Management Constraints
    DEFAULT_RISK_PER_TRADE_PERCENT: float = 1.0
    MAX_OPEN_TRADES: int = 3
    # 3.0 vs 5.0: docs/strategy-specification.md (Group D §1) originally said
    # 5.0%; the configured default everywhere is 3.0%. The discrepancy is
    # documented in the spec and BLOCKED on an explicit owner decision —
    # changing a risk cap is a trading-policy decision, never a code fix.
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_DRAWDOWN_PERCENT: float = 6.0
    MAX_ALLOWED_SPREAD_POINTS: int = 35
    MIN_DATA_POINTS_REQUIRED: int = 100
    # Strategy Rules (Pages 14-15): enforce Friday-close protection and the
    # 00:00-09:30 UTC session window on every production risk assessment
    # (signal generation and demo dispatch). Wired through the risk-engine
    # call sites; tests may disable via this flag.
    ENFORCE_SESSION_FILTER: bool = True

    # Market Monitoring Worker Settings
    WORKER_ENABLED: bool = True
    WORKER_SYMBOL: str = "XAUUSD"
    WORKER_TIMEFRAME: str = "M1"
    WORKER_POLL_INTERVAL_SECONDS: float = 3.0
    WORKER_RETRY_BACKOFF_BASE: float = 2.0
    WORKER_MAX_RETRY_DELAY_SECONDS: float = 60.0
    WORKER_STALE_TICK_SECONDS: int = 120

    # Item 16: retention windows (days) for the operationally-growing tables,
    # applied once at startup via prune_operational_records. 0 = DISABLED —
    # nothing is ever deleted without the operator explicitly setting a
    # window, so audit history is never pruned by default.
    WORKER_CYCLE_RETENTION_DAYS: int = 0
    AUDIT_LOG_RETENTION_DAYS: int = 0

    # Database Settings
    # Absolute defaults: relative paths here would silently follow pytest's CWD.
    # BASE_DIR is the repo root; runtime DB and dataset artifacts live at the root.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    ARTIFACTS_MODELS_DIR: Path = BASE_DIR / "artifacts" / "models"
    ARTIFACTS_REPORTS_DIR: Path = BASE_DIR / "artifacts" / "reports"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'adaptive_trader.db'}"

    @field_validator("EXECUTION_MODE", mode="before")
    @classmethod
    def validate_execution_mode(cls, v: Union[str, ExecutionMode]) -> ExecutionMode:
        if isinstance(v, ExecutionMode):
            return v

        mode_str = str(v).strip().upper()
        if "LIVE" in mode_str or "REAL" in mode_str:
            raise LiveTradingProhibitedError(
                f"Configuration rejected: '{mode_str}' is forbidden. Live trading is strictly prohibited."
            )

        try:
            return ExecutionMode(mode_str)
        except ValueError:
            raise InvalidExecutionModeError(
                f"Invalid execution mode '{v}'. Allowed modes are: {[m.value for m in ExecutionMode]}"
            )

    @field_validator("MT5_SERVER_TIME_OFFSET_HOURS")
    @classmethod
    def validate_server_time_offset(cls, v: float) -> float:
        # NaN bypasses any `abs(v) > limit` comparison (all NaN comparisons
        # are False), so non-finite values must be rejected explicitly.
        if not math.isfinite(v) or abs(v) > 12.0:
            raise ValueError(
                f"MT5_SERVER_TIME_OFFSET_HOURS={v} is implausible; the offset "
                "must be a finite value with |offset| <= 12h."
            )
        return v


# Global settings instance
import secrets as _secrets

settings = Settings()

# Fail-closed token bootstrap: an empty configured token falls back to a random
# per-process secret so that "guess the default" never works. (API_TOKEN from
# the environment or .env already lands in the field via BaseSettings.)
# Tests and local runs should set API_TOKEN explicitly.
if not settings.API_TOKEN.strip():
    settings.API_TOKEN = _secrets.token_urlsafe(32)
