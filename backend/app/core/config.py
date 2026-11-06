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
    # Buglist item 43: SQL statement echo was tied to DEBUG — enabling debug
    # mode logged EVERY query (with the data inside) to the log. SQL echo is
    # now its own switch, default off.
    SQL_ECHO: bool = False
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
    # OWNER DECISION (2026-11-01): the daily loss cap is 5.0%. The historical
    # 3.0-vs-5.0 discrepancy (docs/strategy-specification.md originally said
    # 5.0%, code defaulted to 3.0%) is resolved toward 5.0% by explicit owner
    # instruction — changing a risk cap is a trading-policy decision, never a
    # code fix.
    MAX_DAILY_LOSS_PERCENT: float = 5.0
    MAX_DRAWDOWN_PERCENT: float = 6.0
    MAX_ALLOWED_SPREAD_POINTS: int = 35
    MIN_DATA_POINTS_REQUIRED: int = 100
    # Strategy Rules (Pages 14-15): enforce Friday-close protection and the
    # 00:00-09:30 UTC session window on every production risk assessment
    # (signal generation and demo dispatch). Wired through the risk-engine
    # call sites; tests may disable via this flag.
    ENFORCE_SESSION_FILTER: bool = True

    # Step 3.4 (owner request 2026-11-01): scheduled automatic candidate
    # training. The scheduler trains ONE candidate per UTC day at the given
    # hour and NEVER promotes — activation stays a human endpoint action.
    TRAINING_SCHEDULE_ENABLED: bool = True
    TRAINING_SCHEDULE_HOUR_UTC: int = 21

    # Buglist item 4: the shared training flow (HTTP /ml/train AND the nightly
    # scheduler) produces per-fold-retrained ML walk-forward evidence for each
    # new candidate (persisted into ml_walk_forward_runs + attached to the
    # candidate). This activates the promotion gate that previously had no
    # producer. Best-effort: on failure the evidence is skipped with an
    # explicit reason in the result, never breaking the training itself.
    # 0 folds disables the producer entirely.
    ML_WALK_FORWARD_FOLDS: int = 4

    # Buglist item 18: per-minute call cap on the heavy CPU endpoints
    # (POST /backtest/run and POST /ml/train), keyed by API token. 0 disables
    # the limiter (tests may rely on that).
    HEAVY_RATE_LIMIT_PER_MINUTE: int = 10

    # Buglist item 33: the probability threshold live inference filters on.
    # The ML walk-forward EVIDENCE is evaluated at the fixed default 0.5
    # (MLWF_THRESHOLD); this live policy value is deliberately separate and
    # is now RECORDED with every model (record column + attested artifact
    # payload) instead of living only as a hidden function default. Changing
    # it is a trading-policy decision, not a code fix.
    LIVE_MODEL_PROB_THRESHOLD: float = 0.55

    # Buglist item 17: TTL cache (seconds) for the trading governance probe
    # behind the PUBLIC /health endpoint — every poll previously deserialized
    # the active model artifact from disk. 0 disables the cache.
    HEALTH_PROBE_CACHE_SECONDS: float = 20.0

    # Market Monitoring Worker Settings
    WORKER_ENABLED: bool = True
    WORKER_SYMBOL: str = "XAUUSD"
    WORKER_TIMEFRAME: str = "M1"
    WORKER_POLL_INTERVAL_SECONDS: float = 3.0
    WORKER_RETRY_BACKOFF_BASE: float = 2.0
    WORKER_MAX_RETRY_DELAY_SECONDS: float = 60.0
    WORKER_STALE_TICK_SECONDS: int = 120

    # Retention windows (days) for the operationally-growing tables, applied
    # once at startup via prune_operational_records. OWNER DECISION
    # (2026-11-01): 30 days for all four tables. A monthly summary is
    # archived before any pruning pass deletes rows. 0 = DISABLED for a
    # single table (nothing is deleted for it).
    WORKER_CYCLE_RETENTION_DAYS: int = 30
    AUDIT_LOG_RETENTION_DAYS: int = 30
    TRADE_SIGNAL_RETENTION_DAYS: int = 30
    DEMO_ORDER_RETENTION_DAYS: int = 30

    # Minor item 7: SQLite busy_timeout (ms). Writers wait up to this long for
    # the database lock instead of failing instantly with "database is locked"
    # when the API and the worker collide on the single-writer file.
    DB_BUSY_TIMEOUT_MS: int = 5000

    # Minor item 16: file-log rotation (logs/app.log + .1..N). An always-on
    # worker must not grow the log file without bound.
    LOG_ROTATE_MAX_BYTES: int = 5 * 1024 * 1024
    LOG_ROTATE_BACKUP_COUNT: int = 3

    # Database Settings
    # Absolute defaults: relative paths here would silently follow pytest's CWD.
    # BASE_DIR is the repo root; runtime DB and dataset artifacts live at the root.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    ARTIFACTS_MODELS_DIR: Path = BASE_DIR / "artifacts" / "models"
    ARTIFACTS_REPORTS_DIR: Path = BASE_DIR / "artifacts" / "reports"
    # Step 3.1 (owner request 2026-11-01): documents dropped into this folder
    # are ingested automatically at application startup (same transactional
    # pipeline as the upload endpoint; processed/failed files are moved aside).
    DOCUMENT_INBOX_DIR: Path = BASE_DIR / "data" / "document_inbox"
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

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        """Buglist item 42: the app always sends CORS with
        allow_credentials=True, and a wildcard origin is INCOMPATIBLE with
        credentials (browsers reject it; starlette mirrors the request origin
        and would reflect ANY origin as allowed). Reject the combination at
        configuration time instead of shipping a silently unsafe policy."""
        cleaned = [str(origin).strip() for origin in v if str(origin).strip()]
        if "*" in cleaned:
            raise ValueError(
                "CORS_ORIGINS must not contain '*': the app serves CORS with "
                "allow_credentials=True, which a wildcard origin makes "
                "meaningless and unsafe. List explicit origins."
            )
        return cleaned

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
