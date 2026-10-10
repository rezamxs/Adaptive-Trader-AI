"""Application Configuration with strict Fail-Closed security.
Live trading is architecturally excluded and rejected at configuration validation.
"""
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
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Database Settings
    DATABASE_URL: str = "sqlite:///./adaptive_trader.db"

    # MetaTrader 5 Configuration
    MT5_PATH: str = r"C:\Program Files\<YourBroker> MT5\terminal64.exe"
    MT5_SERVER: str = "YourBroker-MT5"
    MT5_LOGIN: int = 0
    MT5_TIMEOUT_MS: int = 60000

    # Demo Trading Hard Gates
    DEMO_TRADING_ENABLED: bool = False
    MAX_SLIPPAGE_POINTS: int = 10

    # Risk Management Constraints
    DEFAULT_RISK_PER_TRADE_PERCENT: float = 1.0
    MAX_OPEN_TRADES: int = 3
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_DRAWDOWN_PERCENT: float = 6.0
    MAX_ALLOWED_SPREAD_POINTS: int = 35
    MIN_DATA_POINTS_REQUIRED: int = 100
    MAX_CONSECUTIVE_EXECUTION_ERRORS: int = 3

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_RAW_DIR: Path = Path("./data/raw")
    DATA_PROCESSED_DIR: Path = Path("./data/processed")
    ARTIFACTS_MODELS_DIR: Path = Path("./artifacts/models")
    ARTIFACTS_REPORTS_DIR: Path = Path("./artifacts/reports")

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


# Global settings instance
settings = Settings()
