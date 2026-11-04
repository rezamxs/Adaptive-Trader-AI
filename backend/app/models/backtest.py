"""BacktestRun Model for storing verifiable backtest results and trade records."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base, utc_now_naive


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    initial_balance: Mapped[float] = mapped_column(Float, default=10000.0, nullable=False)
    final_balance: Mapped[float] = mapped_column(Float, nullable=False)
    net_profit: Mapped[float] = mapped_column(Float, nullable=False)
    gross_profit: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    gross_loss: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    max_drawdown_percent: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown_amount: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    total_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    expectancy: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    average_trade: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Assumptions & Limitations
    assumptions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    limitations: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    
    # Detailed series
    equity_curve: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    trades_summary: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, nullable=False)
