"""Regression tests for broker-aware position sizing."""
from backend.app.services.market_data.base import SymbolInfo
from backend.app.services.risk.position_sizer import PositionSizer


def _symbol_info(**overrides):
    values = dict(
        name="XAUUSD", digits=2, point=0.01, spread=20, tick_size=0.01,
        tick_value=1.0, volume_min=0.01, volume_max=10.0, volume_step=0.01,
        trade_contract_size=100.0, currency_base="XAU", currency_profit="USD",
        currency_margin="USD",
    )
    values.update(overrides)
    return SymbolInfo(**values)


def test_tick_value_is_used_for_risk():
    info = _symbol_info(tick_size=0.01, tick_value=2.0, trade_contract_size=100.0)
    ok, lots, reason = PositionSizer.calculate_and_verify_lots(
        equity=10000.0, entry_price=2650.0, stop_loss=2640.0,
        symbol_info=info, risk_percent=1.0, max_allowed_risk_percent=1.5,
    )
    assert ok
    assert lots == 0.05
    assert "Actual risk: $100.00" in reason


def test_volume_is_rounded_down_not_up_to_nearest_step():
    info = _symbol_info(tick_size=0.01, tick_value=1.0, trade_contract_size=100.0)
    ok, lots, reason = PositionSizer.calculate_and_verify_lots(
        equity=10000.0, entry_price=2700.0, stop_loss=2634.0,
        symbol_info=info, risk_percent=1.0, max_allowed_risk_percent=2.0,
    )
    assert ok
    assert lots == 0.01
    assert "Actual risk: $66.00" in reason



def test_max_volume_is_rounded_down_to_a_valid_broker_step():
    info = _symbol_info(volume_step=0.01, volume_min=0.01, volume_max=0.025)
    ok, lots, _ = PositionSizer.calculate_and_verify_lots(
        equity=100000.0, entry_price=2700.0, stop_loss=2699.0,
        symbol_info=info, risk_percent=1.0, max_allowed_risk_percent=2.0,
    )
    assert ok
    assert lots == 0.02
