"""
risk_manager.py
===============
نظام إدارة مخاطر وتحليل أداء للتداول.

المبادئ:
- يحافظ على الدوال القديمة لتجنب كسر الاستيرادات الحالية.
- يدعم Kelly بحذر مع حد أقصى للحجم.
- يضيف تحديد حجم الصفقة حسب المخاطرة ووقف الخسارة.
- يدعم Parametric VaR و Historical VaR و CVaR/Expected Shortfall.
- يصحح حساب مدة الـDrawdown لتكون أطول فترة متصلة تحت القمة.
- يضيف Profit Factor و Expectancy و Calmar Ratio وبعض مؤشرات الجودة.
- ينظف القيم غير الصالحة NaN/Inf بدل السماح لها بتلويث التقرير.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("risk_manager")


Number = Union[int, float, np.number]
SeriesLike = Union[pd.Series, np.ndarray, Iterable[Number], list, tuple]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _clean_returns(returns: SeriesLike) -> np.ndarray:
    """تحول returns إلى مصفوفة رقمية نظيفة بدون NaN/Inf."""
    if isinstance(returns, pd.Series):
        values = returns.to_numpy(dtype=float)
    else:
        values = np.asarray(list(returns) if not isinstance(returns, np.ndarray) else returns, dtype=float)

    values = values.reshape(-1)
    values = values[np.isfinite(values)]
    return values


def _clean_equity_curve(equity_curve: SeriesLike) -> np.ndarray:
    """تحول منحنى رأس المال إلى مصفوفة رقمية نظيفة."""
    if isinstance(equity_curve, pd.Series):
        values = equity_curve.to_numpy(dtype=float)
    else:
        values = np.asarray(
            list(equity_curve) if not isinstance(equity_curve, np.ndarray) else equity_curve,
            dtype=float,
        )

    values = values.reshape(-1)
    values = values[np.isfinite(values)]
    return values


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence يجب أن تكون بين 0 و1، مثل 0.95 أو 0.99.")


def _safe_round(value: Number, digits: int = 4) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return round(value, digits)


# -----------------------------------------------------------------------------
# Position sizing
# -----------------------------------------------------------------------------

def calculate_kelly_fraction(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> float:
    """
    تحسب نسبة رأس المال وفق Kelly Fractional Kelly.

    win_loss_ratio = متوسط الربح / القيمة المطلقة لمتوسط الخسارة.
    fraction=0.5 يعني Half-Kelly.
    max_fraction يحمي النظام من الأحجام المفرطة الناتجة عن تقديرات تاريخية غير مستقرة.
    """
    if not 0.0 < win_rate < 1.0:
        log.warning("win_rate غير صالح: %s", win_rate)
        return 0.0

    if win_loss_ratio <= 0 or not np.isfinite(win_loss_ratio):
        log.warning("win_loss_ratio غير صالح: %s", win_loss_ratio)
        return 0.0

    if not 0.0 < fraction <= 1.0:
        log.warning("fraction غير صالح: %s", fraction)
        return 0.0

    if not 0.0 < max_fraction <= 1.0:
        log.warning("max_fraction غير صالح: %s", max_fraction)
        return 0.0

    # Kelly: f* = (p*b - q) / b
    p = float(win_rate)
    q = 1.0 - p
    b = float(win_loss_ratio)

    kelly = (p * b - q) / b

    if not np.isfinite(kelly) or kelly <= 0.0:
        log.info("Kelly <= 0؛ لا يوجد edge إحصائي كافٍ: %.6f", kelly)
        return 0.0

    fractional = kelly * fraction
    capped = min(fractional, max_fraction)

    if fractional > max_fraction:
        log.warning(
            "Kelly fraction %.2f%% تجاوز الحد %.2f%%؛ تم القص.",
            fractional * 100.0,
            max_fraction * 100.0,
        )

    return round(capped, 6)


def calculate_kelly_position_size(
    win_rate: float,
    win_loss_ratio: float,
    balance: float,
    fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> float:
    """تحسب قيمة المركز بالدولار/العملة الأساسية باستخدام Fractional Kelly."""
    if balance <= 0 or not np.isfinite(balance):
        log.warning("balance غير صالح: %s", balance)
        return 0.0

    kelly_fraction = calculate_kelly_fraction(
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        fraction=fraction,
        max_fraction=max_fraction,
    )

    position_size = balance * kelly_fraction
    log.info(
        "Kelly: win_rate=%.2f%%, ratio=%.3f, fraction=%.4f, pos=%.2f",
        win_rate * 100.0,
        win_loss_ratio,
        kelly_fraction,
        position_size,
    )
    return round(position_size, 2)


def calculate_position_size_by_risk(
    balance: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float = 0.01,
    max_position_pct: float = 0.25,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> Dict[str, float]:
    """
    يحدد حجم الصفقة من المخاطرة الفعلية عند وقف الخسارة.

    مناسب أكثر من Kelly عندما يكون الهدف:
    "لا أريد خسارة أكثر من X% من رأس المال إذا ضرب وقف الخسارة."

    يفترض صفقة Spot/Long واحدة وبدون رافعة.
    """
    result = {
        "risk_cash": 0.0,
        "risk_per_unit": 0.0,
        "quantity": 0.0,
        "position_value": 0.0,
        "position_pct": 0.0,
    }

    if balance <= 0 or not np.isfinite(balance):
        return result

    if entry_price <= 0 or stop_price <= 0:
        return result

    if stop_price >= entry_price:
        log.warning(
            "لصفقة Spot/Long يجب أن يكون stop_price أقل من entry_price."
        )
        return result

    if not 0 < risk_per_trade <= 0.10:
        log.warning("risk_per_trade غير منطقي: %s", risk_per_trade)
        return result

    if not 0 < max_position_pct <= 1.0:
        log.warning("max_position_pct غير صالح: %s", max_position_pct)
        return result

    if fee_rate < 0 or slippage_rate < 0:
        return result

    risk_cash = balance * risk_per_trade
    max_position_value = balance * max_position_pct

    # خسارة السعر عند الوقف + تكلفة تقريبية للدخول والخروج.
    price_loss_per_unit = entry_price - stop_price
    transaction_cost_per_unit = (
        entry_price * (fee_rate + slippage_rate)
        + stop_price * (fee_rate + slippage_rate)
    )

    risk_per_unit = price_loss_per_unit + transaction_cost_per_unit

    if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
        return result

    quantity_by_risk = risk_cash / risk_per_unit
    quantity_by_cap = max_position_value / entry_price
    quantity = min(quantity_by_risk, quantity_by_cap)

    position_value = quantity * entry_price
    position_pct = position_value / balance

    result.update(
        {
            "risk_cash": round(risk_cash, 8),
            "risk_per_unit": round(risk_per_unit, 8),
            "quantity": round(quantity, 8),
            "position_value": round(position_value, 8),
            "position_pct": round(position_pct, 6),
        }
    )

    return result


# -----------------------------------------------------------------------------
# VaR / CVaR
# -----------------------------------------------------------------------------

def calculate_var(
    returns: SeriesLike,
    confidence: float = 0.95,
    portfolio_value: float = 10000.0,
    method: str = "parametric",
) -> float:
    """
    تحسب VaR كقيمة خسارة موجبة.

    method:
    - parametric: يفترض تقريباً توزيعاً طبيعياً.
    - historical: يعتمد مباشرة على التوزيع التاريخي للعوائد.
    """
    _validate_confidence(confidence)

    if portfolio_value <= 0:
        return 0.0

    x = _clean_returns(returns)
    if len(x) < 2:
        return 0.0

    method = method.lower().strip()

    if method == "historical":
        quantile = float(np.quantile(x, 1.0 - confidence))
        var = max(0.0, -quantile * portfolio_value)

    elif method == "parametric":
        mu = float(np.mean(x))
        sigma = float(np.std(x, ddof=1))

        if sigma <= 0 or not np.isfinite(sigma):
            return max(0.0, -mu * portfolio_value)

        z = float(norm.ppf(1.0 - confidence))
        loss_quantile = mu + z * sigma
        var = max(0.0, -loss_quantile * portfolio_value)

    else:
        raise ValueError("method يجب أن يكون 'parametric' أو 'historical'.")

    return round(var, 2)


def calculate_cvar(
    returns: SeriesLike,
    confidence: float = 0.95,
    portfolio_value: float = 10000.0,
    method: str = "historical",
) -> float:
    """
    Expected Shortfall / CVaR كمتوسط الخسائر التي تقع أسوأ من VaR.

    افتراض: returns عوائد نسبية مثل 0.01 = +1%.
    """
    _validate_confidence(confidence)

    if portfolio_value <= 0:
        return 0.0

    x = _clean_returns(returns)
    if len(x) < 2:
        return 0.0

    method = method.lower().strip()

    if method == "historical":
        threshold = float(np.quantile(x, 1.0 - confidence))
        tail = x[x <= threshold]

        if len(tail) == 0:
            return 0.0

        cvar = -float(np.mean(tail)) * portfolio_value
        return round(max(0.0, cvar), 2)

    if method == "parametric":
        mu = float(np.mean(x))
        sigma = float(np.std(x, ddof=1))

        if sigma <= 0 or not np.isfinite(sigma):
            return max(0.0, round(-mu * portfolio_value, 2))

        alpha = 1.0 - confidence
        z_alpha = float(norm.ppf(alpha))

        # Expected value in the lower tail of N(mu, sigma)
        tail_mean = mu - sigma * float(norm.pdf(z_alpha)) / alpha
        cvar = max(0.0, -tail_mean * portfolio_value)
        return round(cvar, 2)

    raise ValueError("method يجب أن يكون 'historical' أو 'parametric'.")


# -----------------------------------------------------------------------------
# Drawdown
# -----------------------------------------------------------------------------

def calculate_max_drawdown(
    equity_curve: SeriesLike,
) -> Tuple[float, int]:
    """
    يحسب:
    - max_drawdown كنسبة سالبة، مثل -0.1234 = -12.34%.
    - أطول مدة متصلة تحت قمة سابقة، بعدد الفترات.
    """
    equity = _clean_equity_curve(equity_curve)

    if len(equity) == 0:
        return 0.0, 0

    # منحنى رأس مال موجب ضروري لنسبة drawdown ذات معنى.
    if np.any(equity <= 0):
        log.warning("equity_curve يحتوي قيم <= 0؛ تم إهمالها.")
        equity = equity[equity > 0]

    if len(equity) == 0:
        return 0.0, 0

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    max_drawdown = float(np.min(drawdown))

    # أطول عدد فترات متتالية drawdown < 0.
    longest_duration = 0
    current_duration = 0

    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            longest_duration = max(longest_duration, current_duration)
        else:
            current_duration = 0

    return round(max_drawdown, 6), int(longest_duration)


def calculate_recovery_factor(
    equity_curve: SeriesLike,
) -> float:
    """Recovery Factor = net profit / absolute max drawdown بالدولار."""
    equity = _clean_equity_curve(equity_curve)

    if len(equity) < 2 or equity[0] <= 0:
        return 0.0

    max_dd, _ = calculate_max_drawdown(equity)
    running_max = np.maximum.accumulate(equity)
    max_dd_value = float(np.max(running_max - equity))

    if max_dd_value <= 0:
        return 0.0

    net_profit = float(equity[-1] - equity[0])
    return round(net_profit / max_dd_value, 4)


# -----------------------------------------------------------------------------
# Risk-adjusted performance
# -----------------------------------------------------------------------------

def calculate_sharpe_ratio(
    returns: SeriesLike,
    periods: int = 252,
    risk_free_rate: float = 0.02,
) -> float:
    """تحسب Sharpe Ratio مع annualization."""
    x = _clean_returns(returns)

    if len(x) < 2 or periods <= 0:
        return 0.0

    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess_returns = x - periodic_rf

    std = float(np.std(excess_returns, ddof=1))
    if std <= 0 or not np.isfinite(std):
        return 0.0

    sharpe = (float(np.mean(excess_returns)) / std) * np.sqrt(periods)
    return round(sharpe, 4)


def calculate_sortino_ratio(
    returns: SeriesLike,
    periods: int = 252,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Sortino باستخدام downside deviation من العوائد تحت معدل العائد الخالي من المخاطر.
    """
    x = _clean_returns(returns)

    if len(x) < 2 or periods <= 0:
        return 0.0

    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess_returns = x - periodic_rf
    downside = np.minimum(excess_returns, 0.0)

    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if downside_deviation <= 0 or not np.isfinite(downside_deviation):
        return 0.0

    sortino = (float(np.mean(excess_returns)) / downside_deviation) * np.sqrt(periods)
    return round(sortino, 4)


def calculate_calmar_ratio(
    equity_curve: SeriesLike,
    periods_per_year: int = 252,
) -> float:
    """
    Calmar تقريبي:
    معدل النمو السنوي المركب / absolute max drawdown.
    """
    equity = _clean_equity_curve(equity_curve)

    if len(equity) < 2 or equity[0] <= 0 or equity[-1] <= 0:
        return 0.0

    max_dd, _ = calculate_max_drawdown(equity)
    if max_dd >= 0:
        return 0.0

    years = max((len(equity) - 1) / max(periods_per_year, 1), 1e-9)
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0

    return round(cagr / abs(max_dd), 4)


# -----------------------------------------------------------------------------
# Trade statistics
# -----------------------------------------------------------------------------

def calculate_trade_statistics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """يستخرج إحصائيات الصفقات من عمود pnl."""
    if trades_df is None or not isinstance(trades_df, pd.DataFrame):
        return {}

    if "pnl" not in trades_df.columns:
        return {}

    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce")
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()

    if len(pnl) == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "break_even_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_consecutive_losses": 0,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    flat = pnl[pnl == 0]

    total = len(pnl)
    win_rate = len(wins) / total if total else 0.0

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf if gross_profit > 0 else 0.0

    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
    else:
        win_loss_ratio = np.inf if avg_win > 0 else 0.0

    expectancy = float(pnl.mean())

    # أطول سلسلة خسائر متتالية.
    max_consecutive_losses = 0
    current_losses = 0
    for value in pnl:
        if value < 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0

    return {
        "total_trades": int(total),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "break_even_trades": int(len(flat)),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "win_loss_ratio": round(float(win_loss_ratio), 4) if np.isfinite(win_loss_ratio) else float("inf"),
        "profit_factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else float("inf"),
        "expectancy": round(expectancy, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "max_consecutive_losses": int(max_consecutive_losses),
    }


# -----------------------------------------------------------------------------
# Complete report
# -----------------------------------------------------------------------------

def generate_performance_report(
    equity_curve: SeriesLike,
    trades_df: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.02,
    var_confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    يولد تقريراً شاملاً للأداء والمخاطرة.

    يعيد {} عندما تكون البيانات الأساسية غير كافية.
    """
    equity = _clean_equity_curve(equity_curve)

    if len(equity) < 2:
        return {}

    if trades_df is None or "pnl" not in trades_df.columns:
        return {}

    if equity[0] <= 0 or equity[-1] <= 0:
        return {}

    returns = pd.Series(equity).pct_change().dropna()
    trade_stats = calculate_trade_statistics(trades_df)

    total_return = equity[-1] / equity[0] - 1.0
    net_profit = equity[-1] - equity[0]

    sharpe = calculate_sharpe_ratio(
        returns,
        periods=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    sortino = calculate_sortino_ratio(
        returns,
        periods=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    max_dd, dd_duration = calculate_max_drawdown(equity)

    initial_capital = float(equity[0])
    var_parametric = calculate_var(
        returns,
        confidence=var_confidence,
        portfolio_value=initial_capital,
        method="parametric",
    )
    var_historical = calculate_var(
        returns,
        confidence=var_confidence,
        portfolio_value=initial_capital,
        method="historical",
    )
    cvar_historical = calculate_cvar(
        returns,
        confidence=var_confidence,
        portfolio_value=initial_capital,
        method="historical",
    )

    calmar = calculate_calmar_ratio(
        equity,
        periods_per_year=periods_per_year,
    )
    recovery_factor = calculate_recovery_factor(equity)

    report: Dict[str, Any] = {
        "total_return": round(float(total_return), 6),
        "net_profit": round(float(net_profit), 4),
        "ending_equity": round(float(equity[-1]), 4),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "drawdown_duration": dd_duration,
        "recovery_factor": recovery_factor,
        "var_parametric": var_parametric,
        "var_95_parametric": var_parametric if var_confidence == 0.95 else calculate_var(
            returns, confidence=0.95, portfolio_value=initial_capital, method="parametric"
        ),
        "var_historical": var_historical,
        "cvar_historical": cvar_historical,
        **trade_stats,
    }

    # نسخ محسوبة للحفاظ على المفتاح القديم.
    # إذا أردت confidence مختلفاً، قيمة var_95_parametric تعكس confidence المستخدم.
    report["var_confidence"] = var_confidence

    log.info("=" * 72)
    log.info("PERFORMANCE / RISK REPORT")
    for key, value in report.items():
        log.info("  %-24s: %s", key, value)
    log.info("=" * 72)

    return report


# -----------------------------------------------------------------------------
# Optional portfolio exposure helper
# -----------------------------------------------------------------------------

def calculate_portfolio_exposure(
    balance: float,
    position_values: Iterable[float],
) -> Dict[str, float]:
    """
    يحسب إجمالي التعرض الاسمي للمراكز.

    لا يسمح بقيم سالبة/غير منتهية.
    """
    if balance <= 0:
        return {
            "gross_exposure": 0.0,
            "exposure_pct": 0.0,
        }

    values = np.asarray(list(position_values), dtype=float)
    values = values[np.isfinite(values)]
    values = np.abs(values)

    gross_exposure = float(values.sum())
    exposure_pct = gross_exposure / balance

    return {
        "gross_exposure": round(gross_exposure, 8),
        "exposure_pct": round(exposure_pct, 6),
    }
