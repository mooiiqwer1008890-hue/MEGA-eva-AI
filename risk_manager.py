"""
risk_manager.py
===============
نظام إدارة مخاطر احترافي.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("risk_manager")


def calculate_kelly_position_size(win_rate, win_loss_ratio, balance, fraction=0.5):
    """تحسب حجم الصفقة الأمثل باستخدام معيار كيلي (Half-Kelly)."""
    if win_rate <= 0 or win_rate >= 1:
        log.warning(f"win_rate غير صالح: {win_rate}")
        return 0.0
    if win_loss_ratio <= 0:
        log.warning(f"win_loss_ratio غير صالح: {win_loss_ratio}")
        return 0.0
    if balance <= 0:
        log.warning(f"balance غير صالح: {balance}")
        return 0.0
    
    f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    if f <= 0:
        log.info(f"Kelly f = {f:.4f} (سالب). لا تتداول.")
        return 0.0
    
    half_kelly = fraction * f
    position_size = balance * half_kelly
    
    log.info(f"Kelly: win_rate={win_rate:.2%}, ratio={win_loss_ratio:.2f}, f={f:.4f}, pos=${position_size:.2f}")
    return round(position_size, 2)


def calculate_var(returns, confidence=0.95, portfolio_value=10000):
    """تحسب قيمة المعرض للخطر (VaR)."""
    if len(returns) == 0:
        return 0.0
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = norm.ppf(1 - confidence, mu, sigma)
    var = portfolio_value - portfolio_value * (alpha + 1)
    
    return round(var, 2)


def calculate_max_drawdown(equity_curve):
    """تحسب أقصى تراجع ومدة التراجع."""
    if len(equity_curve) == 0:
        return (0.0, 0)
    
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    drawdown_duration = (drawdown < 0).sum()
    
    return (round(max_drawdown, 4), int(drawdown_duration))


def calculate_sharpe_ratio(returns, periods=252, risk_free_rate=0.02):
    """تحسب نسبة شارب."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods
    std = excess_returns.std()
    if std == 0:
        return 0.0
    
    sharpe = (excess_returns.mean() / std) * np.sqrt(periods)
    return round(sharpe, 4)


def calculate_sortino_ratio(returns, periods=252, risk_free_rate=0.02):
    """تحسب نسبة سورتينو."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods)
    return round(sortino, 4)


def generate_performance_report(equity_curve, trades_df):
    """تولد تقريراً كاملاً بالأداء."""
    if len(equity_curve) < 2:
        return {}
    if 'pnl' not in trades_df.columns:
        return {}
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    returns = equity_curve.pct_change().dropna()
    
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, dd_duration = calculate_max_drawdown(equity_curve)
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
    
    if avg_loss != 0 and not np.isnan(avg_loss):
        win_loss_ratio = abs(avg_win / avg_loss)
    else:
        win_loss_ratio = 1.0
    
    net_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
    
    report = {
        'total_return': round(total_return, 4),
        'net_profit': round(net_profit, 2),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'drawdown_duration': dd_duration,
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'win_loss_ratio': round(win_loss_ratio, 4),
    }
    
    log.info("=" * 60)
    log.info("PERFORMANCE REPORT")
    for key, value in report.items():
        log.info(f"  {key}: {value}")
    log.info("=" * 60)
    
    return report
