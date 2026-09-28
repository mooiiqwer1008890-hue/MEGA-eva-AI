"""
paper_trader.py
===============
محاكي التداول الورقي (Paper Trading) محلي.
- لا يحتاج API أو Binance Testnet.
- يحفظ الصفقات في trades_log.csv.
- يحفظ الرصيد في paper_balance.txt.
- يستخدم إدارة المخاطر من risk_manager.py.
"""

import os
import csv
import json
import logging
from datetime import datetime

from risk_manager import calculate_kelly_position_size

log = logging.getLogger("paper_trader")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TRADES_LOG = "trades_log.csv"
BALANCE_FILE = "paper_balance.txt"

# إعدادات إدارة المخاطر
WIN_RATE = float(os.environ.get("WIN_RATE", "0.55"))
WIN_LOSS_RATIO = float(os.environ.get("WIN_LOSS_RATIO", "1.5"))
KELLY_FRACTION = float(os.environ.get("KELLY_FRACTION", "0.5"))
INITIAL_BALANCE = float(os.environ.get("BALANCE", "1000"))

# إعدادات الصفقة
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "0.02"))    # 2%
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.04")) # 4%
HOLD_CANDLES = int(os.environ.get("HOLD_CANDLES", "10"))


# ---------------------------------------------------------------------
# BALANCE MANAGEMENT
# ---------------------------------------------------------------------
def load_balance():
    """يقرأ الرصيد الحالي من الملف."""
    if os.path.exists(BALANCE_FILE):
        try:
            with open(BALANCE_FILE, "r") as f:
                return float(f.read().strip())
        except Exception as e:
            log.warning(f"Failed to read balance file: {e}")
    # إذا لم يوجد الملف، نبدأ بالرصيد الابتدائي
    save_balance(INITIAL_BALANCE)
    return INITIAL_BALANCE


def save_balance(balance):
    """يحفظ الرصيد الحالي في الملف."""
    with open(BALANCE_FILE, "w") as f:
        f.write(f"{balance:.4f}")


# ---------------------------------------------------------------------
# TRADES LOG
# ---------------------------------------------------------------------
def init_trades_log():
    """ينشئ ملف الصفقات إذا لم يكن موجوداً."""
    if not os.path.exists(TRADES_LOG):
        with open(TRADES_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_open", "timestamp_close", "symbol",
                "entry_price", "exit_price", "quantity",
                "position_size_usd", "pnl_usd", "pnl_pct",
                "balance_after", "exit_reason"
            ])


def append_trade(trade):
    """يضيف صفقة جديدة إلى ملف CSV."""
    init_trades_log()
    with open(TRADES_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            trade.get("timestamp_open", ""),
            trade.get("timestamp_close", ""),
            trade.get("symbol", ""),
            trade.get("entry_price", ""),
            trade.get("exit_price", ""),
            trade.get("quantity", ""),
            trade.get("position_size_usd", ""),
            trade.get("pnl_usd", ""),
            trade.get("pnl_pct", ""),
            trade.get("balance_after", ""),
            trade.get("exit_reason", ""),
        ])


# ---------------------------------------------------------------------
# POSITION MANAGEMENT
# ---------------------------------------------------------------------
def position_file(symbol):
    return f"paper_position_{symbol}.json"


def load_position(symbol):
    path = position_file(symbol)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_position(symbol, position):
    with open(position_file(symbol), "w") as f:
        json.dump(position, f)


def clear_position(symbol):
    path = position_file(symbol)
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------------------
def open_paper_position(symbol, price, z):
    """
    يفتح صفقة وهمية باستخدام Kelly Position Sizing.
    """
    balance = load_balance()
    position_size = calculate_kelly_position_size(
        WIN_RATE, WIN_LOSS_RATIO, balance, KELLY_FRACTION
    )
    if position_size <= 0:
        log.info(f"[PAPER:{symbol}] Kelly position size = 0. Skipping.")
        return None

    qty = position_size / price
    stop_loss = price * (1 - STOP_LOSS_PCT)
    take_profit = price * (1 + TAKE_PROFIT_PCT)

    position = {
        "symbol": symbol,
        "entry_price": price,
        "quantity": qty,
        "position_size_usd": position_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "opened_candle_count": 0,
        "timestamp_open": datetime.utcnow().isoformat(),
    }
    save_position(symbol, position)

    log.info(
        f"[PAPER:{symbol}] OPENED: entry={price:.4f} qty={qty:.6f} "
        f"size=${position_size:.2f} SL={stop_loss:.4f} TP={take_profit:.4f}"
    )
    return position


def check_and_close_position(symbol, price):
    """
    يتحقق إذا كانت الصفقة قد لمست Stop Loss أو Take Profit.
    إذا نعم، يغلقها ويسجل الصفقة.
    """
    position = load_position(symbol)
    if position is None:
        return None

    position["opened_candle_count"] += 1

    # التحقق من Stop Loss و Take Profit
    exit_reason = None
    if price <= position["stop_loss"]:
        exit_reason = "STOP_LOSS"
    elif price >= position["take_profit"]:
        exit_reason = "TAKE_PROFIT"
    elif position["opened_candle_count"] >= HOLD_CANDLES:
        exit_reason = "TIME_LIMIT"

    if exit_reason is None:
        # لا تزال مفتوحة
        save_position(symbol, position)
        return None

    # إغلاق الصفقة
    entry = position["entry_price"]
    qty = position["quantity"]
    exit_price = price

    pnl_usd = (exit_price - entry) * qty
    pnl_pct = (exit_price - entry) / entry * 100

    # تحديث الرصيد
    balance = load_balance()
    new_balance = balance + pnl_usd
    save_balance(new_balance)

    # تسجيل الصفقة
    trade = {
        "timestamp_open": position["timestamp_open"],
        "timestamp_close": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "entry_price": round(entry, 6),
        "exit_price": round(exit_price, 6),
        "quantity": round(qty, 8),
        "position_size_usd": round(position["position_size_usd"], 2),
        "pnl_usd": round(pnl_usd, 4),
        "pnl_pct": round(pnl_pct, 4),
        "balance_after": round(new_balance, 4),
        "exit_reason": exit_reason,
    }
    append_trade(trade)
    clear_position(symbol)

    log.info(
        f"[PAPER:{symbol}] CLOSED ({exit_reason}): "
        f"entry={entry:.4f} exit={exit_price:.4f} "
        f"pnl={pnl_pct:+.3f}% (${pnl_usd:+.4f}) balance=${new_balance:.2f}"
    )
    return trade


# ---------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------
def get_stats():
    """يقرأ ملف الصفقات ويحسب إحصائيات."""
    if not os.path.exists(TRADES_LOG):
        return None

    trades = []
    with open(TRADES_LOG, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                trades.append({
                    "pnl_usd": float(row["pnl_usd"]),
                    "pnl_pct": float(row["pnl_pct"]),
                    "exit_reason": row["exit_reason"],
                })
            except (ValueError, KeyError):
                continue

    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "net_pnl_usd": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "balance": load_balance(),
            "initial_balance": INITIAL_BALANCE,
            "total_return_pct": 0.0,
        }

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] < 0]
    net_pnl = sum(t["pnl_usd"] for t in trades)
    balance = load_balance()

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "net_pnl_usd": round(net_pnl, 4),
        "avg_win_usd": round(sum(t["pnl_usd"] for t in wins) / len(wins), 4) if wins else 0.0,
        "avg_loss_usd": round(sum(t["pnl_usd"] for t in losses) / len(losses), 4) if losses else 0.0,
        "balance": round(balance, 2),
        "initial_balance": INITIAL_BALANCE,
        "total_return_pct": round((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 4),
    }


# ---------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------
init_trades_log()
