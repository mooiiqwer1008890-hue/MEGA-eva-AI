"""
paper_trader.py  (FIXED)
=========================
التعديل الجوهري عن النسخة السابقة:

المشكلة الأصلية: كان `opened_candle_count` يُزاد بمقدار 1 في كل استدعاء
لـ check_and_close_position، بافتراض أن الدالة تُستدعى مرة واحدة فقط لكل
شمعة (15 دقيقة). لكن هذا افتراض هش ينهار بمجرد وجود أكثر من دورة تشغيل
(دورة سريعة كل 5 دقائق + دورة كاملة كل 30 دقيقة) - وهو بالضبط وضعنا
الحالي.

الحل: بدل عدّ "مرات الاستدعاء"، نقيس "الوقت الفعلي المنقضي" منذ
timestamp_open (وهو مُسجَّل بالفعل). هذا يجعل الدالة آمنة تماماً
للاستدعاء بأي تردد تريده - كل 5 دقائق، كل دقيقة، لا فرق - والنتيجة
تبقى صحيحة دائماً لأنها مبنية على الساعة الحقيقية لا على عدد المكالمات.

النتيجة العملية: يمكن الآن استدعاء check_and_close_position من الدورة
السريعة (كل 5 دقائق) لفحص وقف الخسارة/جني الربح بشكل فعلي متجاوب،
بدل انتظار الدورة الكاملة كل 30 دقيقة كما كان يحدث سابقاً.
"""

import os
import csv
import json
import logging
import re
from datetime import datetime, timezone

from risk_manager import calculate_kelly_position_size

log = logging.getLogger("paper_trader")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TRADES_LOG = "trades_log.csv"
BALANCE_FILE = "paper_balance.txt"

WIN_RATE = float(os.environ.get("WIN_RATE", "0.55"))
WIN_LOSS_RATIO = float(os.environ.get("WIN_LOSS_RATIO", "1.5"))
KELLY_FRACTION = float(os.environ.get("KELLY_FRACTION", "0.5"))
INITIAL_BALANCE = float(os.environ.get("BALANCE", "1000"))

STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.04"))
HOLD_CANDLES = int(os.environ.get("HOLD_CANDLES", "10"))


def _interval_to_minutes(interval_str):
    """يحوّل '15m' -> 15, '1h' -> 60, '4h' -> 240, '1d' -> 1440."""
    m = re.match(r"(\d+)([mhd])", interval_str.strip())
    if not m:
        log.warning(f"Unrecognized INTERVAL='{interval_str}', defaulting to 15 minutes.")
        return 15
    n, unit = int(m.group(1)), m.group(2)
    return n * {"m": 1, "h": 60, "d": 1440}[unit]


INTERVAL_MINUTES = _interval_to_minutes(os.environ.get("INTERVAL", "15m"))
HOLD_MINUTES = HOLD_CANDLES * INTERVAL_MINUTES  # e.g. 10 * 15 = 150 minutes

_USING_PLACEHOLDER_KELLY = (
    os.environ.get("WIN_RATE") is None and os.environ.get("WIN_LOSS_RATIO") is None
)
if _USING_PLACEHOLDER_KELLY:
    log.warning(
        "⚠️ WIN_RATE و WIN_LOSS_RATIO غير مضبوطين — يُستخدم الآن 0.55 و1.5 "
        "كقيم افتراضية وهمية، وليستا مقاسة من Backtest حقيقي. حجم الصفقة "
        "الناتج عن Kelly غير موثوق حتى تُستبدل هذه القيم بنتائج backtest_zscore.py "
        "الفعلية لكل عملة."
    )


# ---------------------------------------------------------------------
# BALANCE MANAGEMENT (unchanged)
# ---------------------------------------------------------------------
def load_balance():
    if os.path.exists(BALANCE_FILE):
        try:
            with open(BALANCE_FILE, "r") as f:
                return float(f.read().strip())
        except Exception as e:
            log.warning(f"Failed to read balance file: {e}")
    save_balance(INITIAL_BALANCE)
    return INITIAL_BALANCE


def save_balance(balance):
    with open(BALANCE_FILE, "w") as f:
        f.write(f"{balance:.4f}")


# ---------------------------------------------------------------------
# TRADES LOG (unchanged)
# ---------------------------------------------------------------------
def init_trades_log():
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
    init_trades_log()
    with open(TRADES_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            trade.get("timestamp_open", ""), trade.get("timestamp_close", ""),
            trade.get("symbol", ""), trade.get("entry_price", ""),
            trade.get("exit_price", ""), trade.get("quantity", ""),
            trade.get("position_size_usd", ""), trade.get("pnl_usd", ""),
            trade.get("pnl_pct", ""), trade.get("balance_after", ""),
            trade.get("exit_reason", ""),
        ])


# ---------------------------------------------------------------------
# POSITION MANAGEMENT (unchanged)
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
        # NOTE: opened_candle_count kept for backward-compatible display
        # only — it no longer drives the exit decision. timestamp_open
        # (wall-clock) is the sole source of truth for the time-based exit.
        "opened_candle_count": 0,
        "timestamp_open": datetime.now(timezone.utc).isoformat(),
    }
    save_position(symbol, position)

    log.info(
        f"[PAPER:{symbol}] OPENED: entry={price:.4f} qty={qty:.6f} "
        f"size=${position_size:.2f} SL={stop_loss:.4f} TP={take_profit:.4f} "
        f"time_exit_after={HOLD_MINUTES}min"
    )
    return position


def check_and_close_position(symbol, price):
    """
    يفحص Stop Loss / Take Profit / انتهاء الوقت. آمن للاستدعاء بأي تردد
    (كل 5 دقائق أو حتى كل دقيقة) لأن قرار الوقت مبني على الساعة الحقيقية.
    """
    position = load_position(symbol)
    if position is None:
        return None

    opened_at = datetime.fromisoformat(position["timestamp_open"])
    if opened_at.tzinfo is None:  # توافق مع صفقات قديمة سُجلت بدون timezone
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    elapsed_minutes = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60.0

    exit_reason = None
    if price <= position["stop_loss"]:
        exit_reason = "STOP_LOSS"
    elif price >= position["take_profit"]:
        exit_reason = "TAKE_PROFIT"
    elif elapsed_minutes >= HOLD_MINUTES:
        exit_reason = "TIME_LIMIT"

    if exit_reason is None:
        # لا تزال مفتوحة — لا حاجة لإعادة الحفظ، لم يتغيّر شيء دائم
        return None

    entry = position["entry_price"]
    qty = position["quantity"]
    exit_price = price

    pnl_usd = (exit_price - entry) * qty
    pnl_pct = (exit_price - entry) / entry * 100

    balance = load_balance()
    new_balance = balance + pnl_usd
    save_balance(new_balance)

    trade = {
        "timestamp_open": position["timestamp_open"],
        "timestamp_close": datetime.now(timezone.utc).isoformat(),
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
        f"[PAPER:{symbol}] CLOSED ({exit_reason} after {elapsed_minutes:.1f}min): "
        f"entry={entry:.4f} exit={exit_price:.4f} "
        f"pnl={pnl_pct:+.3f}% (${pnl_usd:+.4f}) balance=${new_balance:.2f}"
    )
    return trade


# ---------------------------------------------------------------------
# STATS (unchanged)
# ---------------------------------------------------------------------
def get_stats():
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
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "net_pnl_usd": 0.0, "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0, "balance": load_balance(),
            "initial_balance": INITIAL_BALANCE, "total_return_pct": 0.0,
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


init_trades_log()
