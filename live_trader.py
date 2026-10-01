"""
live_trader.py
==============

Professional Spot Trading Orchestrator for Binance.

المسؤوليات:
-----------
1. استقبال الإشارة من طبقة الاستراتيجية.
2. التحقق من صلاحية السوق قبل التنفيذ.
3. حساب حجم الصفقة من المخاطرة وليس من نسبة ثابتة من الرصيد.
4. فرض حدود:
      - Risk per trade
      - Maximum position
      - Maximum daily loss
      - Maximum open positions
      - Maximum aggregate risk
      - Maximum signal staleness
5. تنفيذ Market BUY/SELL عبر BinanceClient.
6. الاعتماد على fills الفعلية لحساب:
      - executed quantity
      - average execution price
      - quote value
      - fees
7. إنشاء حماية Stop Loss بعد الدخول.
8. مراقبة Take Profit.
9. تخزين الحالة في SQLite بدل JSON.
10. إعادة تحميل الحالة بعد إعادة تشغيل Railway.
11. إعادة حساب Daily PnL من دفتر الصفقات.
12. عدم إعادة إرسال أمر بشكل أعمى بعد timeout.
13. منع تداول الرمز نفسه أكثر من مرة في الوقت نفسه.
14. حفظ signal metadata مثل Z-Score دون جعل هذا الملف استراتيجية بحد ذاته.

مهم:
-----
هذا الملف لا يحتوي على Alpha Model.

لا يوجد هنا:
    Z-Score strategy
    HMM
    GARCH
    Triple Barrier
    Meta Labeling
    ML prediction

هذه الطبقات يجب أن تنتج:
    direction
    confidence
    entry reference
    stop loss
    take profit

ثم تسلم القرار إلى LiveTrader.

المعمارية:

Market Data
     ↓
Feature Engineering
     ↓
Regime Detection
     ↓
Signal / Z-Score
     ↓
Meta Labeling
     ↓
Risk Manager
     ↓
LiveTrader   ← هذا الملف
     ↓
BinanceClient
     ↓
Binance Spot


مراجع تقنية:
--------------
- Binance Spot REST API
- Binance Spot Order Lifecycle
- Binance Spot Exchange Filters
- Binance Spot User Data / Execution Reports
- مبادئ Position Sizing وRisk Budgeting
- مبادئ Execution Risk وSlippage Control

ملاحظة:
--------
هذه النسخة تعمل فوق BinanceClient المطور سابقًا.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from binance_client import BinanceClient


# ============================================================================
# Logging
# ============================================================================

log = logging.getLogger("live_trader")


# ============================================================================
# Decimal helper
# ============================================================================

def D(value: Any) -> Decimal:
    """
    Safe Decimal conversion.
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid numeric value: {value!r}"
        ) from exc


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_iso(value: str) -> datetime:
    """
    Parse stored ISO-8601 timestamp.
    """
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


# ============================================================================
# Configuration
# ============================================================================

LIVE_MODE = (
    os.getenv(
        "LIVE_MODE",
        "false",
    ).strip().lower()
    == "true"
)

USE_TESTNET = (
    os.getenv(
        "USE_TESTNET",
        "true",
    ).strip().lower()
    == "true"
)

# ---------------------------------------------------------------------------
# Hard capital limits
# ---------------------------------------------------------------------------

MAX_POSITION_USD = D(
    os.getenv(
        "MAX_POSITION_USD",
        "50",
    )
)

MAX_DAILY_LOSS_USD = D(
    os.getenv(
        "MAX_DAILY_LOSS_USD",
        "20",
    )
)

MAX_OPEN_POSITIONS = int(
    os.getenv(
        "MAX_OPEN_POSITIONS",
        "2",
    )
)

# ---------------------------------------------------------------------------
# Risk-based position sizing
# ---------------------------------------------------------------------------

RISK_PER_TRADE_PCT = D(
    os.getenv(
        "RISK_PER_TRADE_PCT",
        "0.50",
    )
)

MAX_BALANCE_USAGE_PCT = D(
    os.getenv(
        "MAX_BALANCE_USAGE_PCT",
        "90",
    )
)

MAX_TOTAL_RISK_PCT = D(
    os.getenv(
        "MAX_TOTAL_RISK_PCT",
        "2.0",
    )
)

# ---------------------------------------------------------------------------
# Stop / reward model
# ---------------------------------------------------------------------------

DEFAULT_STOP_LOSS_PCT = D(
    os.getenv(
        "DEFAULT_STOP_LOSS_PCT",
        "2.0",
    )
)

DEFAULT_REWARD_RISK = D(
    os.getenv(
        "DEFAULT_REWARD_RISK",
        "2.0",
    )
)

MIN_REWARD_RISK = D(
    os.getenv(
        "MIN_REWARD_RISK",
        "1.5",
    )
)

# ---------------------------------------------------------------------------
# Signal quality / freshness
# ---------------------------------------------------------------------------

MAX_SIGNAL_DEVIATION_BPS = D(
    os.getenv(
        "MAX_SIGNAL_DEVIATION_BPS",
        "50",
    )
)

MAX_SIGNAL_AGE_SECONDS = int(
    os.getenv(
        "MAX_SIGNAL_AGE_SECONDS",
        "300",
    )
)

# ---------------------------------------------------------------------------
# Position maintenance
# ---------------------------------------------------------------------------

MAX_HOLD_MINUTES = int(
    os.getenv(
        "MAX_HOLD_MINUTES",
        "1440",
    )
)

SELL_BALANCE_BUFFER_PCT = D(
    os.getenv(
        "SELL_BALANCE_BUFFER_PCT",
        "0.10",
    )
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

STATE_DB = os.getenv(
    "TRADER_STATE_DB",
    "trader_state.sqlite3",
)

# ---------------------------------------------------------------------------
# Pair configuration
# ---------------------------------------------------------------------------

QUOTE_ASSET = os.getenv(
    "QUOTE_ASSET",
    "USDT",
).upper()


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class Position:
    symbol: str
    quantity: Decimal

    entry_price: Decimal
    entry_quote_cost: Decimal
    entry_fee_usd: Decimal

    stop_loss: Decimal
    take_profit: Decimal

    entry_order_id: Optional[int]
    stop_order_id: Optional[int]

    entry_time: str

    zscore: Optional[Decimal]
    signal_id: Optional[str]

    status: str = "OPEN"


@dataclass
class Execution:
    symbol: str
    order_id: int
    status: str

    executed_qty: Decimal
    quote_qty: Decimal
    average_price: Decimal

    fee_usd: Decimal
    base_fee_qty: Decimal


# ============================================================================
# LiveTrader
# ============================================================================

class LiveTrader:
    """
    Spot risk + execution supervisor.

    Testnet is the default.

    Live trading requires BOTH:
        LIVE_MODE=true
        BinanceClient's explicit live-trading confirmation
    """

    def __init__(self) -> None:

        # ------------------------------------------------------------------
        # Safety gate
        # ------------------------------------------------------------------

        if LIVE_MODE and USE_TESTNET:

            log.warning(
                "[LIVE] LIVE_MODE=true ولكن USE_TESTNET=true. "
                "سيتم التداول على Testnet وليس الحساب الحقيقي."
            )

        self.client = BinanceClient(
            testnet=USE_TESTNET,
            dry_run=not LIVE_MODE,
        )

        # ------------------------------------------------------------------
        # Concurrency protection
        # ------------------------------------------------------------------

        self._lock = threading.RLock()

        # ------------------------------------------------------------------
        # Persistent state
        # ------------------------------------------------------------------

        self.db = sqlite3.connect(
            STATE_DB,
            check_same_thread=False,
            timeout=30,
        )

        self.db.row_factory = sqlite3.Row

        self._initialize_database()

        # ------------------------------------------------------------------
        # Startup reconciliation
        # ------------------------------------------------------------------

        self._startup_report()

        environment = (
            "TESTNET"
            if USE_TESTNET
            else "LIVE"
        )

        log.info(
            "[LIVE] Engine ready | "
            "Environment=%s | "
            "Execution=%s | "
            "MaxPosition=$%s | "
            "DailyLoss=$%s | "
            "Risk/Trade=%s%% | "
            "MaxOpen=%s",
            environment,
            "ACTIVE" if LIVE_MODE else "DRY_RUN",
            MAX_POSITION_USD,
            MAX_DAILY_LOSS_USD,
            RISK_PER_TRADE_PCT,
            MAX_OPEN_POSITIONS,
        )

    # ======================================================================
    # Database
    # ======================================================================

    def _initialize_database(self) -> None:

        with self._lock:

            cursor = self.db.cursor()

            cursor.execute(
                """
                PRAGMA journal_mode=WAL
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,

                    quantity TEXT NOT NULL,

                    entry_price TEXT NOT NULL,
                    entry_quote_cost TEXT NOT NULL,
                    entry_fee_usd TEXT NOT NULL,

                    stop_loss TEXT NOT NULL,
                    take_profit TEXT NOT NULL,

                    entry_order_id INTEGER,
                    stop_order_id INTEGER,

                    entry_time TEXT NOT NULL,

                    zscore TEXT,
                    signal_id TEXT,

                    status TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,

                    order_id INTEGER,

                    quantity TEXT NOT NULL,
                    average_price TEXT NOT NULL,
                    quote_qty TEXT NOT NULL,

                    fee_usd TEXT NOT NULL,

                    pnl_usd TEXT NOT NULL,

                    reason TEXT,
                    timestamp TEXT NOT NULL,

                    UNIQUE(symbol, side, order_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_positions_status
                ON positions(status)
                """
            )

            self.db.commit()

    # ======================================================================
    # Startup
    # ======================================================================

    def _startup_report(self) -> None:

        positions = self._load_positions()

        daily_pnl = self.get_daily_realized_pnl()

        log.info(
            "[LIVE] Startup state | "
            "Positions=%s | "
            "DailyRealizedPnL=$%s",
            len(positions),
            daily_pnl,
        )

        for position in positions:

            log.info(
                "[LIVE] Restored position | "
                "%s | qty=%s | entry=%s | SL=%s | TP=%s",
                position.symbol,
                position.quantity,
                position.entry_price,
                position.stop_loss,
                position.take_profit,
            )

    # ======================================================================
    # Position persistence
    # ======================================================================

    def _load_positions(self) -> List[Position]:

        rows = self.db.execute(
            """
            SELECT *
            FROM positions
            WHERE status = 'OPEN'
            ORDER BY entry_time
            """
        ).fetchall()

        result: List[Position] = []

        for row in rows:

            result.append(
                Position(
                    symbol=row["symbol"],
                    quantity=D(row["quantity"]),

                    entry_price=D(row["entry_price"]),
                    entry_quote_cost=D(
                        row["entry_quote_cost"]
                    ),
                    entry_fee_usd=D(
                        row["entry_fee_usd"]
                    ),

                    stop_loss=D(row["stop_loss"]),
                    take_profit=D(row["take_profit"]),

                    entry_order_id=(
                        int(row["entry_order_id"])
                        if row["entry_order_id"] is not None
                        else None
                    ),

                    stop_order_id=(
                        int(row["stop_order_id"])
                        if row["stop_order_id"] is not None
                        else None
                    ),

                    entry_time=row["entry_time"],

                    zscore=(
                        D(row["zscore"])
                        if row["zscore"] is not None
                        else None
                    ),

                    signal_id=row["signal_id"],

                    status=row["status"],
                )
            )

        return result

    def _get_position(
        self,
        symbol: str,
    ) -> Optional[Position]:

        row = self.db.execute(
            """
            SELECT *
            FROM positions
            WHERE symbol = ?
              AND status = 'OPEN'
            """,
            (symbol.upper(),),
        ).fetchone()

        if row is None:
            return None

        return Position(
            symbol=row["symbol"],
            quantity=D(row["quantity"]),

            entry_price=D(row["entry_price"]),
            entry_quote_cost=D(
                row["entry_quote_cost"]
            ),
            entry_fee_usd=D(
                row["entry_fee_usd"]
            ),

            stop_loss=D(row["stop_loss"]),
            take_profit=D(row["take_profit"]),

            entry_order_id=(
                int(row["entry_order_id"])
                if row["entry_order_id"] is not None
                else None
            ),

            stop_order_id=(
                int(row["stop_order_id"])
                if row["stop_order_id"] is not None
                else None
            ),

            entry_time=row["entry_time"],

            zscore=(
                D(row["zscore"])
                if row["zscore"] is not None
                else None
            ),

            signal_id=row["signal_id"],

            status=row["status"],
        )

    def _insert_position(
        self,
        position: Position,
    ) -> None:

        self.db.execute(
            """
            INSERT OR REPLACE INTO positions (
                symbol,
                quantity,

                entry_price,
                entry_quote_cost,
                entry_fee_usd,

                stop_loss,
                take_profit,

                entry_order_id,
                stop_order_id,

                entry_time,

                zscore,
                signal_id,

                status
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                position.symbol,
                str(position.quantity),

                str(position.entry_price),
                str(position.entry_quote_cost),
                str(position.entry_fee_usd),

                str(position.stop_loss),
                str(position.take_profit),

                position.entry_order_id,
                position.stop_order_id,

                position.entry_time,

                (
                    str(position.zscore)
                    if position.zscore is not None
                    else None
                ),

                position.signal_id,

                position.status,
            ),
        )

        self.db.commit()

    def _update_stop_order(
        self,
        symbol: str,
        stop_order_id: Optional[int],
    ) -> None:

        self.db.execute(
            """
            UPDATE positions
            SET stop_order_id = ?
            WHERE symbol = ?
            """,
            (
                stop_order_id,
                symbol.upper(),
            ),
        )

        self.db.commit()

    def _delete_position(
        self,
        symbol: str,
    ) -> None:

        self.db.execute(
            """
            DELETE FROM positions
            WHERE symbol = ?
            """,
            (symbol.upper(),),
        )

        self.db.commit()

    # ======================================================================
    # Trade ledger
    # ======================================================================

    def _record_trade(
        self,
        *,
        symbol: str,
        side: str,
        order_id: int,
        quantity: Decimal,
        average_price: Decimal,
        quote_qty: Decimal,
        fee_usd: Decimal,
        pnl_usd: Decimal,
        reason: str,
    ) -> None:

        self.db.execute(
            """
            INSERT OR IGNORE INTO trades (
                symbol,
                side,
                order_id,

                quantity,
                average_price,
                quote_qty,

                fee_usd,
                pnl_usd,

                reason,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.upper(),
                side.upper(),
                order_id,

                str(quantity),
                str(average_price),
                str(quote_qty),

                str(fee_usd),
                str(pnl_usd),

                reason,
                iso_now(),
            ),
        )

        self.db.commit()

    # ======================================================================
    # Daily PnL
    # ======================================================================

    def get_daily_realized_pnl(self) -> Decimal:

        now = utc_now()

        start = datetime(
            now.year,
            now.month,
            now.day,
            tzinfo=timezone.utc,
        )

        row = self.db.execute(
            """
            SELECT COALESCE(
                SUM(CAST(pnl_usd AS REAL)),
                0
            ) AS pnl
            FROM trades
            WHERE side = 'SELL'
              AND timestamp >= ?
            """,
            (start.isoformat(),),
        ).fetchone()

        return D(row["pnl"] or 0)

    # ======================================================================
    # Unrealized PnL
    # ======================================================================

    def get_unrealized_pnl(self) -> Decimal:

        total = D("0")

        for position in self._load_positions():

            try:

                current_price = self.client.get_price_decimal(
                    position.symbol
                )

                mark_value = (
                    position.quantity
                    * current_price
                )

                estimated = (
                    mark_value
                    - position.entry_quote_cost
                    - position.entry_fee_usd
                )

                total += estimated

            except Exception as exc:

                log.error(
                    "[LIVE] Unrealized PnL failed for %s: %s",
                    position.symbol,
                    exc,
                )

        return total

    # ======================================================================
    # Risk calculations
    # ======================================================================

    def _aggregate_open_risk(self) -> Decimal:

        total_risk = D("0")

        for position in self._load_positions():

            risk_per_unit = (
                position.entry_price
                - position.stop_loss
            )

            if risk_per_unit <= 0:
                continue

            total_risk += (
                risk_per_unit
                * position.quantity
            )

        return total_risk

    def _daily_loss_guard(self) -> Tuple[bool, str]:

        realized = self.get_daily_realized_pnl()

        unrealized = self.get_unrealized_pnl()

        combined = (
            realized
            + unrealized
        )

        if combined <= -MAX_DAILY_LOSS_USD:

            return (
                False,
                (
                    "Daily loss limit reached: "
                    f"realized=${realized:.4f}, "
                    f"unrealized=${unrealized:.4f}, "
                    f"combined=${combined:.4f}"
                ),
            )

        return True, "OK"

    def _can_trade(
        self,
        symbol: str,
    ) -> Tuple[bool, str]:

        symbol = symbol.upper()

        # --------------------------------------------------------------
        # Global daily loss protection
        # --------------------------------------------------------------

        allowed, reason = self._daily_loss_guard()

        if not allowed:
            return False, reason

        # --------------------------------------------------------------
        # Duplicate position protection
        # --------------------------------------------------------------

        if self._get_position(symbol):

            return (
                False,
                f"{symbol} already has an open position",
            )

        # --------------------------------------------------------------
        # Maximum number of positions
        # --------------------------------------------------------------

        positions = self._load_positions()

        if len(positions) >= MAX_OPEN_POSITIONS:

            return (
                False,
                (
                    "Maximum open positions reached: "
                    f"{MAX_OPEN_POSITIONS}"
                ),
            )

        # --------------------------------------------------------------
        # Balance
        # --------------------------------------------------------------

        usdt = self.client.get_balance_decimal(
            QUOTE_ASSET
        )

        if usdt <= 0:

            return (
                False,
                "No available quote balance",
            )

        # --------------------------------------------------------------
        # Aggregate portfolio risk
        # --------------------------------------------------------------

        max_total_risk = (
            usdt
            * MAX_TOTAL_RISK_PCT
            / D("100")
        )

        current_risk = self._aggregate_open_risk()

        if current_risk >= max_total_risk:

            return (
                False,
                (
                    "Aggregate portfolio risk limit reached: "
                    f"${current_risk:.4f} >= "
                    f"${max_total_risk:.4f}"
                ),
            )

        return True, "OK"

    # ======================================================================
    # Signal freshness
    # ======================================================================

    def _validate_signal_age(
        self,
        signal_timestamp: Optional[str],
    ) -> None:

        if not signal_timestamp:
            return

        signal_time = parse_iso(
            signal_timestamp
        )

        age = (
            utc_now()
            - signal_time
        ).total_seconds()

        if age < 0:

            raise ValueError(
                "Signal timestamp is in the future."
            )

        if age > MAX_SIGNAL_AGE_SECONDS:

            raise ValueError(
                f"Signal is stale: "
                f"{age:.1f}s > "
                f"{MAX_SIGNAL_AGE_SECONDS}s"
            )

    # ======================================================================
    # Signal price validation
    # ======================================================================

    def _validate_signal_price(
        self,
        symbol: str,
        signal_price: Optional[Any],
    ) -> Decimal:

        snapshot = self.client.get_market_snapshot(
            symbol
        )

        current_reference = snapshot.ask

        if signal_price is None:

            return current_reference

        signal_price_dec = D(
            signal_price
        )

        if signal_price_dec <= 0:

            raise ValueError(
                "Invalid signal price."
            )

        deviation_bps = (
            abs(
                current_reference
                - signal_price_dec
            )
            / signal_price_dec
        ) * D("10000")

        if deviation_bps > MAX_SIGNAL_DEVIATION_BPS:

            raise ValueError(
                (
                    f"Signal price is stale: "
                    f"deviation={deviation_bps:.2f} bps "
                    f"> {MAX_SIGNAL_DEVIATION_BPS:.2f} bps"
                )
            )

        return current_reference

    # ======================================================================
    # Risk-based position sizing
    # ======================================================================

    def _calculate_position_size(
        self,
        *,
        symbol: str,
        entry_price: Decimal,
        stop_loss: Decimal,
        usdt_balance: Decimal,
    ) -> Decimal:

        if stop_loss >= entry_price:

            raise ValueError(
                (
                    f"Long stop must be below entry: "
                    f"entry={entry_price}, "
                    f"stop={stop_loss}"
                )
            )

        stop_distance = (
            entry_price
            - stop_loss
        )

        if stop_distance <= 0:

            raise ValueError(
                "Stop distance must be positive."
            )

        # --------------------------------------------------------------
        # Risk budget
        #
        # Example:
        #
        # balance = $100
        # risk = 0.5%
        #
        # risk budget = $0.50
        # --------------------------------------------------------------

        risk_budget = (
            usdt_balance
            * RISK_PER_TRADE_PCT
            / D("100")
        )

        # --------------------------------------------------------------
        # Quantity allowed by stop-loss risk.
        # --------------------------------------------------------------

        quantity_by_risk = (
            risk_budget
            / stop_distance
        )

        # --------------------------------------------------------------
        # Position notional cap.
        # --------------------------------------------------------------

        quantity_by_position_cap = (
            MAX_POSITION_USD
            / entry_price
        )

        # --------------------------------------------------------------
        # Available balance cap.
        # --------------------------------------------------------------

        usable_quote = (
            usdt_balance
            * MAX_BALANCE_USAGE_PCT
            / D("100")
        )

        quantity_by_balance = (
            usable_quote
            / entry_price
        )

        quantity = min(
            quantity_by_risk,
            quantity_by_position_cap,
            quantity_by_balance,
        )

        if quantity <= 0:

            raise ValueError(
                "Calculated position quantity <= 0."
            )

        # --------------------------------------------------------------
        # Verify aggregate risk.
        # --------------------------------------------------------------

        current_risk = (
            self._aggregate_open_risk()
        )

        max_total_risk = (
            usdt_balance
            * MAX_TOTAL_RISK_PCT
            / D("100")
        )

        remaining_risk = (
            max_total_risk
            - current_risk
        )

        if remaining_risk <= 0:

            raise ValueError(
                "No remaining aggregate portfolio risk budget."
            )

        quantity_by_portfolio_risk = (
            remaining_risk
            / stop_distance
        )

        quantity = min(
            quantity,
            quantity_by_portfolio_risk,
        )

        return self.client.normalize_quantity(
            symbol=symbol,
            quantity=quantity,
            market=True,
        )

    # ======================================================================
    # Parse execution
    # ======================================================================

    def _parse_execution(
        self,
        symbol: str,
        order: Dict[str, Any],
    ) -> Execution:

        order_id = int(
            order["orderId"]
        )

        status = str(
            order.get(
                "status",
                "",
            )
        )

        executed_qty = D(
            order.get(
                "executedQty",
                "0",
            )
        )

        quote_qty = D(
            order.get(
                "cummulativeQuoteQty",
                "0",
            )
        )

        fills = order.get(
            "fills",
            [],
        )

        if (
            executed_qty <= 0
            and fills
        ):

            executed_qty = sum(
                (
                    D(
                        fill.get(
                            "qty",
                            "0",
                        )
                    )
                    for fill in fills
                ),
                D("0"),
            )

        if (
            quote_qty <= 0
            and fills
        ):

            quote_qty = sum(
                (
                    D(
                        fill.get(
                            "price",
                            "0",
                        )
                    )
                    *
                    D(
                        fill.get(
                            "qty",
                            "0",
                        )
                    )
                    for fill in fills
                ),
                D("0"),
            )

        if executed_qty <= 0:

            raise ValueError(
                (
                    f"Order {order_id} "
                    f"has no executed quantity."
                )
            )

        if quote_qty <= 0:

            raise ValueError(
                (
                    f"Order {order_id} "
                    f"has no quote quantity."
                )
            )

        average_price = (
            quote_qty
            / executed_qty
        )

        base_asset = symbol.upper()

        if not base_asset.endswith(
            QUOTE_ASSET
        ):

            raise ValueError(
                (
                    f"Symbol {symbol} does not end with "
                    f"configured quote asset {QUOTE_ASSET}"
                )
            )

        base_asset = base_asset[
            :-len(QUOTE_ASSET)
        ]

        fee_usd = D("0")
        base_fee_qty = D("0")

        for fill in fills:

            commission = D(
                fill.get(
                    "commission",
                    "0",
                )
            )

            commission_asset = str(
                fill.get(
                    "commissionAsset",
                    "",
                )
            ).upper()

            if commission <= 0:
                continue

            # ----------------------------------------------------------
            # Fee paid in quote asset.
            # ----------------------------------------------------------

            if commission_asset == QUOTE_ASSET:

                fee_usd += commission

            # ----------------------------------------------------------
            # Fee paid in base asset.
            #
            # For BUY:
            # actual received position is reduced by this quantity.
            #
            # For SELL:
            # quote proceeds are not reduced by base commission,
            # but the base asset balance is.
            # ----------------------------------------------------------

            elif commission_asset == base_asset:

                base_fee_qty += commission

            # ----------------------------------------------------------
            # Fee paid in another asset (e.g. BNB).
            #
            # Convert it to quote currency when possible.
            # ----------------------------------------------------------

            else:

                fee_symbol = (
                    f"{commission_asset}"
                    f"{QUOTE_ASSET}"
                )

                try:

                    fee_price = (
                        self.client.get_price_decimal(
                            fee_symbol
                        )
                    )

                    fee_usd += (
                        commission
                        * fee_price
                    )

                except Exception as exc:

                    log.warning(
                        "[LIVE] Could not convert "
                        "fee asset %s to %s: %s",
                        commission_asset,
                        QUOTE_ASSET,
                        exc,
                    )

                    # Conservative fallback:
                    # use configured execution fee estimate.
                    fallback_bps = D(
                        os.getenv(
                            "UNKNOWN_FEE_BPS",
                            "20",
                        )
                    )

                    fee_usd += (
                        quote_qty
                        * fallback_bps
                        / D("10000")
                    )

        return Execution(
            symbol=symbol.upper(),
            order_id=order_id,
            status=status,

            executed_qty=executed_qty,
            quote_qty=quote_qty,
            average_price=average_price,

            fee_usd=fee_usd,
            base_fee_qty=base_fee_qty,
        )

    # ======================================================================
    # Wait for market order
    # ======================================================================

    def _ensure_filled(
        self,
        symbol: str,
        order: Dict[str, Any],
    ) -> Dict[str, Any]:

        status = order.get(
            "status"
        )

        if status == "FILLED":

            return order

        order_id = order.get(
            "orderId"
        )

        if order_id is None:

            raise RuntimeError(
                "Order has no orderId."
            )

        final_order = self.client.wait_for_order(
            symbol=symbol,
            order_id=int(order_id),
            timeout_seconds=30,
            poll_seconds=0.5,
        )

        final_status = final_order.get(
            "status"
        )

        if final_status != "FILLED":

            raise RuntimeError(
                (
                    f"Order {order_id} "
                    f"did not reach FILLED state: "
                    f"{final_status}"
                )
            )

        return final_order

    # ======================================================================
    # Open position
    # ======================================================================

    def open_position(
        self,
        symbol: str,
        z: Optional[Any] = None,
        price: Optional[Any] = None,
        stop_loss: Optional[Any] = None,
        take_profit: Optional[Any] = None,
        signal_timestamp: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        with self._lock:

            # ----------------------------------------------------------
            # Safety gate
            # ----------------------------------------------------------

            if not LIVE_MODE:

                log.info(
                    "[LIVE:%s] Trading disabled "
                    "(LIVE_MODE=false) — no order sent.",
                    symbol,
                )

                return {
                    "status": "DISABLED",
                    "symbol": symbol,
                }

            # ----------------------------------------------------------
            # Eligibility
            # ----------------------------------------------------------

            allowed, reason = self._can_trade(
                symbol
            )

            if not allowed:

                log.warning(
                    "[LIVE:%s] ⛔ Blocked: %s",
                    symbol,
                    reason,
                )

                return {
                    "status": "BLOCKED",
                    "symbol": symbol,
                    "reason": reason,
                }

            # ----------------------------------------------------------
            # Signal age
            # ----------------------------------------------------------

            self._validate_signal_age(
                signal_timestamp
            )

            # ----------------------------------------------------------
            # Current market reference
            # ----------------------------------------------------------

            current_reference = (
                self._validate_signal_price(
                    symbol,
                    price,
                )
            )

            # ----------------------------------------------------------
            # Stop loss
            #
            # Preferred:
            #   generated by Risk Manager / ATR / volatility model
            #
            # Fallback:
            #   percentage stop
            # ----------------------------------------------------------

            if stop_loss is None:

                stop_loss_dec = (
                    current_reference
                    * (
                        D("1")
                        - (
                            DEFAULT_STOP_LOSS_PCT
                            / D("100")
                        )
                    )
                )

            else:

                stop_loss_dec = D(
                    stop_loss
                )

            stop_loss_dec = (
                self.client.normalize_price(
                    symbol,
                    stop_loss_dec,
                )
            )

            if stop_loss_dec >= current_reference:

                raise ValueError(
                    (
                        f"Invalid stop loss: "
                        f"stop={stop_loss_dec}, "
                        f"entryReference={current_reference}"
                    )
                )

            # ----------------------------------------------------------
            # Take profit
            # ----------------------------------------------------------

            risk_distance = (
                current_reference
                - stop_loss_dec
            )

            if take_profit is None:

                take_profit_dec = (
                    current_reference
                    + (
                        risk_distance
                        * DEFAULT_REWARD_RISK
                    )
                )

            else:

                take_profit_dec = D(
                    take_profit
                )

            take_profit_dec = (
                self.client.normalize_price(
                    symbol,
                    take_profit_dec,
                )
            )

            reward_distance = (
                take_profit_dec
                - current_reference
            )

            if reward_distance <= 0:

                raise ValueError(
                    "Take profit must be above entry for a long position."
                )

            reward_risk = (
                reward_distance
                / risk_distance
            )

            if reward_risk < MIN_REWARD_RISK:

                raise ValueError(
                    (
                        f"Reward/Risk too low: "
                        f"{reward_risk:.3f} "
                        f"< minimum {MIN_REWARD_RISK}"
                    )
                )

            # ----------------------------------------------------------
            # Balance
            # ----------------------------------------------------------

            usdt_balance = (
                self.client.get_balance_decimal(
                    QUOTE_ASSET
                )
            )

            if usdt_balance <= 0:

                raise RuntimeError(
                    "No available USDT balance."
                )

            # ----------------------------------------------------------
            # Market quality
            # ----------------------------------------------------------

            diagnostics = (
                self.client.execution_diagnostics(
                    symbol=symbol,
                    quantity=self.client.normalize_quantity(
                        symbol,
                        D("1"),
                        market=True,
                    ),
                    side="BUY",
                )
            )

            if not diagnostics.get(
                "market_safe",
                False,
            ):

                raise RuntimeError(
                    (
                        "Market execution safety check failed: "
                        f"{diagnostics}"
                    )
                )

            # ----------------------------------------------------------
            # Risk-based quantity
            # ----------------------------------------------------------

            quantity = self._calculate_position_size(
                symbol=symbol,
                entry_price=current_reference,
                stop_loss=stop_loss_dec,
                usdt_balance=usdt_balance,
            )

            # ----------------------------------------------------------
            # Exact notional check.
            # ----------------------------------------------------------

            estimated_notional = (
                quantity
                * current_reference
            )

            if estimated_notional > MAX_POSITION_USD:

                quantity = (
                    self.client.normalize_quantity(
                        symbol,
                        MAX_POSITION_USD
                        / current_reference,
                        market=True,
                    )
                )

            if quantity <= 0:

                raise ValueError(
                    "Final position quantity <= 0."
                )

            log.info(
                "[LIVE:%s] Preparing entry | "
                "reference=%s | qty=%s | "
                "SL=%s | TP=%s | RR=%s | "
                "risk≈$%s",
                symbol,
                current_reference,
                quantity,
                stop_loss_dec,
                take_profit_dec,
                reward_risk,
                quantity
                * (
                    current_reference
                    - stop_loss_dec
                ),
            )

            # ----------------------------------------------------------
            # Execute BUY
            # ----------------------------------------------------------

            order = self.client.place_market_buy(
                symbol=symbol,
                quantity=quantity,
            )

            if not order:

                raise RuntimeError(
                    "Binance returned an empty order response."
                )

            order = self._ensure_filled(
                symbol,
                order,
            )

            execution = self._parse_execution(
                symbol=symbol,
                order=order,
            )

            # ----------------------------------------------------------
            # Actual position quantity.
            #
            # If fee is paid in base asset, remove it from quantity.
            # ----------------------------------------------------------

            actual_position_qty = (
                execution.executed_qty
                - execution.base_fee_qty
            )

            if actual_position_qty <= 0:

                raise RuntimeError(
                    "Actual received position quantity <= 0."
                )

            actual_entry_price = (
                execution.average_price
            )

            # Recalculate risk levels around actual fill.
            #
            # This prevents a stale reference price from determining
            # the final recorded position economics.
            # ----------------------------------------------------------

            stop_distance = (
                actual_entry_price
                - stop_loss_dec
            )

            if stop_distance <= 0:

                # Entry moved through the planned stop.
                # We must not leave an unprotected position.
                log.critical(
                    "[LIVE:%s] Entry moved through stop. "
                    "Emergency exit initiated.",
                    symbol,
                )

                emergency_order = (
                    self.client.place_market_sell(
                        symbol=symbol,
                        quantity=actual_position_qty
                        * (
                            D("1")
                            - (
                                SELL_BALANCE_BUFFER_PCT
                                / D("100")
                            )
                        ),
                    )
                )

                raise RuntimeError(
                    (
                        "Position entered through invalid "
                        "stop geometry; emergency exit sent. "
                        f"order={emergency_order}"
                    )
                )

            # ----------------------------------------------------------
            # Record position BEFORE placing protective stop.
            #
            # This is intentional:
            # if the process dies after the BUY, the state is still
            # recoverable and can be reconciled.
            # ----------------------------------------------------------

            position = Position(
                symbol=symbol,
                quantity=actual_position_qty,

                entry_price=actual_entry_price,
                entry_quote_cost=execution.quote_qty,
                entry_fee_usd=execution.fee_usd,

                stop_loss=stop_loss_dec,
                take_profit=take_profit_dec,

                entry_order_id=execution.order_id,
                stop_order_id=None,

                entry_time=iso_now(),

                zscore=(
                    D(z)
                    if z is not None
                    else None
                ),

                signal_id=signal_id,

                status="OPEN",
            )

            self._insert_position(
                position
            )

            # ----------------------------------------------------------
            # Protective stop
            # ----------------------------------------------------------

            try:

                stop_order = (
                    self.client.place_stop_loss(
                        symbol=symbol,
                        quantity=(
                            actual_position_qty
                            * (
                                D("1")
                                - (
                                    SELL_BALANCE_BUFFER_PCT
                                    / D("100")
                                )
                            )
                        ),
                        stop_price=stop_loss_dec,
                    )
                )

                stop_order_id = stop_order.get(
                    "orderId"
                )

                if stop_order_id is None:

                    raise RuntimeError(
                        "Protective stop returned no orderId."
                    )

                self._update_stop_order(
                    symbol=symbol,
                    stop_order_id=int(
                        stop_order_id
                    ),
                )

                position.stop_order_id = int(
                    stop_order_id
                )

            except Exception as exc:

                # ------------------------------------------------------
                # CRITICAL:
                # The position exists but has no protection.
                #
                # Do not continue normally.
                # Attempt emergency liquidation.
                # ------------------------------------------------------

                log.critical(
                    "[LIVE:%s] ❌ Protective stop FAILED: %s",
                    symbol,
                    exc,
                )

                try:

                    emergency_qty = (
                        actual_position_qty
                        * (
                            D("1")
                            - (
                                SELL_BALANCE_BUFFER_PCT
                                / D("100")
                            )
                        )
                    )

                    emergency = (
                        self.client.place_market_sell(
                            symbol=symbol,
                            quantity=emergency_qty,
                        )
                    )

                    log.critical(
                        "[LIVE:%s] Emergency liquidation sent: %s",
                        symbol,
                        emergency,
                    )

                except Exception as emergency_exc:

                    log.critical(
                        "[LIVE:%s] 🚨 EMERGENCY EXIT FAILED: %s",
                        symbol,
                        emergency_exc,
                    )

                raise RuntimeError(
                    (
                        "Protective stop failed after entry. "
                        "Position was placed into emergency handling."
                    )
                ) from exc

            # ----------------------------------------------------------
            # Final log
            # ----------------------------------------------------------

            log.info(
                "[LIVE:%s] ✅ POSITION OPENED | "
                "qty=%s | avgEntry=%s | "
                "stop=%s | TP=%s | "
                "orderId=%s | stopOrderId=%s",
                symbol,
                actual_position_qty,
                actual_entry_price,
                stop_loss_dec,
                take_profit_dec,
                execution.order_id,
                position.stop_order_id,
            )

            return {
                "status": "OPEN",
                "symbol": symbol,

                "quantity": actual_position_qty,
                "entry_price": actual_entry_price,

                "entry_quote_cost": execution.quote_qty,
                "entry_fee_usd": execution.fee_usd,

                "stop_loss": stop_loss_dec,
                "take_profit": take_profit_dec,

                "order_id": execution.order_id,
                "stop_order_id": position.stop_order_id,

                "entry_time": position.entry_time,

                "zscore": (
                    position.zscore
                    if position.zscore is not None
                    else None
                ),

                "signal_id": signal_id,
            }

    # ======================================================================
    # Close position
    # ======================================================================

    def close_position(
        self,
        symbol: str,
        price: Optional[Any] = None,
        reason: str = "MANUAL",
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        with self._lock:

            if not LIVE_MODE:

                return {
                    "status": "DISABLED",
                    "symbol": symbol,
                }

            position = self._get_position(
                symbol
            )

            if position is None:

                return {
                    "status": "NO_POSITION",
                    "symbol": symbol,
                }

            # ----------------------------------------------------------
            # Cancel protective stop first.
            #
            # We must never market-sell while the stop order remains
            # active unless we have reconciled its state.
            # ----------------------------------------------------------

            if position.stop_order_id is not None:

                try:

                    stop_state = self.client.get_order(
                        symbol=symbol,
                        order_id=position.stop_order_id,
                    )

                    stop_status = stop_state.get(
                        "status"
                    )

                    if stop_status == "FILLED":

                        # Stop already closed the position.
                        return self._finalize_stop_exit(
                            position=position,
                            order=stop_state,
                            reason="STOP_FILLED",
                        )

                    if stop_status in {
                        "NEW",
                        "PARTIALLY_FILLED",
                    }:

                        self.client.cancel_order(
                            symbol=symbol,
                            order_id=position.stop_order_id,
                        )

                except Exception as exc:

                    # --------------------------------------------------
                    # Critical safety principle:
                    #
                    # If we cannot confirm that the protective order
                    # is canceled, do not blindly send another sell.
                    # --------------------------------------------------

                    log.critical(
                        "[LIVE:%s] Could not safely cancel stop: %s",
                        symbol,
                        exc,
                    )

                    raise RuntimeError(
                        (
                            "Protective stop state is uncertain. "
                            "Manual reconciliation required."
                        )
                    ) from exc

            # ----------------------------------------------------------
            # Determine sell quantity.
            # ----------------------------------------------------------

            sell_quantity = (
                position.quantity
                * (
                    D("1")
                    - (
                        SELL_BALANCE_BUFFER_PCT
                        / D("100")
                    )
                )
            )

            sell_quantity = (
                self.client.normalize_quantity(
                    symbol=symbol,
                    quantity=sell_quantity,
                    market=True,
                )
            )

            if sell_quantity <= 0:

                raise RuntimeError(
                    "Sell quantity became zero."
                )

            # ----------------------------------------------------------
            # Execute SELL
            # ----------------------------------------------------------

            order = (
                self.client.place_market_sell(
                    symbol=symbol,
                    quantity=sell_quantity,
                )
            )

            if not order:

                raise RuntimeError(
                    "Sell order returned empty response."
                )

            order = self._ensure_filled(
                symbol,
                order,
            )

            execution = self._parse_execution(
                symbol=symbol,
                order=order,
            )

            # ----------------------------------------------------------
            # Cash PnL
            #
            # Entry:
            #   quote cost + quote/other fees
            #
            # Exit:
            #   quote proceeds - quote/other fees
            #
            # Base-asset fee is already represented by reduced
            # position quantity and therefore should not be subtracted
            # twice.
            # ----------------------------------------------------------

            exit_proceeds = (
                execution.quote_qty
                - execution.fee_usd
            )

            entry_cost = (
                position.entry_quote_cost
                + position.entry_fee_usd
            )

            pnl_usd = (
                exit_proceeds
                - entry_cost
            )

            pnl_pct = (
                (
                    pnl_usd
                    / entry_cost
                )
                * D("100")
                if entry_cost > 0
                else D("0")
            )

            # ----------------------------------------------------------
            # Record trade + remove position atomically.
            # ----------------------------------------------------------

            self._record_trade(
                symbol=symbol,
                side="SELL",
                order_id=execution.order_id,
                quantity=execution.executed_qty,
                average_price=execution.average_price,
                quote_qty=execution.quote_qty,
                fee_usd=execution.fee_usd,
                pnl_usd=pnl_usd,
                reason=reason,
            )

            self._delete_position(
                symbol
            )

            daily_pnl = (
                self.get_daily_realized_pnl()
            )

            log.info(
                "[LIVE:%s] ✅ POSITION CLOSED | "
                "reason=%s | qty=%s | "
                "exit=%s | PnL=%+.6f USD | "
                "PnL%%=%+.4f%% | Daily=%+.6f",
                symbol,
                reason,
                execution.executed_qty,
                execution.average_price,
                pnl_usd,
                pnl_pct,
                daily_pnl,
            )

            return {
                "status": "CLOSED",
                "symbol": symbol,

                "entry_price": position.entry_price,
                "exit_price": execution.average_price,

                "quantity": execution.executed_qty,

                "entry_quote_cost": position.entry_quote_cost,
                "exit_quote_proceeds": exit_proceeds,

                "entry_fee_usd": position.entry_fee_usd,
                "exit_fee_usd": execution.fee_usd,

                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,

                "exit_reason": reason,

                "order_id": execution.order_id,

                "daily_realized_pnl": daily_pnl,
            }

    # ======================================================================
    # Stop filled finalization
    # ======================================================================

    def _finalize_stop_exit(
        self,
        *,
        position: Position,
        order: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:

        execution = self._parse_execution(
            symbol=position.symbol,
            order=order,
        )

        exit_proceeds = (
            execution.quote_qty
            - execution.fee_usd
        )

        entry_cost = (
            position.entry_quote_cost
            + position.entry_fee_usd
        )

        pnl_usd = (
            exit_proceeds
            - entry_cost
        )

        pnl_pct = (
            (
                pnl_usd
                / entry_cost
            )
            * D("100")
            if entry_cost > 0
            else D("0")
        )

        self._record_trade(
            symbol=position.symbol,
            side="SELL",
            order_id=execution.order_id,
            quantity=execution.executed_qty,
            average_price=execution.average_price,
            quote_qty=execution.quote_qty,
            fee_usd=execution.fee_usd,
            pnl_usd=pnl_usd,
            reason=reason,
        )

        self._delete_position(
            position.symbol
        )

        daily_pnl = (
            self.get_daily_realized_pnl()
        )

        log.warning(
            "[LIVE:%s] 🛑 STOP LOSS FILLED | "
            "entry=%s | exit=%s | "
            "PnL=%+.6f | Daily=%+.6f",
            position.symbol,
            position.entry_price,
            execution.average_price,
            pnl_usd,
            daily_pnl,
        )

        return {
            "status": "STOPPED",
            "symbol": position.symbol,

            "entry_price": position.entry_price,
            "exit_price": execution.average_price,

            "quantity": execution.executed_qty,

            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,

            "exit_reason": reason,
            "order_id": execution.order_id,

            "daily_realized_pnl": daily_pnl,
        }

    # ======================================================================
    # Position manager
    # ======================================================================

    def manage_positions_once(self) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []

        with self._lock:

            positions = self._load_positions()

            for position in positions:

                symbol = position.symbol

                try:

                    # --------------------------------------------------
                    # Check stop order first.
                    # --------------------------------------------------

                    if position.stop_order_id is not None:

                        stop_order = self.client.get_order(
                            symbol=symbol,
                            order_id=position.stop_order_id,
                        )

                        stop_status = stop_order.get(
                            "status"
                        )

                        if stop_status == "FILLED":

                            result = (
                                self._finalize_stop_exit(
                                    position=position,
                                    order=stop_order,
                                    reason="STOP_FILLED",
                                )
                            )

                            results.append(
                                result
                            )

                            continue

                        if stop_status in {
                            "CANCELED",
                            "EXPIRED",
                            "REJECTED",
                        }:

                            log.critical(
                                "[LIVE:%s] Protective stop "
                                "is no longer active: %s",
                                symbol,
                                stop_status,
                            )

                            # Re-arm stop.
                            new_stop = (
                                self.client.place_stop_loss(
                                    symbol=symbol,
                                    quantity=(
                                        position.quantity
                                        * (
                                            D("1")
                                            - (
                                                SELL_BALANCE_BUFFER_PCT
                                                / D("100")
                                            )
                                        )
                                    ),
                                    stop_price=position.stop_loss,
                                )
                            )

                            new_stop_id = new_stop.get(
                                "orderId"
                            )

                            if new_stop_id is None:

                                raise RuntimeError(
                                    (
                                        "Could not re-arm "
                                        "protective stop."
                                    )
                                )

                            self._update_stop_order(
                                symbol=symbol,
                                stop_order_id=int(
                                    new_stop_id
                                ),
                            )

                    # --------------------------------------------------
                    # Current market
                    # --------------------------------------------------

                    current_price = (
                        self.client.get_price_decimal(
                            symbol
                        )
                    )

                    # --------------------------------------------------
                    # Take profit
                    # --------------------------------------------------

                    if (
                        current_price
                        >= position.take_profit
                    ):

                        result = self.close_position(
                            symbol=symbol,
                            price=current_price,
                            reason="TAKE_PROFIT",
                        )

                        results.append(
                            result
                        )

                        continue

                    # --------------------------------------------------
                    # Time-based exit.
                    # --------------------------------------------------

                    held_for = (
                        utc_now()
                        - parse_iso(
                            position.entry_time
                        )
                    )

                    if (
                        held_for
                        > timedelta(
                            minutes=MAX_HOLD_MINUTES
                        )
                    ):

                        result = self.close_position(
                            symbol=symbol,
                            price=current_price,
                            reason="TIME_EXIT",
                        )

                        results.append(
                            result
                        )

                        continue

                except Exception as exc:

                    log.exception(
                        "[LIVE:%s] Position management failed: %s",
                        symbol,
                        exc,
                    )

                    results.append(
                        {
                            "status": "ERROR",
                            "symbol": symbol,
                            "error": str(exc),
                        }
                    )

        return results

    # ======================================================================
    # Reconciliation
    # ======================================================================

    def reconcile_positions(self) -> List[Dict[str, Any]]:

        report: List[Dict[str, Any]] = []

        positions = self._load_positions()

        for position in positions:

            try:

                actual_balance = (
                    self.client.get_position(
                        position.symbol
                    )
                )

                expected = position.quantity

                tolerance = (
                    expected
                    * D("0.01")
                )

                difference = (
                    actual_balance
                    - expected
                )

                if abs(difference) > tolerance:

                    item = {
                        "symbol": position.symbol,
                        "status": "MISMATCH",

                        "expected_qty": expected,
                        "actual_qty": actual_balance,

                        "difference": difference,
                    }

                    log.critical(
                        "[LIVE:%s] Position mismatch | "
                        "expected=%s actual=%s diff=%s",
                        position.symbol,
                        expected,
                        actual_balance,
                        difference,
                    )

                    report.append(item)

                else:

                    report.append(
                        {
                            "symbol": position.symbol,
                            "status": "OK",
                            "expected_qty": expected,
                            "actual_qty": actual_balance,
                        }
                    )

            except Exception as exc:

                report.append(
                    {
                        "symbol": position.symbol,
                        "status": "ERROR",
                        "error": str(exc),
                    }
                )

        return report

    # ======================================================================
    # Status
    # ======================================================================

    def status(self) -> Dict[str, Any]:

        positions = self._load_positions()

        realized = (
            self.get_daily_realized_pnl()
        )

        unrealized = (
            self.get_unrealized_pnl()
        )

        return {
            "live_mode": LIVE_MODE,
            "testnet": USE_TESTNET,

            "positions": len(positions),
            "max_positions": MAX_OPEN_POSITIONS,

            "daily_realized_pnl": realized,
            "unrealized_pnl": unrealized,

            "combined_pnl": (
                realized
                + unrealized
            ),

            "daily_loss_limit": (
                MAX_DAILY_LOSS_USD
            ),

            "aggregate_open_risk": (
                self._aggregate_open_risk()
            ),

            "max_total_risk_pct": (
                MAX_TOTAL_RISK_PCT
            ),

            "symbols": [
                position.symbol
                for position in positions
            ],
        }


# ============================================================================
# Standalone diagnostics
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s - "
            "%(message)s"
        ),
    )

    trader = LiveTrader()

    print(
        trader.status()
    )

    print(
        trader.reconcile_positions()
    )

    # --------------------------------------------------------------
    # Position monitor:
    #
    # Call periodically from your scheduler:
    #
    # trader.manage_positions_once()
    #
    # Example:
    #
    # import time
    #
    # while True:
    #     trader.manage_positions_once()
    #     time.sleep(10)
    # --------------------------------------------------------------
