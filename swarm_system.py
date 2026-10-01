"""
swarm_system.py
===============

Adaptive Strategy Swarm Controller
-----------------------------------

هذا الملف هو "العقل الإداري" للسرب.

ليس الهدف:
    تشغيل 100 استراتيجية ثم عدّ الأصوات.

الهدف:

    Strategy Population
            ↓
    Paper Execution
            ↓
    Realistic Costs
            ↓
    Risk-Adjusted Fitness
            ↓
    Regime Compatibility
            ↓
    Family Diversification
            ↓
    Strategy Selection
            ↓
    Ensemble Consensus
            ↓
    Final Signal

===========================================================
IMPORTANT
===========================================================

هذا الملف مصمم للعمل مع النسخة المطورة من:

    swarm_strategies.py

التي تحتوي على:

    BaseStrategy
    StrategyResult
    create_swarm()
    RegimeDetector

===========================================================
PAPER EXECUTION MODEL
===========================================================

لمنع look-ahead:

    Candle[t] closes
          ↓
    strategy generates signal[t]
          ↓
    signal is stored as PENDING
          ↓
    Candle[t+1] opens
          ↓
    hypothetical paper execution

أي أننا لا نقول:

    "رأت الإشارة عند Close
     ونفذنا بنفس Close"

لأن ذلك غالبًا متفائل وغير واقعي.

===========================================================
PAPER COST MODEL
===========================================================

كل صفقة تتضمن:

    entry slippage
    exit slippage
    entry fee
    exit fee

حتى لا يصبح السرب متفوقًا فقط لأنه يتداول
بافتراض تنفيذ مثالي.

===========================================================
FITNESS
===========================================================

لا نستخدم:

    fitness = total_pnl

بل نستخدم مزيجًا من:

    expectancy
    downside risk
    win rate
    profit factor
    maximum drawdown
    stability
    sample-size shrinkage
    recency

والـfitness هنا:

    RANKING SCORE

وليس:
    Sharpe Ratio رسمي
    أو ضمانًا للعائد المستقبلي.

===========================================================
SELECTION
===========================================================

لا يتم اختيار:

    Top N by PnL

بل:

    Top N eligible
        ×
    regime compatibility
        ×
    family diversity

بحيث لا يستطيع:

    ZScore(10)
    ZScore(15)
    ZScore(20)

احتلال السرب بالكامل لمجرد أن جميعها
من نفس الفرضية الإحصائية.

===========================================================
STATE
===========================================================

بدلاً من JSON كمصدر حالة رئيسي:

    SQLite

يتم حفظ:

    strategy states
    trades
    pending signals
    paper positions
    equity
    fitness
    drawdown
    bar state

ويمكن أيضًا إنشاء JSON snapshot
لاستخدامه في Telegram أو debugging.

===========================================================
BACKTESTING
===========================================================

هذا الملف ليس بديلًا عن:

    Purged CV
    Walk-Forward
    CPCV
    Deflated Sharpe
    PBO
    Triple Barrier

بل يجب أن تأتي المصادقة الإحصائية
من Backtest Engine مستقل.

===========================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from swarm_strategies import (
    BaseStrategy,
    StrategyResult,
    create_swarm,
    RegimeDetector,
)


# ============================================================================
# Logging
# ============================================================================

log = logging.getLogger("swarm_system")


# ============================================================================
# Constants / configuration
# ============================================================================

BUY = 1
SELL = -1
HOLD = 0


SWARM_DB_FILE = os.getenv(
    "SWARM_DB_FILE",
    "swarm_state.sqlite3",
)

SWARM_STATE_FILE = os.getenv(
    "SWARM_STATE_FILE",
    "swarm_state.json",
)

TOP_N_STRATEGIES = int(
    os.getenv(
        "TOP_N_STRATEGIES",
        "3",
    )
)

MIN_FITNESS_TRADES = int(
    os.getenv(
        "MIN_FITNESS_TRADES",
        "20",
    )
)

FITNESS_DECAY = float(
    os.getenv(
        "FITNESS_DECAY",
        "0.97",
    )
)

FITNESS_PRIOR_STRENGTH = float(
    os.getenv(
        "FITNESS_PRIOR_STRENGTH",
        "25",
    )
)

MIN_FITNESS_TO_ACTIVATE = float(
    os.getenv(
        "MIN_FITNESS_TO_ACTIVATE",
        "0.05",
    )
)

MAX_ACTIVE_PER_FAMILY = int(
    os.getenv(
        "MAX_ACTIVE_PER_FAMILY",
        "2",
    )
)

MIN_CONFIRMING_FAMILIES = int(
    os.getenv(
        "MIN_CONFIRMING_FAMILIES",
        "2",
    )
)

ENTRY_THRESHOLD = float(
    os.getenv(
        "SWARM_ENTRY_THRESHOLD",
        "0.25",
    )
)

PAPER_FEE_BPS = float(
    os.getenv(
        "PAPER_FEE_BPS",
        "10",
    )
)

PAPER_SLIPPAGE_BPS = float(
    os.getenv(
        "PAPER_SLIPPAGE_BPS",
        "5",
    )
)

PAPER_MAX_HOLD_BARS = int(
    os.getenv(
        "PAPER_MAX_HOLD_BARS",
        "96",
    )
)

PAPER_ALLOW_SHORT = (
    os.getenv(
        "PAPER_ALLOW_SHORT",
        "false",
    ).strip().lower()
    == "true"
)

MAX_TRADE_HISTORY = int(
    os.getenv(
        "MAX_TRADE_HISTORY",
        "500",
    )
)

MAX_RETURN_HISTORY = int(
    os.getenv(
        "MAX_RETURN_HISTORY",
        "500",
    )
)

# Target magnitude used only to normalize expectancy
# in the internal ranking score.
FITNESS_TARGET_RETURN = float(
    os.getenv(
        "FITNESS_TARGET_RETURN",
        "0.0025",
    )
)

# Drawdown level at which the ranking receives
# a strong penalty.
FITNESS_DRAWDOWN_REFERENCE = float(
    os.getenv(
        "FITNESS_DRAWDOWN_REFERENCE",
        "0.10",
    )
)


# ============================================================================
# Utility functions
# ============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):

        return default


def clip01(
    value: float,
) -> float:

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def strategy_bar_fingerprint(
    candles: np.ndarray,
) -> str:
    """
    Fingerprint the latest candle.

    Prevents processing the same closed candle twice.

    Important:
    callers should provide CLOSED candles only.
    """

    latest = np.asarray(
        candles[-1],
        dtype=np.float64,
    )

    return hashlib.sha256(
        latest.tobytes()
    ).hexdigest()


# ============================================================================
# Paper Position
# ============================================================================

@dataclass
class PaperPosition:

    direction: int

    entry_price: float
    entry_fill_price: float

    entry_bar: int

    entry_confidence: float

    bars_held: int = 0

    entry_reason: str = "SIGNAL"


# ============================================================================
# Trade Record
# ============================================================================

@dataclass
class PaperTrade:

    strategy: str

    direction: int

    entry_price: float
    exit_price: float

    entry_fill_price: float
    exit_fill_price: float

    gross_return: float
    net_return: float

    bars_held: int

    exit_reason: str

    bar_index: int

    timestamp: str


# ============================================================================
# Strategy Tracker
# ============================================================================

class StrategyTracker:
    """
    Tracks one strategy entirely in Paper Mode.

    This object does NOT send Binance orders.

    It estimates what would have happened
    under the configured paper execution model.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
    ):

        self.strategy = strategy

        # --------------------------------------------------------------
        # State
        # --------------------------------------------------------------

        self.position: Optional[
            PaperPosition
        ] = None

        self.pending_signal: int = HOLD

        self.pending_confidence: float = 0.0

        self.pending_metadata: Dict[
            str,
            Any
        ] = {}

        self.last_processed_bar: int = -1

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        self.trades: Deque[
            float
        ] = deque(
            maxlen=MAX_TRADE_HISTORY
        )

        self.return_history: Deque[
            float
        ] = deque(
            maxlen=MAX_RETURN_HISTORY
        )

        self.wins: int = 0
        self.losses: int = 0

        self.total_pnl: float = 0.0

        self.gross_profit: float = 0.0

        self.gross_loss: float = 0.0

        self.equity: float = 1.0

        self.peak_equity: float = 1.0

        self.max_drawdown: float = 0.0

        # --------------------------------------------------------------
        # Online / exponentially weighted statistics
        # --------------------------------------------------------------

        self.ewma_expectancy: float = 0.0

        self.ewma_downside_square: float = 0.0

        self.ewma_abs_return: float = 0.0

        self.ewma_win_rate: float = 0.5

        # --------------------------------------------------------------
        # Meta
        # --------------------------------------------------------------

        self.last_trade_timestamp: Optional[
            str
        ] = None

        self.observations: int = 0

        self.total_bars: int = 0

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def trade_count(self) -> int:

        return len(
            self.trades
        )

    @property
    def win_rate(self) -> float:

        total = (
            self.wins
            + self.losses
        )

        if total <= 0:
            return 0.0

        return (
            self.wins
            / total
        )

    @property
    def expectancy(self) -> float:

        if not self.trades:
            return 0.0

        return float(
            np.mean(
                self.trades
            )
        )

    @property
    def downside_deviation(self) -> float:

        if self.ewma_downside_square <= 0:
            return 0.0

        return math.sqrt(
            self.ewma_downside_square
        )

    @property
    def profit_factor(self) -> float:

        if self.gross_loss >= 0:

            if self.gross_profit > 0:
                return float("inf")

            return 0.0

        return (
            self.gross_profit
            / abs(self.gross_loss)
        )

    @property
    def sortino_like(self) -> float:
        """
        Non-annualized downside-adjusted expectancy.

        Deliberately NOT called "Sortino Ratio"
        because no annualization/frequency normalization
        is performed.
        """

        downside = (
            self.downside_deviation
        )

        if downside <= 1e-12:

            if self.ewma_expectancy > 0:
                return 10.0

            return 0.0

        return (
            self.ewma_expectancy
            / downside
        )

    @property
    def drawdown_stability(self) -> float:

        if self.max_drawdown <= 0:
            return 1.0

        ratio = (
            self.max_drawdown
            / max(
                FITNESS_DRAWDOWN_REFERENCE,
                1e-9,
            )
        )

        return float(
            np.exp(
                -ratio
            )
        )

    @property
    def eligible(self) -> bool:

        return (
            self.trade_count
            >= MIN_FITNESS_TRADES
        )

    @property
    def fitness(self) -> float:
        """
        Internal ranking score in approximately [-1, +1].

        Components:
            40% downside-adjusted edge
            25% expectancy
            20% profit factor
            15% win rate

        Then:
            sample-size shrinkage
            drawdown penalty

        This is NOT a predictive guarantee.
        """

        if not self.eligible:

            return 0.0

        # --------------------------------------------------------------
        # Sample-size shrinkage
        # --------------------------------------------------------------

        reliability = (
            self.trade_count
            / (
                self.trade_count
                + FITNESS_PRIOR_STRENGTH
            )
        )

        # --------------------------------------------------------------
        # Risk-adjusted edge
        # --------------------------------------------------------------

        risk_component = math.tanh(
            self.sortino_like / 2.0
        )

        # --------------------------------------------------------------
        # Expectancy
        # --------------------------------------------------------------

        expectancy_component = math.tanh(
            self.ewma_expectancy
            / max(
                FITNESS_TARGET_RETURN,
                1e-9,
            )
        )

        # --------------------------------------------------------------
        # Profit factor
        #
        # Maps:
        #     PF = 1 → 0
        #     PF > 1 → positive
        #     PF < 1 → negative
        # --------------------------------------------------------------

        pf = self.profit_factor

        if not np.isfinite(pf):

            pf_component = 1.0

        else:

            pf_component = (
                pf - 1.0
            ) / max(
                pf + 1.0,
                1e-9,
            )

        # --------------------------------------------------------------
        # Win-rate component
        # --------------------------------------------------------------

        win_component = (
            2.0
            * (
                self.ewma_win_rate
                - 0.5
            )
        )

        raw_score = (
            0.40 * risk_component
            + 0.25 * expectancy_component
            + 0.20 * pf_component
            + 0.15 * win_component
        )

        # --------------------------------------------------------------
        # Sample-size shrinkage
        # --------------------------------------------------------------

        score = (
            reliability
            * raw_score
        )

        # --------------------------------------------------------------
        # Drawdown penalty
        # --------------------------------------------------------------

        drawdown_penalty = (
            0.25
            * min(
                self.max_drawdown
                / max(
                    FITNESS_DRAWDOWN_REFERENCE,
                    1e-9,
                ),
                1.0,
            )
        )

        score -= (
            drawdown_penalty
        )

        return float(
            np.clip(
                score,
                -1.0,
                1.0,
            )
        )

    @property
    def selection_weight(self) -> float:

        fitness = self.fitness

        # Negative fitness must NEVER become
        # positive weight through abs().
        #
        # Mapping:
        #     fitness = -1 → 0.05
        #     fitness =  0 → 0.50
        #     fitness = +1 → 0.95

        return float(
            np.clip(
                0.50
                + 0.45 * fitness,
                0.05,
                0.95,
            )
        )

    # ==================================================================
    # Paper execution
    # ==================================================================

    @staticmethod
    def _entry_fill(
        price: float,
        direction: int,
    ) -> float:

        slippage = (
            PAPER_SLIPPAGE_BPS
            / 10000.0
        )

        if direction == BUY:

            return (
                price
                * (
                    1.0
                    + slippage
                )
            )

        # Synthetic short entry.
        return (
            price
            * (
                1.0
                - slippage
            )
        )

    @staticmethod
    def _exit_fill(
        price: float,
        direction: int,
    ) -> float:

        slippage = (
            PAPER_SLIPPAGE_BPS
            / 10000.0
        )

        if direction == BUY:

            return (
                price
                * (
                    1.0
                    - slippage
                )
            )

        # Synthetic short exit.
        return (
            price
            * (
                1.0
                + slippage
            )
        )

    def submit_pending_signal(
        self,
        result: StrategyResult,
    ) -> None:

        self.pending_signal = int(
            result.signal
        )

        self.pending_confidence = float(
            result.confidence
        )

        self.pending_metadata = dict(
            result.metadata
        )

    def on_bar_open(
        self,
        open_price: float,
        bar_index: int,
        timestamp: Optional[str] = None,
    ) -> List[PaperTrade]:
        """
        Execute the previous candle's signal
        at the current candle OPEN.

        This is the critical anti-lookahead mechanism.
        """

        open_price = safe_float(
            open_price
        )

        if open_price <= 0:

            return []

        if (
            bar_index
            == self.last_processed_bar
        ):

            return []

        self.last_processed_bar = (
            bar_index
        )

        self.total_bars += 1

        closed_trades: List[
            PaperTrade
        ] = []

        # --------------------------------------------------------------
        # Increment holding time.
        # --------------------------------------------------------------

        if self.position is not None:

            self.position.bars_held += 1

        signal = int(
            self.pending_signal
        )

        # --------------------------------------------------------------
        # NO CURRENT POSITION
        # --------------------------------------------------------------

        if self.position is None:

            if signal == BUY:

                self._open_position(
                    direction=BUY,
                    price=open_price,
                    bar_index=bar_index,
                )

            elif (
                signal == SELL
                and PAPER_ALLOW_SHORT
            ):

                self._open_position(
                    direction=SELL,
                    price=open_price,
                    bar_index=bar_index,
                )

            return closed_trades

        # --------------------------------------------------------------
        # CURRENT LONG
        # --------------------------------------------------------------

        if (
            self.position.direction
            == BUY
        ):

            # Opposite signal:
            # close long.
            if signal == SELL:

                trade = (
                    self._close_position(
                        exit_price=open_price,
                        bar_index=bar_index,
                        reason="SIGNAL_REVERSAL",
                        timestamp=timestamp,
                    )
                )

                if trade:
                    closed_trades.append(
                        trade
                    )

                # Optional synthetic reversal.
                if (
                    PAPER_ALLOW_SHORT
                    and signal == SELL
                ):

                    self._open_position(
                        direction=SELL,
                        price=open_price,
                        bar_index=bar_index,
                    )

                return closed_trades

        # --------------------------------------------------------------
        # CURRENT SHORT
        # --------------------------------------------------------------

        if (
            self.position.direction
            == SELL
        ):

            if signal == BUY:

                trade = (
                    self._close_position(
                        exit_price=open_price,
                        bar_index=bar_index,
                        reason="SIGNAL_REVERSAL",
                        timestamp=timestamp,
                    )
                )

                if trade:
                    closed_trades.append(
                        trade
                    )

                self._open_position(
                    direction=BUY,
                    price=open_price,
                    bar_index=bar_index,
                )

                return closed_trades

        # --------------------------------------------------------------
        # TIME EXIT
        # --------------------------------------------------------------

        if (
            self.position is not None
            and self.position.bars_held
            >= PAPER_MAX_HOLD_BARS
        ):

            trade = (
                self._close_position(
                    exit_price=open_price,
                    bar_index=bar_index,
                    reason="TIME_EXIT",
                    timestamp=timestamp,
                )
            )

            if trade:
                closed_trades.append(
                    trade
                )

        return closed_trades

    def _open_position(
        self,
        direction: int,
        price: float,
        bar_index: int,
    ) -> None:

        if (
            direction == SELL
            and not PAPER_ALLOW_SHORT
        ):
            return

        fill = self._entry_fill(
            price,
            direction,
        )

        self.position = PaperPosition(
            direction=direction,

            entry_price=price,

            entry_fill_price=fill,

            entry_bar=bar_index,

            entry_confidence=(
                self.pending_confidence
            ),

            bars_held=0,

            entry_reason="SIGNAL",
        )

    def _close_position(
        self,
        exit_price: float,
        bar_index: int,
        reason: str,
        timestamp: Optional[str],
    ) -> Optional[PaperTrade]:

        if self.position is None:

            return None

        position = self.position

        exit_fill = self._exit_fill(
            exit_price,
            position.direction,
        )

        if (
            position.direction
            == BUY
        ):

            gross_return = (
                exit_fill
                / position.entry_fill_price
                - 1.0
            )

        else:

            gross_return = (
                position.entry_fill_price
                / exit_fill
                - 1.0
            )

        # Two-sided fee.
        fee_fraction = (
            2.0
            * PAPER_FEE_BPS
            / 10000.0
        )

        net_return = (
            gross_return
            - fee_fraction
        )

        trade = PaperTrade(
            strategy=self.strategy.name,

            direction=position.direction,

            entry_price=(
                position.entry_price
            ),

            exit_price=exit_price,

            entry_fill_price=(
                position.entry_fill_price
            ),

            exit_fill_price=exit_fill,

            gross_return=gross_return,

            net_return=net_return,

            bars_held=(
                max(
                    1,
                    bar_index
                    - position.entry_bar,
                )
            ),

            exit_reason=reason,

            bar_index=bar_index,

            timestamp=(
                timestamp
                or utc_iso()
            ),
        )

        self._record_trade(
            trade
        )

        self.position = None

        return trade

    # ==================================================================
    # Statistics
    # ==================================================================

    def _record_trade(
        self,
        trade: PaperTrade,
    ) -> None:

        r = float(
            trade.net_return
        )

        self.trades.append(
            r
        )

        self.return_history.append(
            r
        )

        self.observations += 1

        self.total_pnl += r

        if r > 0:

            self.wins += 1

            self.gross_profit += r

        else:

            self.losses += 1

            self.gross_loss += r

        # --------------------------------------------------------------
        # EWMA
        # --------------------------------------------------------------

        decay = float(
            np.clip(
                FITNESS_DECAY,
                0.50,
                0.9999,
            )
        )

        self.ewma_expectancy = (
            decay
            * self.ewma_expectancy
            + (
                1.0 - decay
            )
            * r
        )

        negative_square = (
            min(
                r,
                0.0,
            )
            ** 2
        )

        self.ewma_downside_square = (
            decay
            * self.ewma_downside_square
            + (
                1.0 - decay
            )
            * negative_square
        )

        self.ewma_abs_return = (
            decay
            * self.ewma_abs_return
            + (
                1.0 - decay
            )
            * abs(r)
        )

        win = (
            1.0
            if r > 0
            else 0.0
        )

        self.ewma_win_rate = (
            decay
            * self.ewma_win_rate
            + (
                1.0 - decay
            )
            * win
        )

        # --------------------------------------------------------------
        # Equity and drawdown
        # --------------------------------------------------------------

        self.equity *= (
            1.0 + r
        )

        self.equity = max(
            self.equity,
            1e-9,
        )

        self.peak_equity = max(
            self.peak_equity,
            self.equity,
        )

        drawdown = (
            1.0
            - (
                self.equity
                / self.peak_equity
            )
        )

        self.max_drawdown = max(
            self.max_drawdown,
            drawdown,
        )

        self.last_trade_timestamp = (
            trade.timestamp
        )

    # ==================================================================
    # Force close
    # ==================================================================

    def finalize(
        self,
        price: float,
        bar_index: int,
        timestamp: Optional[str] = None,
    ) -> Optional[PaperTrade]:

        if self.position is None:
            return None

        return self._close_position(
            exit_price=price,
            bar_index=bar_index,
            reason="FINALIZE",
            timestamp=timestamp,
        )

    # ==================================================================
    # Serialization
    # ==================================================================

    def to_dict(self) -> Dict[str, Any]:

        position = None

        if self.position is not None:

            position = {
                "direction": (
                    self.position.direction
                ),
                "entry_price": (
                    self.position.entry_price
                ),
                "entry_fill_price": (
                    self.position.entry_fill_price
                ),
                "entry_bar": (
                    self.position.entry_bar
                ),
                "entry_confidence": (
                    self.position.entry_confidence
                ),
                "bars_held": (
                    self.position.bars_held
                ),
                "entry_reason": (
                    self.position.entry_reason
                ),
            }

        return {
            "name": self.strategy.name,

            "family": getattr(
                self.strategy,
                "family",
                "unknown",
            ),

            "params": dict(
                getattr(
                    self.strategy,
                    "params",
                    {},
                )
            ),

            "position": position,

            "pending_signal": (
                self.pending_signal
            ),

            "pending_confidence": (
                self.pending_confidence
            ),

            "pending_metadata": (
                self.pending_metadata
            ),

            "last_processed_bar": (
                self.last_processed_bar
            ),

            "trades": list(
                self.trades
            ),

            "return_history": list(
                self.return_history
            ),

            "wins": self.wins,
            "losses": self.losses,

            "total_pnl": self.total_pnl,

            "gross_profit": (
                self.gross_profit
            ),

            "gross_loss": (
                self.gross_loss
            ),

            "equity": self.equity,

            "peak_equity": (
                self.peak_equity
            ),

            "max_drawdown": (
                self.max_drawdown
            ),

            "ewma_expectancy": (
                self.ewma_expectancy
            ),

            "ewma_downside_square": (
                self.ewma_downside_square
            ),

            "ewma_abs_return": (
                self.ewma_abs_return
            ),

            "ewma_win_rate": (
                self.ewma_win_rate
            ),

            "last_trade_timestamp": (
                self.last_trade_timestamp
            ),

            "observations": (
                self.observations
            ),

            "total_bars": (
                self.total_bars
            ),
        }

    def load_dict(
        self,
        data: Dict[str, Any],
    ) -> None:

        position_data = (
            data.get(
                "position"
            )
        )

        if position_data:

            self.position = PaperPosition(
                direction=int(
                    position_data.get(
                        "direction",
                        HOLD,
                    )
                ),

                entry_price=safe_float(
                    position_data.get(
                        "entry_price",
                        0,
                    )
                ),

                entry_fill_price=safe_float(
                    position_data.get(
                        "entry_fill_price",
                        0,
                    )
                ),

                entry_bar=int(
                    position_data.get(
                        "entry_bar",
                        -1,
                    )
                ),

                entry_confidence=safe_float(
                    position_data.get(
                        "entry_confidence",
                        0,
                    )
                ),

                bars_held=int(
                    position_data.get(
                        "bars_held",
                        0,
                    )
                ),

                entry_reason=str(
                    position_data.get(
                        "entry_reason",
                        "SIGNAL",
                    )
                ),
            )

        self.pending_signal = int(
            data.get(
                "pending_signal",
                HOLD,
            )
        )

        self.pending_confidence = (
            safe_float(
                data.get(
                    "pending_confidence",
                    0,
                )
            )
        )

        self.pending_metadata = dict(
            data.get(
                "pending_metadata",
                {},
            )
        )

        self.last_processed_bar = int(
            data.get(
                "last_processed_bar",
                -1,
            )
        )

        self.trades = deque(
            (
                safe_float(x)
                for x in data.get(
                    "trades",
                    [],
                )
            ),
            maxlen=MAX_TRADE_HISTORY,
        )

        self.return_history = deque(
            (
                safe_float(x)
                for x in data.get(
                    "return_history",
                    [],
                )
            ),
            maxlen=MAX_RETURN_HISTORY,
        )

        self.wins = int(
            data.get(
                "wins",
                0,
            )
        )

        self.losses = int(
            data.get(
                "losses",
                0,
            )
        )

        self.total_pnl = safe_float(
            data.get(
                "total_pnl",
                0,
            )
        )

        self.gross_profit = safe_float(
            data.get(
                "gross_profit",
                0,
            )
        )

        self.gross_loss = safe_float(
            data.get(
                "gross_loss",
                0,
            )
        )

        self.equity = max(
            safe_float(
                data.get(
                    "equity",
                    1,
                )
            ),
            1e-9,
        )

        self.peak_equity = max(
            safe_float(
                data.get(
                    "peak_equity",
                    self.equity,
                )
            ),
            1e-9,
        )

        self.max_drawdown = max(
            safe_float(
                data.get(
                    "max_drawdown",
                    0,
                )
            ),
            0,
        )

        self.ewma_expectancy = (
            safe_float(
                data.get(
                    "ewma_expectancy",
                    0,
                )
            )
        )

        self.ewma_downside_square = max(
            safe_float(
                data.get(
                    "ewma_downside_square",
                    0,
                )
            ),
            0,
        )

        self.ewma_abs_return = max(
            safe_float(
                data.get(
                    "ewma_abs_return",
                    0,
                )
            ),
            0,
        )

        self.ewma_win_rate = clip01(
            safe_float(
                data.get(
                    "ewma_win_rate",
                    0.5,
                )
            )
        )

        self.last_trade_timestamp = (
            data.get(
                "last_trade_timestamp"
            )
        )

        self.observations = int(
            data.get(
                "observations",
                0,
            )
        )

        self.total_bars = int(
            data.get(
                "total_bars",
                0,
            )
        )


# ============================================================================
# Swarm System
# ============================================================================

class SwarmSystem:
    """
    Adaptive strategy-selection system.

    All strategies are paper-tracked.

    Only selected eligible strategies contribute
    to the final consensus.
    """

    def __init__(
        self,
        top_n: int = TOP_N_STRATEGIES,
        state_db: str = SWARM_DB_FILE,
        state_file: str = SWARM_STATE_FILE,
    ):

        self.top_n = max(
            1,
            int(top_n),
        )

        self.state_db = state_db

        self.state_file = state_file

        # --------------------------------------------------------------
        # Strategy population
        # --------------------------------------------------------------

        self.swarm: List[
            BaseStrategy
        ] = create_swarm()

        if not self.swarm:

            raise RuntimeError(
                "create_swarm() returned no strategies."
            )

        self.trackers: Dict[
            str,
            StrategyTracker
        ] = {
            strategy.name:
                StrategyTracker(strategy)
            for strategy in self.swarm
        }

        # --------------------------------------------------------------
        # Regime
        # --------------------------------------------------------------

        self.regime_detector = (
            RegimeDetector()
        )

        self.last_regime = None

        # --------------------------------------------------------------
        # Bar state
        # --------------------------------------------------------------

        self.bar_index: int = -1

        self.last_bar_fingerprint: Optional[
            str
        ] = None

        self.last_result: Dict[
            str,
            Any
        ] = {}

        self.last_active_names: List[
            str
        ] = []

        self.last_family_scores: Dict[
            str,
            float
        ] = {}

        # --------------------------------------------------------------
        # Database
        # --------------------------------------------------------------

        self.db = sqlite3.connect(
            self.state_db,
            check_same_thread=False,
            timeout=30,
        )

        self.db.row_factory = sqlite3.Row

        self._initialize_database()

        self._load_state()

        log.info(
            "[SWARM] Initialized | "
            "strategies=%d | top_n=%d | "
            "min_trades=%d | db=%s",
            len(self.swarm),
            self.top_n,
            MIN_FITNESS_TRADES,
            self.state_db,
        )

    # ==================================================================
    # Database
    # ==================================================================

    def _initialize_database(
        self,
    ) -> None:

        cursor = self.db.cursor()

        cursor.execute(
            """
            PRAGMA journal_mode=WAL
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_state (
                name TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                strategy_name TEXT NOT NULL,

                direction INTEGER NOT NULL,

                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,

                entry_fill_price REAL NOT NULL,
                exit_fill_price REAL NOT NULL,

                gross_return REAL NOT NULL,
                net_return REAL NOT NULL,

                bars_held INTEGER NOT NULL,

                exit_reason TEXT NOT NULL,

                bar_index INTEGER NOT NULL,

                timestamp TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trade_strategy
            ON trade_log(strategy_name)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trade_bar
            ON trade_log(bar_index)
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        self.db.commit()

    # ==================================================================
    # State persistence
    # ==================================================================

    def _save_system_value(
        self,
        key: str,
        value: Any,
    ) -> None:

        self.db.execute(
            """
            INSERT INTO system_state (
                key,
                value,
                updated_at
            )
            VALUES (?, ?, ?)

            ON CONFLICT(key)
            DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (
                key,
                json.dumps(
                    value,
                    ensure_ascii=False,
                ),
                utc_iso(),
            ),
        )

    def _save_tracker(
        self,
        tracker: StrategyTracker,
    ) -> None:

        state = tracker.to_dict()

        self.db.execute(
            """
            INSERT INTO strategy_state (
                name,
                state_json,
                updated_at
            )
            VALUES (?, ?, ?)

            ON CONFLICT(name)
            DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (
                tracker.strategy.name,
                json.dumps(
                    state,
                    ensure_ascii=False,
                ),
                utc_iso(),
            ),
        )

    def _save_trade(
        self,
        trade: PaperTrade,
    ) -> None:

        self.db.execute(
            """
            INSERT INTO trade_log (
                strategy_name,

                direction,

                entry_price,
                exit_price,

                entry_fill_price,
                exit_fill_price,

                gross_return,
                net_return,

                bars_held,

                exit_reason,

                bar_index,

                timestamp
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                trade.strategy,

                trade.direction,

                trade.entry_price,
                trade.exit_price,

                trade.entry_fill_price,
                trade.exit_fill_price,

                trade.gross_return,
                trade.net_return,

                trade.bars_held,

                trade.exit_reason,

                trade.bar_index,

                trade.timestamp,
            ),
        )

    def _save_state(
        self,
    ) -> None:
        """
        Save all strategy state atomically.
        """

        try:

            self.db.execute(
                "BEGIN"
            )

            for tracker in (
                self.trackers.values()
            ):

                self._save_tracker(
                    tracker
                )

            self._save_system_value(
                "bar_index",
                self.bar_index,
            )

            self._save_system_value(
                "last_bar_fingerprint",
                self.last_bar_fingerprint,
            )

            self._save_system_value(
                "last_active_names",
                self.last_active_names,
            )

            self._save_system_value(
                "last_result",
                self.last_result,
            )

            self._save_system_value(
                "last_family_scores",
                self.last_family_scores,
            )

            self.db.commit()

            # Optional human-readable snapshot.
            self._export_json_snapshot()

        except Exception as exc:

            self.db.rollback()

            log.exception(
                "[SWARM] State save failed: %s",
                exc,
            )

    def _load_state(
        self,
    ) -> None:

        try:

            rows = self.db.execute(
                """
                SELECT name, state_json
                FROM strategy_state
                """
            ).fetchall()

            for row in rows:

                name = row["name"]

                if name not in self.trackers:
                    continue

                try:

                    state = json.loads(
                        row["state_json"]
                    )

                    self.trackers[
                        name
                    ].load_dict(
                        state
                    )

                except Exception as exc:

                    log.warning(
                        "[SWARM] Failed to load "
                        "strategy %s: %s",
                        name,
                        exc,
                    )

            state_rows = self.db.execute(
                """
                SELECT key, value
                FROM system_state
                """
            ).fetchall()

            system_values = {}

            for row in state_rows:

                try:

                    system_values[
                        row["key"]
                    ] = json.loads(
                        row["value"]
                    )

                except Exception:
                    continue

            self.bar_index = int(
                system_values.get(
                    "bar_index",
                    -1,
                )
                or -1
            )

            self.last_bar_fingerprint = (
                system_values.get(
                    "last_bar_fingerprint"
                )
            )

            self.last_active_names = list(
                system_values.get(
                    "last_active_names",
                    [],
                )
            )

            self.last_result = dict(
                system_values.get(
                    "last_result",
                    {},
                )
            )

            self.last_family_scores = dict(
                system_values.get(
                    "last_family_scores",
                    {},
                )
            )

            if rows:

                log.info(
                    "[SWARM] State restored | "
                    "strategies=%d | bar=%d",
                    len(rows),
                    self.bar_index,
                )

        except Exception as exc:

            log.warning(
                "[SWARM] Could not restore state: %s",
                exc,
            )

    def _export_json_snapshot(
        self,
    ) -> None:

        try:

            snapshot = {
                "saved_at": utc_iso(),

                "bar_index": (
                    self.bar_index
                ),

                "last_result": (
                    self.last_result
                ),

                "active_strategies": (
                    self.last_active_names
                ),

                "family_scores": (
                    self.last_family_scores
                ),

                "strategies": {
                    name: tracker.to_dict()
                    for name, tracker
                    in self.trackers.items()
                },
            }

            temp_file = (
                f"{self.state_file}.tmp"
            )

            with open(
                temp_file,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    snapshot,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            os.replace(
                temp_file,
                self.state_file,
            )

        except Exception as exc:

            log.warning(
                "[SWARM] JSON snapshot failed: %s",
                exc,
            )

    # ==================================================================
    # Candle validation
    # ==================================================================

    @staticmethod
    def _validate_candles(
        candles: np.ndarray,
    ) -> np.ndarray:

        arr = np.asarray(
            candles,
            dtype=float,
        )

        if arr.ndim != 2:

            raise ValueError(
                "candles must be 2D."
            )

        if arr.shape[1] < 5:

            raise ValueError(
                "candles must contain OHLCV."
            )

        arr = arr[:, :5]

        if len(arr) == 0:

            raise ValueError(
                "candles is empty."
            )

        if not np.all(
            np.isfinite(arr)
        ):

            raise ValueError(
                "candles contains NaN/Inf."
            )

        opens = arr[:, 0]
        highs = arr[:, 1]
        lows = arr[:, 2]
        closes = arr[:, 3]
        volumes = arr[:, 4]

        if np.any(
            opens <= 0
        ) or np.any(
            highs <= 0
        ) or np.any(
            lows <= 0
        ) or np.any(
            closes <= 0
        ):

            raise ValueError(
                "OHLC prices must be positive."
            )

        if np.any(
            volumes < 0
        ):

            raise ValueError(
                "Volume cannot be negative."
            )

        if np.any(
            highs < lows
        ):

            raise ValueError(
                "Found High < Low."
            )

        if np.any(
            closes > highs
        ) or np.any(
            closes < lows
        ):

            raise ValueError(
                "Close lies outside High/Low."
            )

        return arr

    # ==================================================================
    # Regime compatibility
    # ==================================================================

    @staticmethod
    def _regime_multiplier(
        strategy: BaseStrategy,
        regime: str,
    ) -> float:

        bias = getattr(
            strategy,
            "regime_bias",
            "neutral",
        )

        if regime in {
            "trend_up",
            "trend_down",
        }:

            if bias in {
                "trend",
                "breakout",
            }:

                return 1.20

            if bias == "mean_reversion":

                return 0.80

            return 1.0

        if regime == "mean_reversion":

            if bias in {
                "mean_reversion",
            }:

                return 1.20

            if bias in {
                "trend",
                "breakout",
            }:

                return 0.80

            return 1.0

        if regime == "transition":

            if bias in {
                "breakout",
            }:

                return 1.10

            return 1.0

        return 1.0

    # ==================================================================
    # Paper processing
    # ==================================================================

    def _process_previous_signals(
        self,
        candles: np.ndarray,
        timestamp: Optional[str],
    ) -> List[PaperTrade]:

        open_price = float(
            candles[-1, 0]
        )

        closed_trades: List[
            PaperTrade
        ] = []

        for tracker in (
            self.trackers.values()
        ):

            trades = tracker.on_bar_open(
                open_price=open_price,
                bar_index=self.bar_index,
                timestamp=timestamp,
            )

            for trade in trades:

                closed_trades.append(
                    trade
                )

                self._save_trade(
                    trade
                )

                log.debug(
                    "[SWARM] Paper trade | "
                    "%s | dir=%s | "
                    "net=%+.5f%% | "
                    "reason=%s",
                    trade.strategy,
                    trade.direction,
                    trade.net_return * 100,
                    trade.exit_reason,
                )

        return closed_trades

    # ==================================================================
    # Ranking
    # ==================================================================

    def _rank_eligible(
        self,
        regime: str,
        results: Dict[
            str,
            StrategyResult
        ],
    ) -> List[Tuple[
        StrategyTracker,
        float
    ]]:
        """
        Rank eligible strategies.

        Score depends on:

            fitness
            regime compatibility
            current signal confidence

        Current signal confidence only acts as a
        tie-breaking / activation factor.

        Historical fitness remains the primary component.
        """

        candidates: List[
            Tuple[
                StrategyTracker,
                float
            ]
        ] = []

        for name, tracker in (
            self.trackers.items()
        ):

            if not tracker.eligible:
                continue

            # Strategy must have acceptable historical
            # risk-adjusted fitness.
            if (
                tracker.fitness
                < MIN_FITNESS_TO_ACTIVATE
            ):

                continue

            strategy = (
                tracker.strategy
            )

            regime_multiplier = (
                self._regime_multiplier(
                    strategy,
                    regime,
                )
            )

            current_result = (
                results.get(
                    name
                )
            )

            confidence_bonus = 1.0

            if (
                current_result is not None
            ):

                confidence_bonus = (
                    0.85
                    + (
                        0.30
                        * current_result.confidence
                    )
                )

            selection_score = (
                tracker.fitness
                * regime_multiplier
                * confidence_bonus
            )

            candidates.append(
                (
                    tracker,
                    selection_score,
                )
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return candidates

    def _select_active(
        self,
        ranked: List[
            Tuple[
                StrategyTracker,
                float
            ]
        ],
    ) -> List[
        Tuple[
            StrategyTracker,
            float
        ]
    ]:

        selected: List[
            Tuple[
                StrategyTracker,
                float
            ]
        ] = []

        family_counts: Dict[
            str,
            int
        ] = defaultdict(int)

        # --------------------------------------------------------------
        # First pass:
        # diversity constrained best strategies.
        # --------------------------------------------------------------

        for tracker, score in ranked:

            family = getattr(
                tracker.strategy,
                "family",
                "unknown",
            )

            if (
                family_counts[family]
                >= MAX_ACTIVE_PER_FAMILY
            ):

                continue

            selected.append(
                (
                    tracker,
                    score,
                )
            )

            family_counts[family] += 1

            if len(
                selected
            ) >= self.top_n:

                break

        # --------------------------------------------------------------
        # Second pass:
        # if diversity constraints prevented enough
        # strategies, fill remaining slots.
        # --------------------------------------------------------------

        if len(selected) < self.top_n:

            selected_names = {
                tracker.strategy.name
                for tracker, _
                in selected
            }

            for tracker, score in ranked:

                if (
                    tracker.strategy.name
                    in selected_names
                ):

                    continue

                selected.append(
                    (
                        tracker,
                        score,
                    )
                )

                if len(
                    selected
                ) >= self.top_n:

                    break

        return selected

    # ==================================================================
    # Analyze
    # ==================================================================

    def analyze(
        self,
        candles: np.ndarray,
        timestamp: Optional[str] = None,
        bar_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a CLOSED candle set.

        Parameters
        ----------
        candles:
            OHLCV matrix.

        timestamp:
            Optional timestamp for logging.

        bar_id:
            Optional external unique bar ID.

        Notes
        -----
        The function assumes the final candle is CLOSED.

        For Binance Klines, the caller should not feed
        the currently forming candle as if it were final.
        """

        candles = self._validate_candles(
            candles
        )

        fingerprint = (
            str(bar_id)
            if bar_id is not None
            else strategy_bar_fingerprint(
                candles
            )
        )

        # --------------------------------------------------------------
        # Duplicate prevention.
        # --------------------------------------------------------------

        if (
            fingerprint
            == self.last_bar_fingerprint
        ):

            return dict(
                self.last_result
            )

        self.bar_index += 1

        # --------------------------------------------------------------
        # 1. Execute previous candle signals
        #    at current OPEN.
        # --------------------------------------------------------------

        closed_trades = (
            self._process_previous_signals(
                candles,
                timestamp,
            )
        )

        # --------------------------------------------------------------
        # 2. Determine current market regime.
        # --------------------------------------------------------------

        regime_state = (
            self.regime_detector.detect(
                candles
            )
        )

        self.last_regime = (
            regime_state
        )

        regime = (
            regime_state.regime
        )

        # --------------------------------------------------------------
        # 3. Run every strategy.
        #
        # All strategies remain in PAPER MODE.
        # Nothing is actually executed here.
        # --------------------------------------------------------------

        results: Dict[
            str,
            StrategyResult
        ] = {}

        all_signals = {
            "BUY": 0,
            "SELL": 0,
            "HOLD": 0,
        }

        for tracker in (
            self.trackers.values()
        ):

            try:

                result = (
                    tracker.strategy.analyze(
                        candles
                    )
                )

                if not isinstance(
                    result,
                    StrategyResult,
                ):

                    raise TypeError(
                        (
                            f"{tracker.strategy.name} "
                            "returned invalid StrategyResult."
                        )
                    )

            except Exception as exc:

                log.exception(
                    "[SWARM] Strategy error: %s",
                    tracker.strategy.name,
                )

                result = StrategyResult(
                    signal=HOLD,
                    confidence=0.0,
                    metadata={
                        "error": str(exc),
                    },
                )

            results[
                tracker.strategy.name
            ] = result

            # ----------------------------------------------------------
            # Store signal for NEXT BAR execution.
            # ----------------------------------------------------------

            tracker.submit_pending_signal(
                result
            )

            if result.signal == BUY:

                all_signals[
                    "BUY"
                ] += 1

            elif result.signal == SELL:

                all_signals[
                    "SELL"
                ] += 1

            else:

                all_signals[
                    "HOLD"
                ] += 1

        # --------------------------------------------------------------
        # 4. Rank eligible strategies.
        # --------------------------------------------------------------

        ranked = self._rank_eligible(
            regime=regime,
            results=results,
        )

        # --------------------------------------------------------------
        # 5. Select active subset.
        # --------------------------------------------------------------

        active = self._select_active(
            ranked
        )

        active_names = [
            tracker.strategy.name
            for tracker, _
            in active
        ]

        self.last_active_names = (
            active_names
        )

        # --------------------------------------------------------------
        # 6. Aggregate active strategies.
        # --------------------------------------------------------------

        family_scores: Dict[
            str,
            float
        ] = defaultdict(float)

        family_weights: Dict[
            str,
            float
        ] = defaultdict(float)

        strategy_scores: Dict[
            str,
            float
        ] = {}

        bullish_families = set()
        bearish_families = set()

        weighted_numerator = 0.0
        weighted_denominator = 0.0

        for tracker, selection_score in active:

            name = (
                tracker.strategy.name
            )

            result = results[
                name
            ]

            regime_multiplier = (
                self._regime_multiplier(
                    tracker.strategy,
                    regime,
                )
            )

            # Historical fitness is primary.
            historical_weight = (
                0.50
                + (
                    0.50
                    * tracker.fitness
                )
            )

            historical_weight = max(
                historical_weight,
                0.05,
            )

            # Current confidence.
            signal_weight = (
                0.25
                + (
                    0.75
                    * result.confidence
                )
            )

            weight = (
                historical_weight
                * signal_weight
                * regime_multiplier
            )

            signed_score = (
                result.signal
                * result.confidence
                * weight
            )

            strategy_scores[
                name
            ] = signed_score

            weighted_numerator += (
                signed_score
            )

            weighted_denominator += (
                abs(weight)
            )

            family = getattr(
                tracker.strategy,
                "family",
                "unknown",
            )

            family_scores[
                family
            ] += signed_score

            family_weights[
                family
            ] += abs(weight)

            if result.signal == BUY:

                bullish_families.add(
                    family
                )

            elif result.signal == SELL:

                bearish_families.add(
                    family
                )

        if weighted_denominator > 0:

            weighted_signal = (
                weighted_numerator
                / weighted_denominator
            )

        else:

            weighted_signal = 0.0

        # --------------------------------------------------------------
        # Normalize family scores.
        # --------------------------------------------------------------

        normalized_family_scores = {}

        for family, score in (
            family_scores.items()
        ):

            denominator = max(
                family_weights[
                    family
                ],
                1e-12,
            )

            normalized_family_scores[
                family
            ] = (
                score
                / denominator
            )

        self.last_family_scores = (
            normalized_family_scores
        )

        # --------------------------------------------------------------
        # Family confirmation.
        # --------------------------------------------------------------

        if weighted_signal >= ENTRY_THRESHOLD:

            confirming_families = (
                len(
                    bullish_families
                )
            )

        elif (
            weighted_signal
            <= -ENTRY_THRESHOLD
        ):

            confirming_families = (
                len(
                    bearish_families
                )
            )

        else:

            confirming_families = 0

        # --------------------------------------------------------------
        # Final decision.
        # --------------------------------------------------------------

        consensus_signal = HOLD

        if (
            weighted_signal
            >= ENTRY_THRESHOLD
            and confirming_families
            >= MIN_CONFIRMING_FAMILIES
        ):

            consensus_signal = BUY

        elif (
            weighted_signal
            <= -ENTRY_THRESHOLD
            and confirming_families
            >= MIN_CONFIRMING_FAMILIES
        ):

            consensus_signal = SELL

        # --------------------------------------------------------------
        # Confidence.
        #
        # Confidence is not "probability of profit".
        # It is an internal strength score.
        # --------------------------------------------------------------

        active_count = len(
            active
        )

        if active_count > 0:

            breadth = (
                sum(
                    1
                    for tracker, _
                    in active
                    if (
                        results[
                            tracker.strategy.name
                        ].signal
                        == consensus_signal
                    )
                )
                / active_count
                if consensus_signal
                != HOLD
                else 0.0
            )

        else:

            breadth = 0.0

        confirmation_strength = (
            min(
                confirming_families
                / max(
                    MIN_CONFIRMING_FAMILIES,
                    1,
                ),
                1.0,
            )
        )

        confidence = clip01(
            0.50
            * min(
                abs(
                    weighted_signal
                ),
                1.0,
            )
            + 0.30 * breadth
            + 0.20 * confirmation_strength
        )

        # --------------------------------------------------------------
        # Minimum-confidence gate.
        # --------------------------------------------------------------

        if (
            consensus_signal != HOLD
            and confidence < 0.20
        ):

            consensus_signal = HOLD

        # --------------------------------------------------------------
        # Top strategy information.
        # --------------------------------------------------------------

        top_strategies = []

        for tracker, score in active:

            result = results[
                tracker.strategy.name
            ]

            top_strategies.append(
                {
                    "name": tracker.strategy.name,

                    "family": getattr(
                        tracker.strategy,
                        "family",
                        "unknown",
                    ),

                    "signal": result.signal,

                    "confidence": round(
                        result.confidence,
                        4,
                    ),

                    "fitness": round(
                        tracker.fitness,
                        6,
                    ),

                    "selection_score": round(
                        score,
                        6,
                    ),

                    "trades": (
                        tracker.trade_count
                    ),

                    "win_rate": round(
                        tracker.win_rate,
                        4,
                    ),

                    "profit_factor": (
                        round(
                            tracker.profit_factor,
                            4,
                        )
                        if np.isfinite(
                            tracker.profit_factor
                        )
                        else None
                    ),

                    "max_drawdown": round(
                        tracker.max_drawdown,
                        6,
                    ),

                    "metadata": (
                        result.metadata
                    ),
                }
            )

        # --------------------------------------------------------------
        # Count eligibility.
        # --------------------------------------------------------------

        eligible_count = sum(
            1
            for tracker
            in self.trackers.values()
            if tracker.eligible
        )

        # --------------------------------------------------------------
        # Build result.
        # --------------------------------------------------------------

        result = {

            "bar_index": (
                self.bar_index
            ),

            "regime": regime,

            "consensus_signal": (
                consensus_signal
            ),

            "consensus_confidence": round(
                confidence,
                4,
            ),

            "weighted_signal": round(
                weighted_signal,
                6,
            ),

            "breadth": round(
                breadth,
                4,
            ),

            "confirming_families": (
                confirming_families
            ),

            "eligible_strategies": (
                eligible_count
            ),

            "active_strategies": (
                active_names
            ),

            "top_strategies": (
                top_strategies
            ),

            "family_scores": {
                key: round(
                    value,
                    6,
                )
                for key, value
                in normalized_family_scores.items()
            },

            "all_signals": (
                all_signals
            ),

            "paper_trades_closed": len(
                closed_trades
            ),

            "regime_diagnostics": {
                "trend_strength": (
                    regime_state.trend_strength
                ),

                "volatility_level": (
                    regime_state.volatility_level
                ),

                "ema_fast": (
                    regime_state.ema_fast
                ),

                "ema_slow": (
                    regime_state.ema_slow
                ),

                "atr": (
                    regime_state.atr
                ),
            },

            "execution_model": {
                "paper_fee_bps": (
                    PAPER_FEE_BPS
                ),

                "paper_slippage_bps": (
                    PAPER_SLIPPAGE_BPS
                ),

                "paper_max_hold_bars": (
                    PAPER_MAX_HOLD_BARS
                ),

                "paper_allow_short": (
                    PAPER_ALLOW_SHORT
                ),

                "next_bar_execution": True,
            },
        }

        # --------------------------------------------------------------
        # Persist.
        # --------------------------------------------------------------

        self.last_bar_fingerprint = (
            fingerprint
        )

        self.last_result = result

        self._save_state()

        return dict(
            result
        )

    # ==================================================================
    # Force finalize
    # ==================================================================

    def finalize(
        self,
        candles: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Close all remaining paper positions.

        Use this at the end of a backtest/paper window.

        It should NOT be called on every live cycle.
        """

        candles = self._validate_candles(
            candles
        )

        final_price = float(
            candles[-1, 3]
        )

        closed = []

        for tracker in (
            self.trackers.values()
        ):

            trade = tracker.finalize(
                price=final_price,
                bar_index=max(
                    self.bar_index,
                    0,
                ),
                timestamp=timestamp,
            )

            if trade:

                closed.append(
                    trade
                )

                self._save_trade(
                    trade
                )

        self._save_state()

        return {
            "status": "FINALIZED",
            "closed_positions": len(
                closed
            ),
        }

    # ==================================================================
    # Strategy ranking / status
    # ==================================================================

    def rank_strategies(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:

        trackers = sorted(
            self.trackers.values(),
            key=lambda tracker: (
                tracker.fitness
            ),
            reverse=True,
        )

        result = []

        for tracker in trackers[
            :max(
                1,
                limit,
            )
        ]:

            result.append(
                {
                    "name": tracker.strategy.name,

                    "family": getattr(
                        tracker.strategy,
                        "family",
                        "unknown",
                    ),

                    "fitness": round(
                        tracker.fitness,
                        6,
                    ),

                    "eligible": (
                        tracker.eligible
                    ),

                    "trades": (
                        tracker.trade_count
                    ),

                    "win_rate": round(
                        tracker.win_rate,
                        4,
                    ),

                    "expectancy": round(
                        tracker.expectancy,
                        6,
                    ),

                    "profit_factor": (
                        round(
                            tracker.profit_factor,
                            4,
                        )
                        if np.isfinite(
                            tracker.profit_factor
                        )
                        else None
                    ),

                    "sortino_like": round(
                        tracker.sortino_like,
                        4,
                    ),

                    "max_drawdown": round(
                        tracker.max_drawdown,
                        6,
                    ),

                    "equity": round(
                        tracker.equity,
                        6,
                    ),

                    "selection_weight": round(
                        tracker.selection_weight,
                        4,
                    ),
                }
            )

        return result

    def get_status(
        self,
    ) -> Dict[str, Any]:

        eligible = [
            tracker
            for tracker
            in self.trackers.values()
            if tracker.eligible
        ]

        total_trades = sum(
            tracker.trade_count
            for tracker
            in self.trackers.values()
        )

        paper_positions = sum(
            1
            for tracker
            in self.trackers.values()
            if tracker.position is not None
        )

        return {
            "total_strategies": (
                len(self.swarm)
            ),

            "eligible_strategies": (
                len(eligible)
            ),

            "active_strategies": (
                self.last_active_names
            ),

            "paper_open_positions": (
                paper_positions
            ),

            "total_paper_trades": (
                total_trades
            ),

            "bar_index": (
                self.bar_index
            ),

            "regime": (
                getattr(
                    self.last_regime,
                    "regime",
                    "unknown",
                )
            ),

            "weighted_signal": (
                self.last_result.get(
                    "weighted_signal",
                    0,
                )
            ),

            "consensus_signal": (
                self.last_result.get(
                    "consensus_signal",
                    HOLD,
                )
            ),

            "consensus_confidence": (
                self.last_result.get(
                    "consensus_confidence",
                    0,
                )
            ),

            "top_5": (
                self.rank_strategies(
                    limit=5
                )
            ),
        }

    # ==================================================================
    # Telegram
    # ==================================================================

    def telegram_summary(
        self,
    ) -> str:

        status = (
            self.get_status()
        )

        signal_map = {
            BUY: "🟢 BUY",
            SELL: "🔴 SELL",
            HOLD: "⚪ HOLD",
        }

        signal = signal_map.get(
            status[
                "consensus_signal"
            ],
            "⚪ HOLD",
        )

        lines = [
            "🐝 *حالة Strategy Swarm*",
            "━━━━━━━━━━━━━━━━━━━━",
            (
                f"📦 الاستراتيجيات: "
                f"`{status['total_strategies']}`"
            ),
            (
                f"✅ المؤهلة: "
                f"`{status['eligible_strategies']}`"
            ),
            (
                f"🎯 النشطة: "
                f"`{len(status['active_strategies'])}`"
            ),
            (
                f"📊 Paper Trades: "
                f"`{status['total_paper_trades']}`"
            ),
            (
                f"🌐 Regime: "
                f"`{status['regime']}`"
            ),
            (
                f"🧠 القرار: "
                f"`{signal}`"
            ),
            (
                f"🎚️ Confidence: "
                f"`{status['consensus_confidence']:.3f}`"
            ),
            (
                f"⚖️ Weighted Score: "
                f"`{status['weighted_signal']:+.4f}`"
            ),
            "━━━━━━━━━━━━━━━━━━━━",
            "🏆 *أفضل الاستراتيجيات:*",
        ]

        for i, strategy in enumerate(
            status["top_5"],
            1,
        ):

            pf = strategy[
                "profit_factor"
            ]

            pf_text = (
                f"{pf:.2f}"
                if pf is not None
                else "∞"
            )

            lines.append(
                (
                    f"{i}. `{strategy['name']}`\n"
                    f"   "
                    f"Fitness="
                    f"`{strategy['fitness']:+.3f}` | "
                    f"WR="
                    f"`{strategy['win_rate'] * 100:.1f}%` | "
                    f"PF="
                    f"`{pf_text}` | "
                    f"Trades="
                    f"`{strategy['trades']}`"
                )
            )

        return "\n".join(
            lines
        )

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self,
        confirm: bool = False,
    ) -> None:

        if not confirm:

            raise ValueError(
                "reset(confirm=True) is required."
            )

        for tracker in (
            self.trackers.values()
        ):

            tracker.__init__(
                tracker.strategy
            )

        self.bar_index = -1

        self.last_bar_fingerprint = None

        self.last_result = {}

        self.last_active_names = []

        self.last_family_scores = {}

        self.db.execute(
            "DELETE FROM strategy_state"
        )

        self.db.execute(
            "DELETE FROM trade_log"
        )

        self.db.execute(
            "DELETE FROM system_state"
        )

        self.db.commit()

        try:

            if os.path.exists(
                self.state_file
            ):

                os.remove(
                    self.state_file
                )

        except OSError:
            pass

        log.warning(
            "[SWARM] All swarm state reset."
        )


# ============================================================================
# Singleton
# ============================================================================

_global_swarm: Optional[
    SwarmSystem
] = None


def get_swarm() -> SwarmSystem:

    global _global_swarm

    if _global_swarm is None:

        _global_swarm = (
            SwarmSystem()
        )

    return _global_swarm


# ============================================================================
# Test / demonstration
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ADAPTIVE SWARM SYSTEM TEST"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------------
    # Deterministic synthetic market
    # --------------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    n = 1200

    returns = (
        0.00015
        + rng.normal(
            0.0,
            0.006,
            n,
        )
    )

    # Add several regime-like sections.
    returns[
        300:500
    ] += 0.0006

    returns[
        700:850
    ] -= 0.0007

    close = (
        100.0
        * np.cumprod(
            1.0 + returns
        )
    )

    open_price = np.empty(
        n,
        dtype=float,
    )

    open_price[0] = (
        close[0]
    )

    open_price[1:] = (
        close[:-1]
    )

    intrabar_spread = (
        np.abs(
            rng.normal(
                0.0,
                0.002,
                n,
            )
        )
        * close
    )

    high = (
        np.maximum(
            open_price,
            close,
        )
        + intrabar_spread
    )

    low = (
        np.minimum(
            open_price,
            close,
        )
        - intrabar_spread
    )

    volume = rng.lognormal(
        mean=8.0,
        sigma=0.50,
        size=n,
    )

    candles = np.column_stack(
        [
            open_price,
            high,
            low,
            close,
            volume,
        ]
    )

    # --------------------------------------------------------------
    # Use isolated DB for demonstration.
    # --------------------------------------------------------------

    demo_db = (
        "swarm_demo.sqlite3"
    )

    demo_json = (
        "swarm_demo.json"
    )

    swarm = SwarmSystem(
        top_n=3,
        state_db=demo_db,
        state_file=demo_json,
    )

    # --------------------------------------------------------------
    # Run sequentially.
    #
    # IMPORTANT:
    # Each iteration represents a CLOSED candle.
    # The signal generated on candle t becomes executable
    # at candle t+1 OPEN.
    # --------------------------------------------------------------

    decisions = []

    start = max(
        150,
        MIN_FITNESS_TRADES * 5,
    )

    for i in range(
        start,
        n,
    ):

        result = swarm.analyze(
            candles[: i + 1],
            timestamp=utc_iso(),
            bar_id=i,
        )

        decisions.append(
            result
        )

        if (
            result["consensus_signal"]
            != HOLD
        ):

            print(
                f"\nBar {i}:"
            )

            print(
                "  Regime:",
                result["regime"],
            )

            print(
                "  Signal:",
                result[
                    "consensus_signal"
                ],
            )

            print(
                "  Confidence:",
                result[
                    "consensus_confidence"
                ],
            )

            print(
                "  Weighted:",
                result[
                    "weighted_signal"
                ],
            )

            print(
                "  Active:",
                result[
                    "active_strategies"
                ],
            )

    # --------------------------------------------------------------
    # Finalize paper positions.
    # --------------------------------------------------------------

    swarm.finalize(
        candles,
        timestamp=utc_iso(),
    )

    # --------------------------------------------------------------
    # Print system status.
    # --------------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        swarm.telegram_summary()
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------------
    # Show top strategies.
    # --------------------------------------------------------------

    for item in swarm.rank_strategies(
        limit=10
    ):

        print(
            f"{item['name']:<36} "
            f"fitness={item['fitness']:+.4f} "
            f"trades={item['trades']:>4} "
            f"WR={item['win_rate']:.2%} "
            f"PF={item['profit_factor']} "
            f"DD={item['max_drawdown']:.2%}"
        )

    print(
        "\n✅ Swarm test completed."
    )
