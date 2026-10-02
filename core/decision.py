"""
MEGA Decision Contract
======================

الهدف:
    توفير كائن موحد يمثل قرار MEGA التحليلي.

مبدأ التصميم:
    هذا الملف لا يحسب المؤشرات.
    هذا الملف لا ينفذ الصفقات.
    هذا الملف لا يتصل بـ Binance أو Telegram.

    وظيفته فقط:
        تجميع نتيجة طبقات التحليل المختلفة
        في Decision واحد يمكن استخدامه لاحقًا بواسطة:

        Signal
          ↓
        Regime
          ↓
        Volatility
          ↓
        Sentiment
          ↓
        Correlation
          ↓
        Risk
          ↓
        Execution
          ↓
        Telegram
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Decision:
    """
    الكائن الموحد لقرار MEGA.

    ملاحظة:
        Decision لا يقرر بنفسه.
        الطبقات التحليلية هي التي تملأ بياناته.
    """

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    symbol: str

    timestamp: str

    # ---------------------------------------------------------
    # Core signal
    # ---------------------------------------------------------

    direction: str = "HOLD"

    strength: float = 0.0

    z_score: Optional[float] = None

    # ---------------------------------------------------------
    # Market regime
    # ---------------------------------------------------------

    regime: Optional[str] = None

    regime_probability: Optional[float] = None

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    volatility_regime: Optional[str] = None

    volatility_ratio: Optional[float] = None

    annualized_volatility: Optional[float] = None

    current_volatility: Optional[float] = None

    forecast_volatility: Optional[float] = None

    # ---------------------------------------------------------
    # Sentiment
    # ---------------------------------------------------------

    sentiment_label: Optional[str] = None

    sentiment_score: Optional[float] = None

    sentiment_count: Optional[int] = None

    # ---------------------------------------------------------
    # Portfolio / correlation
    # ---------------------------------------------------------

    correlation_allowed: Optional[bool] = None

    correlation_reason: Optional[str] = None

    # ---------------------------------------------------------
    # Risk
    # ---------------------------------------------------------

    confidence: float = 0.0

    allowed: bool = False

    risk: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Decision explanation
    # ---------------------------------------------------------

    reasons: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Extra metadata
    # ---------------------------------------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------

    @classmethod
    def now(
        cls,
        symbol: str,
        **kwargs: Any,
    ) -> "Decision":
        """
        إنشاء Decision مع توقيت UTC واضح.
        """

        return cls(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )

    # ---------------------------------------------------------
    # Reason helpers
    # ---------------------------------------------------------

    def add_reason(self, reason: str) -> None:
        """
        إضافة سبب للقرار بدون تكراره.
        """

        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        تحويل Decision إلى dict قابل للحفظ بصيغة JSON.
        """

        return asdict(self)

    # ---------------------------------------------------------
    # Telegram payload
    # ---------------------------------------------------------

    def to_telegram_dict(self) -> Dict[str, Any]:
        """
        نسخة مستقرة مخصصة لطبقة Telegram.
        """

        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "strength": self.strength,
            "z_score": self.z_score,

            "regime": self.regime,
            "regime_probability": self.regime_probability,

            "volatility_regime": self.volatility_regime,
            "volatility_ratio": self.volatility_ratio,
            "annualized_volatility": self.annualized_volatility,
            "current_volatility": self.current_volatility,
            "forecast_volatility": self.forecast_volatility,

            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "sentiment_count": self.sentiment_count,

            "correlation_allowed": self.correlation_allowed,
            "correlation_reason": self.correlation_reason,

            "confidence": self.confidence,
            "allowed": self.allowed,

            "risk": dict(self.risk),

            "reasons": list(self.reasons),
        }
