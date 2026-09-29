"""
correlation_filter.py
=====================
يمنع فتح صفقات متزامنة في عملات مترابطة.

المشكلة:
    البوت يفتح 5 صفقات BUY في نفس اللحظة
    (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT)
    كلها مترابطة بنسبة 80-95%
    
    لو انهار السوق → 5 خسائر متزامنة!

الحل:
    - احسب مصفوفة الارتباط بين العملات (آخر 100 شمعة)
    - إذا كانت هناك صفقة مفتوحة، امنع فتح صفقة جديدة
      مع عملة مرتبطة بها > 0.7
    - استثناء: إذا كانت الصفقة الجديدة عكسية (Hedge)

المصدر:
    مستوحى من "Modern Portfolio Theory" + كتاب López de Prado
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("correlation_filter")


# ============================================================
# CONFIG
# ============================================================
DEFAULT_THRESHOLD = 0.70      # حد الارتباط الذي نمنع عنده
DEFAULT_LOOKBACK = 100         # عدد الشموع لحساب الارتباط
MAX_CORRELATED_POSITIONS = 2   # حد أقصى لصفقات مترابطة (شامل المفتوحة)


class CorrelationFilter:
    """
    فلتر الارتباط - يمنع فتح صفقات مترابطة
    """
    
    def __init__(self,
                 threshold: float = DEFAULT_THRESHOLD,
                 lookback: int = DEFAULT_LOOKBACK,
                 max_correlated: int = MAX_CORRELATED_POSITIONS):
        """
        Parameters
        ----------
        threshold : float
            حد الارتباط (0.7 = 70%)
        lookback : int
            عدد الشموع لحساب مصفوفة الارتباط
        max_correlated : int
            حد أقصى لعدد الصفقات المترابطة المسموح بها
        """
        self.threshold = threshold
        self.lookback = lookback
        self.max_correlated = max_correlated
        self._cache: Dict[str, np.ndarray] = {}
    
    # ========================================================
    # تحديث مخزون الأسعار
    # ========================================================
    def update_prices(self, symbol: str, closes: np.ndarray) -> None:
        """
        تحديث أسعار عملة (يُستدعى عند كل Full Cycle)
        """
        # حفظ آخر lookback شمعة فقط
        self._cache[symbol] = closes[-self.lookback:].copy()
    
    # ========================================================
    # حساب مصفوفة الارتباط
    # ========================================================
    def _compute_returns(self, closes: np.ndarray) -> np.ndarray:
        """حساب العوائد اللوغاريتمية"""
        if len(closes) < 2:
            return np.array([])
        return np.diff(np.log(closes))
    
    def _correlation(self, symbol_a: str, symbol_b: str) -> float:
        """
        حساب الارتباط بين عملتين
        """
        if symbol_a not in self._cache or symbol_b not in self._cache:
            return 0.0  # لا نعرف → نفترض لا يوجد ارتباط
        
        closes_a = self._cache[symbol_a]
        closes_b = self._cache[symbol_b]
        
        # محاذاة الطول
        min_len = min(len(closes_a), len(closes_b))
        if min_len < 10:
            return 0.0
        
        closes_a = closes_a[-min_len:]
        closes_b = closes_b[-min_len:]
        
        # حساب العوائد
        returns_a = self._compute_returns(closes_a)
        returns_b = self._compute_returns(closes_b)
        
        if len(returns_a) < 5 or len(returns_b) < 5:
            return 0.0
        
        # حساب الارتباط
        try:
            corr = np.corrcoef(returns_a, returns_b)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except Exception as e:
            log.warning(f"فشل حساب الارتباط {symbol_a}/{symbol_b}: {e}")
            return 0.0
    
    # ========================================================
    # فحص إمكانية فتح صفقة جديدة
    # ========================================================
    def can_open_position(self,
                          new_symbol: str,
                          open_symbols: List[str]) -> Tuple[bool, str]:
        """
        هل يمكن فتح صفقة جديدة؟
        
        Parameters
        ----------
        new_symbol : str
            العملة الجديدة المراد فتح صفقة عليها
        open_symbols : list[str]
            قائمة العملات التي لديها صفقات مفتوحة حالياً
        
        Returns
        -------
        (allowed: bool, reason: str)
        """
        if not open_symbols:
            return True, "لا توجد صفقات مفتوحة"
        
        # فلترة العملات المترابطة
        correlated_with: List[Tuple[str, float]] = []
        for existing in open_symbols:
            corr = self._correlation(new_symbol, existing)
            if abs(corr) >= self.threshold:
                correlated_with.append((existing, corr))
        
        if not correlated_with:
            return True, "لا يوجد ارتباط"
        
        # هل عدد المترابطات تجاوز الحد؟
        if len(correlated_with) >= self.max_correlated:
            reasons = ", ".join([f"{s}({c:+.2f})" for s, c in correlated_with])
            return False, f"مرتبط بـ {len(correlated_with)} صفقة: {reasons}"
        
        # مسموح إذا كان العدد أقل من الحد
        return True, "مرتبط لكن ضمن الحد"
    
    # ========================================================
    # فلترة قائمة كاملة من الإشارات
    # ========================================================
    def filter_signals(self,
                       signals: List[Dict],
                       open_symbols: List[str]) -> List[Dict]:
        """
        فلترة قائمة إشارات، وإرجاع الإشارات المسموح بها فقط
        
        Parameters
        ----------
        signals : list[dict]
            قائمة إشارات بالشكل:
            [{"symbol": "BTCUSDT", "z": -2.3, "price": 83000}, ...]
        open_symbols : list[str]
            العملات المفتوحة حالياً
        
        Returns
        -------
        list[dict]
            الإشارات المسموح بها
        """
        # ترتيب الإشارات حسب القوة (Z الأكثر انحرافاً أولاً)
        sorted_signals = sorted(signals, key=lambda s: abs(s.get("z", 0)), reverse=True)
        
        allowed = []
        current_open = list(open_symbols)
        
        for sig in sorted_signals:
            symbol = sig["symbol"]
            can_open, reason = self.can_open_position(symbol, current_open)
            
            if can_open:
                allowed.append(sig)
                current_open.append(symbol)
                log.info(f"[CORR] ✅ مسموح: {symbol} | {reason}")
            else:
                log.info(f"[CORR] ❌ مرفوض: {symbol} | {reason}")
                sig["rejection_reason"] = reason
        
        return allowed
    
    # ========================================================
    # ملخص مصفوفة الارتباط (للـ Debug)
    # ========================================================
    def correlation_summary(self, symbols: List[str]) -> str:
        """طباعة ملخص مصفوفة الارتباط"""
        lines = ["مصفوفة الارتباط:"]
        lines.append("         " + " ".join(f"{s[:6]:>8s}" for s in symbols))
        for s1 in symbols:
            row = [f"{s1[:6]:>8s}"]
            for s2 in symbols:
                if s1 == s2:
                    row.append(f"{1.00:>8.2f}")
                else:
                    corr = self._correlation(s1, s2)
                    row.append(f"{corr:>8.2f}")
            lines.append(" ".join(row))
        return "\n".join(lines)


# ============================================================
# Singleton للاستخدام في البوت
# ============================================================
_global_filter: Optional[CorrelationFilter] = None


def get_correlation_filter() -> CorrelationFilter:
    """الحصول على instance عام (Singleton)"""
    global _global_filter
    if _global_filter is None:
        _global_filter = CorrelationFilter()
    return _global_filter


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    cf = CorrelationFilter(threshold=0.7, lookback=100)
    
    # محاكاة بيانات: عملات مترابطة عالياً
    np.random.seed(42)
    base = np.cumsum(np.random.randn(100)) + 100
    
    cf.update_prices("BTCUSDT", base + np.random.randn(100) * 0.5)  # ارتباط عالي
    cf.update_prices("ETHUSDT", base + np.random.randn(100) * 0.5)  # ارتباط عالي
    cf.update_prices("BNBUSDT", base + np.random.randn(100) * 0.5)  # ارتباط عالي
    cf.update_prices("GOLD",    np.cumsum(np.random.randn(100)) + 100)  # مستقل
    
    print(cf.correlation_summary(["BTCUSDT", "ETHUSDT", "BNBUSDT", "GOLD"]))
    print()
    
    # اختبار الفلتر
    signals = [
        {"symbol": "BTCUSDT", "z": -2.5, "price": 83000},
        {"symbol": "ETHUSDT", "z": -2.3, "price": 2668},
        {"symbol": "BNBUSDT", "z": -2.1, "price": 755},
        {"symbol": "GOLD",    "z": -2.4, "price": 2000},
    ]
    
    allowed = cf.filter_signals(signals, open_symbols=[])
    print(f"\n✅ الإشارات المسموح بها: {[s['symbol'] for s in allowed]}")
