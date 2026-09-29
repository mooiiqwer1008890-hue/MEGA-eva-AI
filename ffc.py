"""
ffc.py
======
Fitness Feedback Control (FFC)
نظام حماية رأس المال.

المصدر:
    كتاب "Professional Automated Trading" - Eugene Durenard
    الفصل 12: "Feedback and Control"

الفكرة:
    - شغّل البوت في Paper Mode دائماً
    - احسب أداءه على آخر N صفقة
    - إذا تجاوزت الخسائر حداً معيناً → أوقف التنفيذ الحقيقي
    - إذا عاد الأداء → استأنف التنفيذ

المعاملات:
    - fitness: مقياس الأداء (Rolling PnL أو Win Rate)
    - PHI_ON: عتبة الاستئناف
    - PHI_OFF: عتبة الإيقاف

الفوائد:
    ✅ يحمي من Drawdown الحاد
    ✅ يتوقف تلقائياً عند فشل السوق
    ✅ يعود تلقائياً عند تحسنه
    ✅ لا يحتاج تدخل بشري
"""

import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

log = logging.getLogger("ffc")


# ============================================================
# CONFIG
# ============================================================
FFC_STATE_FILE = "ffc_state.json"

# عتبات الأداء (قابلة للتعديل)
PHI_OFF_PCT = -5.0         # إيقاف عند خسارة 5% من رأس المال
PHI_ON_PCT = 2.0           # استئناف عند ربح 2%
LOOKBACK_TRADES = 20       # عدد الصفقات لحساب الأداء
MIN_TRADES_TO_JUDGE = 5    # حد أدنى قبل الحكم


class FFC:
    """
    Fitness Feedback Control
    يحمي رأس المال عبر إيقاف/تشغيل البوت
    """
    
    def __init__(self,
                 state_file: str = FFC_STATE_FILE,
                 phi_off_pct: float = PHI_OFF_PCT,
                 phi_on_pct: float = PHI_ON_PCT,
                 lookback: int = LOOKBACK_TRADES):
        self.state_file = state_file
        self.phi_off_pct = phi_off_pct
        self.phi_on_pct = phi_on_pct
        self.lookback = lookback
        
        # الحالة
        self.is_live: bool = True           # هل البوت يفتح صفقات؟
        self.recent_trades: Deque[float] = deque(maxlen=lookback)  # آخر PnL
        self.capital_history: Deque[float] = deque(maxlen=lookback + 1)
        
        # إحصائيات
        self.total_halts: int = 0
        self.last_halt_time: Optional[str] = None
        self.last_resume_time: Optional[str] = None
        
        # تحميل الحالة من الملف
        self._load()
    
    # ========================================================
    # تحديث بعد كل صفقة
    # ========================================================
    def update_after_trade(self, pnl_usd: float, capital: float) -> None:
        """
        يُستدعى بعد إغلاق كل صفقة
        
        Parameters
        ----------
        pnl_usd : float
            الربح/الخسارة بالدولار
        capital : float
            رأس المال الحالي
        """
        self.recent_trades.append(pnl_usd)
        self.capital_history.append(capital)
        
        # حساب الأداء
        fitness = self._compute_fitness()
        
        log.info(f"[FFC] صفقة جديدة: PnL=${pnl_usd:+.2f} | "
                 f"رأس المال=${capital:.2f} | "
                 f"Fitness={fitness:+.2f}% | "
                 f"الحالة={'🟢 نشط' if self.is_live else '🔴 متوقف'}")
        
        # التحقق من شروط الإيقاف/الاستئناف
        if self.is_live and fitness < self.phi_off_pct:
            self._halt(fitness, "خسارة متتالية")
        elif not self.is_live and fitness > self.phi_on_pct:
            self._resume(fitness, "تحسن الأداء")
        
        self._save()
    
    # ========================================================
    # حساب مقياس الأداء
    # ========================================================
    def _compute_fitness(self) -> float:
        """
        حساب مقياس الأداء (نسبة التغير في رأس المال)
        
        Returns
        -------
        float
            النسبة المئوية (%)
        """
        if len(self.recent_trades) < MIN_TRADES_TO_JUDGE:
            return 0.0
        
        # الطريقة 1: مجموع PnL الأخير كنسبة من رأس المال
        total_pnl = sum(self.recent_trades)
        
        if len(self.capital_history) >= 2:
            # رأس المال في بداية النافذة
            oldest_capital = self.capital_history[0]
            if oldest_capital > 0:
                return (total_pnl / oldest_capital) * 100.0
        
        # Fallback: مجموع PnL مباشرة
        return total_pnl
    
    # ========================================================
    # الإيقاف
    # ========================================================
    def _halt(self, fitness: float, reason: str) -> None:
        """إيقاف البوت مؤقتاً"""
        self.is_live = False
        self.total_halts += 1
        self.last_halt_time = datetime.utcnow().isoformat()
        log.warning(f"🔴 [FFC] إيقاف البوت! السبب: {reason} | Fitness: {fitness:+.2f}%")
    
    # ========================================================
    # الاستئناف
    # ========================================================
    def _resume(self, fitness: float, reason: str) -> None:
        """استئناف البوت"""
        self.is_live = True
        self.last_resume_time = datetime.utcnow().isoformat()
        log.info(f"🟢 [FFC] استئناف البوت! السبب: {reason} | Fitness: {fitness:+.2f}%")
    
    # ========================================================
    # استعلام عن الحالة
    # ========================================================
    def can_open_position(self) -> bool:
        """هل يمكن فتح صفقة جديدة؟"""
        return self.is_live
    
    def status(self) -> Dict:
        """ملخص حالة FFC"""
        fitness = self._compute_fitness()
        return {
            "is_live": self.is_live,
            "fitness": round(fitness, 2),
            "trades_count": len(self.recent_trades),
            "total_halts": self.total_halts,
            "last_halt": self.last_halt_time,
            "last_resume": self.last_resume_time,
        }
    
    # ========================================================
    # حفظ/تحميل الحالة
    # ========================================================
    def _save(self) -> None:
        """حفظ الحالة في ملف"""
        try:
            state = {
                "is_live": self.is_live,
                "recent_trades": list(self.recent_trades),
                "capital_history": list(self.capital_history),
                "total_halts": self.total_halts,
                "last_halt_time": self.last_halt_time,
                "last_resume_time": self.last_resume_time,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"[FFC] فشل حفظ الحالة: {e}")
    
    def _load(self) -> None:
        """تحميل الحالة من ملف"""
        if not os.path.exists(self.state_file):
            log.info("[FFC] لا يوجد ملف حالة — بداية جديدة")
            return
        
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.is_live = state.get("is_live", True)
            self.recent_trades = deque(state.get("recent_trades", []), maxlen=self.lookback)
            self.capital_history = deque(state.get("capital_history", []), maxlen=self.lookback + 1)
            self.total_halts = state.get("total_halts", 0)
            self.last_halt_time = state.get("last_halt_time")
            self.last_resume_time = state.get("last_resume_time")
            
            log.info(f"[FFC] تم تحميل الحالة: "
                     f"is_live={self.is_live} | "
                     f"trades={len(self.recent_trades)} | "
                     f"halts={self.total_halts}")
        except Exception as e:
            log.error(f"[FFC] فشل تحميل الحالة: {e} — بداية جديدة")
    
    # ========================================================
    # رسالة Telegram
    # ========================================================
    def telegram_status(self) -> str:
        """رسالة جاهزة لإرسالها على Telegram"""
        s = self.status()
        emoji = "🟢" if s["is_live"] else "🔴"
        return (
            f"{emoji} *حالة FFC*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"الحالة: {'نشط' if s['is_live'] else 'متوقف'}\n"
            f"الأداء: {s['fitness']:+.2f}%\n"
            f"آخر {s['trades_count']} صفقة\n"
            f"مرات الإيقاف: {s['total_halts']}\n"
            f"آخر إيقاف: {s['last_halt'] or 'لا يوجد'}\n"
            f"آخر استئناف: {s['last_resume'] or 'لا يوجد'}"
        )


# ============================================================
# Singleton
# ============================================================
_global_ffc: Optional[FFC] = None


def get_ffc() -> FFC:
    """الحصول على instance عام"""
    global _global_ffc
    if _global_ffc is None:
        _global_ffc = FFC()
    return _global_ffc


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # حذف ملف حالة قديم للاختبار
    if os.path.exists(FFC_STATE_FILE):
        os.remove(FFC_STATE_FILE)
    
    ffc = FFC()
    capital = 1000.0
    
    # محاكاة سلسلة صفقات
    print("\n=== محاكاة: 5 خسائر متتالية ===\n")
    for i in range(5):
        pnl = -15.0  # خسارة $15 في كل صفقة
        capital += pnl
        ffc.update_after_trade(pnl, capital)
    
    print(f"\n{ffc.telegram_status()}\n")
    
    # محاكاة تحسن
    print("\n=== محاكاة: 3 أرباح متتالية ===\n")
    for i in range(3):
        pnl = +25.0
        capital += pnl
        ffc.update_after_trade(pnl, capital)
    
    print(f"\n{ffc.telegram_status()}\n")
