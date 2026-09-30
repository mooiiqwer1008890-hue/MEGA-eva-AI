"""
ffc.py
======
Fitness Feedback Control (FFC)
نظام حماية رأس المال والتحكم بالتنفيذ.

الفكرة:
- يحتفظ بأداء آخر N صفقة.
- يوقف فتح الصفقات الحقيقية عند تجاوز حد الخسارة.
- يستمر Paper Mode أثناء التوقف حتى يستطيع النظام تقييم التعافي.
- يعيد السماح بالتنفيذ الحقيقي بعد تجاوز عتبة الاستئناف.
- يحوي حواجز إضافية: Drawdown، الخسائر المتتالية، وCooldown.
- يحفظ الحالة بشكل ذري Atomic لتقليل خطر تلف ملف JSON.

ملاحظات مهمة:
- PnL يجب أن يكون Net PnL بعد الرسوم إن أمكن.
- capital يجب أن يكون رأس المال بعد إغلاق الصفقة.
- لا يعتبر FFC بديلاً عن إدارة المخاطر الخاصة بكل صفقة.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

log = logging.getLogger("ffc")


# ============================================================
# CONFIG
# ============================================================
FFC_STATE_FILE = os.getenv("FFC_STATE_FILE", "ffc_state.json")

# Rolling fitness
PHI_OFF_PCT = float(os.getenv("PHI_OFF_PCT", "-5.0"))
PHI_ON_PCT = float(os.getenv("PHI_ON_PCT", "2.0"))
LOOKBACK_TRADES = int(os.getenv("LOOKBACK_TRADES", "20"))
MIN_TRADES_TO_JUDGE = int(os.getenv("MIN_TRADES_TO_JUDGE", "5"))

# تأكيد الاستئناف: يحتاج البوت إلى N تقييمات ناجحة متتالية قبل إعادة LIVE.
RESUME_CONFIRM_TRADES = int(os.getenv("RESUME_CONFIRM_TRADES", "2"))

# حماية إضافية من سلسلة الخسائر.
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))

# Drawdown من أعلى رأس مال مسجل منذ بداية سجل FFC.
# 0 أو قيمة سالبة = تعطيل هذا الحاجز.
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "10.0"))

# بعد الإيقاف، لا يسمح بإعادة LIVE مباشرة حتى تمر عدة صفقات Paper.
RESUME_COOLDOWN_TRADES = int(os.getenv("RESUME_COOLDOWN_TRADES", "1"))


class FFC:
    """
    Fitness Feedback Control.

    is_live:
        True  => السماح بفتح صفقات حقيقية.
        False => منع الصفقات الحقيقية، مع إمكانية استمرار Paper Mode.
    """

    VERSION = 2

    def __init__(
        self,
        state_file: str = FFC_STATE_FILE,
        phi_off_pct: float = PHI_OFF_PCT,
        phi_on_pct: float = PHI_ON_PCT,
        lookback: int = LOOKBACK_TRADES,
        min_trades_to_judge: int = MIN_TRADES_TO_JUDGE,
        resume_confirm_trades: int = RESUME_CONFIRM_TRADES,
        max_consecutive_losses: int = MAX_CONSECUTIVE_LOSSES,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        resume_cooldown_trades: int = RESUME_COOLDOWN_TRADES,
    ) -> None:
        self.state_file = state_file
        self.phi_off_pct = float(phi_off_pct)
        self.phi_on_pct = float(phi_on_pct)
        self.lookback = int(lookback)
        self.min_trades_to_judge = int(min_trades_to_judge)
        self.resume_confirm_trades = int(resume_confirm_trades)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.resume_cooldown_trades = int(resume_cooldown_trades)

        self._validate_config()

        # حالة التنفيذ الحقيقي
        self.is_live: bool = True
        self.halt_reason: Optional[str] = None

        # آخر صفقات الأداء
        self.recent_trades: Deque[float] = deque(maxlen=self.lookback)
        # رأس المال بعد كل صفقة، بنفس ترتيب recent_trades
        self.capital_history: Deque[float] = deque(maxlen=self.lookback)

        # أعلى رأس مال مسجل للحساب منذ بداية FFC
        self.peak_capital: Optional[float] = None

        # إحصائيات
        self.total_halts: int = 0
        self.total_resumes: int = 0
        self.last_halt_time: Optional[str] = None
        self.last_resume_time: Optional[str] = None
        self.resume_confirm_count: int = 0
        self.paper_trades_since_halt: int = 0

        self._load()

    # ========================================================
    # Validation
    # ========================================================
    def _validate_config(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if self.min_trades_to_judge < 1:
            raise ValueError("min_trades_to_judge must be >= 1")
        if self.min_trades_to_judge > self.lookback:
            raise ValueError("min_trades_to_judge cannot exceed lookback")
        if self.phi_on_pct <= self.phi_off_pct:
            raise ValueError("phi_on_pct must be greater than phi_off_pct")
        if self.resume_confirm_trades < 1:
            raise ValueError("resume_confirm_trades must be >= 1")
        if self.max_consecutive_losses < 0:
            raise ValueError("max_consecutive_losses must be >= 0")
        if self.max_drawdown_pct < 0:
            raise ValueError("max_drawdown_pct must be >= 0")
        if self.resume_cooldown_trades < 0:
            raise ValueError("resume_cooldown_trades must be >= 0")

    # ========================================================
    # Public update API
    # ========================================================
    def update_after_trade(
        self,
        pnl_usd: float,
        capital: float,
        source: str = "paper",
    ) -> Dict[str, object]:
        """
        تحديث FFC بعد إغلاق صفقة.

        Parameters
        ----------
        pnl_usd:
            صافي ربح/خسارة الصفقة بالدولار.
        capital:
            رأس المال بعد إغلاق الصفقة.
        source:
            "paper" أو "live".

        Returns
        -------
        dict:
            حالة FFC بعد التحديث.
        """
        source = source.strip().lower()
        if source not in {"paper", "live"}:
            raise ValueError("source must be 'paper' or 'live'")

        pnl_usd = float(pnl_usd)
        capital = float(capital)

        if not math.isfinite(pnl_usd):
            raise ValueError("pnl_usd must be finite")
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError("capital must be a positive finite number")

        self.recent_trades.append(pnl_usd)
        self.capital_history.append(capital)

        if self.peak_capital is None or capital > self.peak_capital:
            self.peak_capital = capital

        if not self.is_live and source == "paper":
            self.paper_trades_since_halt += 1

        fitness = self._compute_fitness()
        drawdown = self._compute_drawdown()
        consecutive_losses = self._consecutive_losses()
        win_rate = self._win_rate()

        log.info(
            "[FFC] %s trade | PnL=$%+.2f | Capital=$%.2f | "
            "Fitness=%+.2f%% | DD=%.2f%% | WinRate=%.1f%% | "
            "LossStreak=%d | State=%s",
            source.upper(),
            pnl_usd,
            capital,
            fitness,
            drawdown,
            win_rate,
            consecutive_losses,
            "LIVE" if self.is_live else "HALTED",
        )

        # ----------------------------------------------------
        # 1) Safety halt
        # ----------------------------------------------------
        if self.is_live and len(self.recent_trades) >= self.min_trades_to_judge:
            if fitness <= self.phi_off_pct:
                self._halt(
                    fitness,
                    f"rolling fitness <= {self.phi_off_pct:.2f}%",
                )
            elif (
                self.max_consecutive_losses > 0
                and consecutive_losses >= self.max_consecutive_losses
            ):
                self._halt(
                    fitness,
                    f"{consecutive_losses} consecutive losses",
                )
            elif (
                self.max_drawdown_pct > 0
                and drawdown >= self.max_drawdown_pct
            ):
                self._halt(
                    fitness,
                    f"drawdown >= {self.max_drawdown_pct:.2f}%",
                )

        # ----------------------------------------------------
        # 2) Resume only from PAPER performance while halted
        # ----------------------------------------------------
        if not self.is_live and source == "paper":
            self._evaluate_resume(fitness, drawdown, consecutive_losses)

        self._save()
        return self.status()

    # ========================================================
    # Fitness
    # ========================================================
    def _compute_fitness(self) -> float:
        """
        Rolling PnL كنسبة من رأس المال في بداية نافذة الصفقات.

        هذه أفضل من الكود القديم لأن capital_history[0] هو رأس المال
        بعد أول صفقة، وليس قبلها.
        """
        n = len(self.recent_trades)
        if n < self.min_trades_to_judge:
            return 0.0

        total_pnl = sum(self.recent_trades)

        if len(self.capital_history) >= n:
            first_capital_after = self.capital_history[0]
            first_pnl = self.recent_trades[0]
            starting_capital = first_capital_after - first_pnl

            if starting_capital > 0:
                return (total_pnl / starting_capital) * 100.0

        # Fallback آمن إذا كانت بيانات الحالة القديمة ناقصة.
        return 0.0

    def _compute_drawdown(self) -> float:
        """Current drawdown from peak capital in percent."""
        if self.peak_capital is None or self.peak_capital <= 0:
            return 0.0
        if not self.capital_history:
            return 0.0

        current = self.capital_history[-1]
        dd = ((self.peak_capital - current) / self.peak_capital) * 100.0
        return max(0.0, dd)

    def _win_rate(self) -> float:
        if not self.recent_trades:
            return 0.0
        wins = sum(1 for pnl in self.recent_trades if pnl > 0)
        return (wins / len(self.recent_trades)) * 100.0

    def _consecutive_losses(self) -> int:
        count = 0
        for pnl in reversed(self.recent_trades):
            if pnl < 0:
                count += 1
            else:
                break
        return count

    # ========================================================
    # Halt / Resume
    # ========================================================
    def _halt(self, fitness: float, reason: str) -> None:
        """إيقاف فتح الصفقات الحقيقية."""
        if not self.is_live:
            return

        self.is_live = False
        self.total_halts += 1
        self.halt_reason = reason
        self.last_halt_time = self._utc_now()
        self.resume_confirm_count = 0
        self.paper_trades_since_halt = 0

        log.warning(
            "🔴 [FFC] HALT | reason=%s | fitness=%+.2f%% | "
            "paper evaluation remains enabled",
            reason,
            fitness,
        )

    def _evaluate_resume(
        self,
        fitness: float,
        drawdown: float,
        consecutive_losses: int,
    ) -> None:
        """تقييم استعادة LIVE باستخدام نتائج Paper فقط."""
        if self.paper_trades_since_halt < self.resume_cooldown_trades:
            return

        # يجب أولاً تجاوز عتبة الاستئناف.
        # ونمنع الاستئناف إذا بقي Drawdown قاتلاً.
        healthy = (
            fitness >= self.phi_on_pct
            and (
                self.max_drawdown_pct <= 0
                or drawdown < self.max_drawdown_pct
            )
            and consecutive_losses == 0
        )

        if healthy:
            self.resume_confirm_count += 1
        else:
            self.resume_confirm_count = 0

        if self.resume_confirm_count >= self.resume_confirm_trades:
            self._resume(fitness, "paper performance recovered")

    def _resume(self, fitness: float, reason: str) -> None:
        """إعادة السماح بفتح الصفقات الحقيقية."""
        self.is_live = True
        self.total_resumes += 1
        self.halt_reason = None
        self.last_resume_time = self._utc_now()
        self.resume_confirm_count = 0
        self.paper_trades_since_halt = 0

        log.info(
            "🟢 [FFC] RESUME | reason=%s | fitness=%+.2f%%",
            reason,
            fitness,
        )

    # ========================================================
    # Execution gate
    # ========================================================
    def can_open_position(self, mode: str = "live") -> bool:
        """
        هل يسمح FFC بفتح صفقة؟

        live  -> يتطلب أن يكون FFC غير متوقف.
        paper -> مسموح دائماً حتى أثناء halt، لأن الهدف من Paper Mode
                 هو اختبار التعافي واستعادة الصلاحية.
        """
        mode = mode.strip().lower()
        if mode == "paper":
            return True
        if mode == "live":
            return self.is_live
        raise ValueError("mode must be 'paper' or 'live'")

    def is_halted(self) -> bool:
        return not self.is_live

    # ========================================================
    # Manual controls
    # ========================================================
    def force_halt(self, reason: str = "manual halt") -> None:
        """إيقاف يدوي."""
        self._halt(self._compute_fitness(), reason)
        self._save()

    def force_resume(self, reason: str = "manual resume") -> None:
        """استئناف يدوي — استخدمه بحذر لأنه يتجاوز شروط FFC التلقائية."""
        self._resume(self._compute_fitness(), reason)
        self._save()

    # ========================================================
    # Status
    # ========================================================
    def status(self) -> Dict[str, object]:
        fitness = self._compute_fitness()
        drawdown = self._compute_drawdown()
        return {
            "version": self.VERSION,
            "is_live": self.is_live,
            "halt_reason": self.halt_reason,
            "fitness": round(fitness, 4),
            "drawdown_pct": round(drawdown, 4),
            "win_rate_pct": round(self._win_rate(), 2),
            "consecutive_losses": self._consecutive_losses(),
            "trades_count": len(self.recent_trades),
            "total_halts": self.total_halts,
            "total_resumes": self.total_resumes,
            "resume_confirm_count": self.resume_confirm_count,
            "paper_trades_since_halt": self.paper_trades_since_halt,
            "last_halt": self.last_halt_time,
            "last_resume": self.last_resume_time,
        }

    # ========================================================
    # Persistence
    # ========================================================
    def _save(self) -> None:
        """حفظ الحالة بشكل ذري لتقليل خطر تلف JSON."""
        try:
            directory = os.path.dirname(os.path.abspath(self.state_file))
            os.makedirs(directory, exist_ok=True)

            state = {
                "version": self.VERSION,
                "is_live": self.is_live,
                "halt_reason": self.halt_reason,
                "recent_trades": list(self.recent_trades),
                "capital_history": list(self.capital_history),
                "peak_capital": self.peak_capital,
                "total_halts": self.total_halts,
                "total_resumes": self.total_resumes,
                "last_halt_time": self.last_halt_time,
                "last_resume_time": self.last_resume_time,
                "resume_confirm_count": self.resume_confirm_count,
                "paper_trades_since_halt": self.paper_trades_since_halt,
            }

            tmp_file = f"{self.state_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_file, self.state_file)

        except Exception as exc:
            log.exception("[FFC] Failed to save state: %s", exc)

    def _load(self) -> None:
        """تحميل الحالة مع التوافق مع ملف FFC القديم."""
        if not os.path.exists(self.state_file):
            log.info("[FFC] No state file — starting fresh")
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            recent = state.get("recent_trades", [])
            capital = state.get("capital_history", [])

            self.is_live = bool(state.get("is_live", True))
            self.halt_reason = state.get("halt_reason")
            self.recent_trades = deque(
                [float(x) for x in recent if math.isfinite(float(x))],
                maxlen=self.lookback,
            )
            self.capital_history = deque(
                [float(x) for x in capital if math.isfinite(float(x)) and float(x) > 0],
                maxlen=self.lookback,
            )

            # توافق مع ملفات الإصدار السابق.
            self.total_halts = int(state.get("total_halts", 0))
            self.total_resumes = int(state.get("total_resumes", 0))
            self.last_halt_time = state.get("last_halt_time")
            self.last_resume_time = state.get("last_resume_time")
            self.resume_confirm_count = int(state.get("resume_confirm_count", 0))
            self.paper_trades_since_halt = int(state.get("paper_trades_since_halt", 0))

            peak = state.get("peak_capital")
            if peak is not None and math.isfinite(float(peak)) and float(peak) > 0:
                self.peak_capital = float(peak)
            elif self.capital_history:
                self.peak_capital = max(self.capital_history)

            # لأن ملفات الإصدار القديم كانت تخزن capital_history بطول lookback+1،
            # نحافظ فقط على الجزء القابل للمطابقة مع recent_trades.
            self._align_history_lengths()

            log.info(
                "[FFC] State loaded | live=%s | trades=%d | halts=%d | resumes=%d",
                self.is_live,
                len(self.recent_trades),
                self.total_halts,
                self.total_resumes,
            )

        except Exception as exc:
            log.exception("[FFC] Failed to load state: %s — starting fresh", exc)
            self.is_live = True
            self.halt_reason = None
            self.recent_trades.clear()
            self.capital_history.clear()
            self.peak_capital = None

    def _align_history_lengths(self) -> None:
        """مزامنة recent_trades و capital_history."""
        n = min(len(self.recent_trades), len(self.capital_history))
        if n <= 0:
            self.recent_trades.clear()
            self.capital_history.clear()
            return

        recent = list(self.recent_trades)[-n:]
        capital = list(self.capital_history)[-n:]
        self.recent_trades = deque(recent, maxlen=self.lookback)
        self.capital_history = deque(capital, maxlen=self.lookback)

    # ========================================================
    # Telegram
    # ========================================================
    def telegram_status(self) -> str:
        """رسالة جاهزة لـ Telegram باستخدام Markdown بسيط."""
        s = self.status()
        emoji = "🟢" if s["is_live"] else "🔴"
        state_text = "نشط LIVE" if s["is_live"] else "متوقف LIVE / PAPER يعمل"
        reason = s["halt_reason"] or "—"

        return (
            f"{emoji} *حالة FFC*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"الحالة: {state_text}\n"
            f"Fitness: {s['fitness']:+.2f}%\n"
            f"Drawdown: {s['drawdown_pct']:.2f}%\n"
            f"Win Rate: {s['win_rate_pct']:.1f}%\n"
            f"الخسائر المتتالية: {s['consecutive_losses']}\n"
            f"الصفقات المسجلة: {s['trades_count']}\n"
            f"مرات الإيقاف: {s['total_halts']}\n"
            f"مرات الاستئناف: {s['total_resumes']}\n"
            f"تأكيد الاستئناف: {s['resume_confirm_count']}/{self.resume_confirm_trades}\n"
            f"Paper منذ الإيقاف: {s['paper_trades_since_halt']}\n"
            f"سبب الإيقاف: {reason}\n"
            f"آخر إيقاف: {s['last_halt'] or 'لا يوجد'}\n"
            f"آخر استئناف: {s['last_resume'] or 'لا يوجد'}"
        )

    # ========================================================
    # Helpers
    # ========================================================
    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


# ============================================================
# Singleton
# ============================================================
_global_ffc: Optional[FFC] = None


def get_ffc() -> FFC:
    """الحصول على instance عام."""
    global _global_ffc
    if _global_ffc is None:
        _global_ffc = FFC()
    return _global_ffc


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    test_state = "/tmp/ffc_test_state.json"
    if os.path.exists(test_state):
        os.remove(test_state)

    ffc = FFC(
        state_file=test_state,
        phi_off_pct=-5.0,
        phi_on_pct=2.0,
        lookback=20,
        min_trades_to_judge=5,
        resume_confirm_trades=2,
        max_consecutive_losses=5,
        max_drawdown_pct=10.0,
        resume_cooldown_trades=1,
    )

    capital = 1000.0

    print("\n=== TEST 1: خمس خسائر ===\n")
    for _ in range(5):
        pnl = -15.0
        capital += pnl
        ffc.update_after_trade(pnl, capital, source="live")

    print(ffc.telegram_status())
    assert ffc.is_halted(), "FFC should halt after the losing sequence"

    print("\n=== TEST 2: Paper recovery ===\n")
    # 5 أرباح × 30$ بعد 5 خسائر × 15$ = +75$ rolling PnL
    # نحتاج تقييمين متتاليين فوق PHI_ON لأن RESUME_CONFIRM_TRADES=2.
    for _ in range(5):
        pnl = 30.0
        capital += pnl
        ffc.update_after_trade(pnl, capital, source="paper")

    print(ffc.telegram_status())
    assert ffc.is_live, "FFC should resume after confirmed paper recovery"

    print("\n=== TEST 3: persistence ===\n")
    ffc2 = FFC(state_file=test_state)
    print(ffc2.telegram_status())
    assert ffc2.is_live == ffc.is_live

    print("\n✅ FFC self-test passed\n")
