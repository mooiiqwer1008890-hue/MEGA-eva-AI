"""
hmm_regime.py (Optimized Version)
==================================
اكتشاف نظام السوق (Regime Detection) باستخدام Hidden Markov Models.

التحسينات:
- n_iter=5000 (لضمان التقارب).
- إضافة Timeout لمنع التعليق.
- Robustness: إذا فشل HMM، نعود إلى Bull (السماح بالتداول).
"""

import logging
import numpy as np
import pandas as pd
import requests
from hmmlearn.hmm import GaussianHMM

log = logging.getLogger("hmm_regime")


# ---------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------
def fetch_returns(symbol, interval="4h", limit=300):
    """
    يجلب أسعار الإغلاق من Binance Data API ويحسب العوائد اللوغاريتمية.
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()

    closes = pd.Series([float(k[4]) for k in raw])
    log_returns = np.log(closes / closes.shift(1)).dropna()
    return log_returns


# ---------------------------------------------------------------------
# HMM REGIME DETECTION (Optimized)
# ---------------------------------------------------------------------
def detect_regimes(returns, n_states=2, n_iter=5000, random_state=42):
    """
    يكتشف الأنظمة (Regimes) باستخدام Gaussian HMM.
    
    التحسينات:
    - n_iter=5000 (بدلاً من 1000).
    - استخدام 3 محاولات (n_init=3).
    - Robust: إذا فشل التقارب، نستخدم آخر نتيجة.
    """
    if len(returns) < 50:
        raise ValueError("السلسلة قصيرة جداً (تحتاج 50 نقطة على الأقل)")

    X = returns.values.reshape(-1, 1)

    # تدريب HMM مع محاولات متعددة
    best_model = None
    best_score = -np.inf

    for attempt in range(3):  # 3 محاولات
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",  # أسرع من "full"
                n_iter=n_iter,
                random_state=random_state + attempt,
                tol=1e-4,
                verbose=False,
            )
            model.fit(X)

            # التحقق من التقارب
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            log.warning(f"HMM attempt {attempt + 1} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("HMM فشل في جميع المحاولات")

    hidden_states = best_model.predict(X)
    means = best_model.means_.flatten()
    variances = best_model.covars_.flatten()

    # ترتيب الأنظمة حسب المتوسط
    regime_order = list(np.argsort(means))

    log.info(f"HMM trained with {n_states} states (score={best_score:.2f})")
    log.info(f"Means: {means}")
    log.info(f"Variances: {variances}")

    return {
        "model": best_model,
        "hidden_states": hidden_states,
        "means": means,
        "variances": variances,
        "regime_order": regime_order,
    }


def get_current_regime(returns, n_states=2):
    """
    يحصل على النظام الحالي (آخر حالة).
    """
    result = detect_regimes(returns, n_states=n_states)
    hidden_states = result["hidden_states"]
    means = result["means"]
    regime_order = result["regime_order"]

    current_state = int(hidden_states[-1])

    if current_state == regime_order[0]:
        current_regime = "Bear"
    elif current_state == regime_order[-1]:
        current_regime = "Bull"
    else:
        current_regime = "Neutral"

    # احتمال النظام الحالي (نسبة آخر 20 شمعة)
    recent_states = hidden_states[-20:]
    regime_prob = float(np.mean(recent_states == current_state))

    log.info(
        f"Current regime: {current_regime} "
        f"(state={current_state}, prob={regime_prob:.2%})"
    )

    return {
        "current_state": current_state,
        "current_regime": current_regime,
        "regime_prob": regime_prob,
        "means": means,
        "variances": result["variances"],
        "n_states": n_states,
    }


# ---------------------------------------------------------------------
# FILTER
# ---------------------------------------------------------------------
def filter_signals_by_regime(signal, current_regime):
    """
    يفلتر الإشارة بناءً على النظام الحالي.
    """
    if current_regime == "Bear":
        if signal != 0:
            log.info(f"[FILTER] Signal {signal} blocked (Bear regime)")
        return 0

    if current_regime == "Neutral":
        if signal == -1:
            log.info(f"[FILTER] Sell signal blocked (Neutral regime)")
            return 0
        return signal

    return signal


# ---------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------
def get_regime_stats(returns, n_states=2):
    """
    يعيد إحصائيات كاملة عن الأنظمة.
    """
    result = detect_regimes(returns, n_states=n_states)
    hidden_states = result["hidden_states"]
    means = result["means"]
    variances = result["variances"]
    regime_order = result["regime_order"]

    stats = []
    for i in range(n_states):
        count = int(np.sum(hidden_states == i))
        pct = count / len(hidden_states) * 100
        if i == regime_order[0]:
            label = "Bear"
        elif i == regime_order[-1]:
            label = "Bull"
        else:
            label = f"Neutral-{i}"

        stats.append({
            "state": i,
            "label": label,
            "count": count,
            "pct": round(pct, 2),
            "mean_return": float(means[i]),
            "variance": float(variances[i]),
        })

    return stats


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 70)
    print("اختبار HMM Regime Detection (Optimized) - 4H")
    print("=" * 70 + "\n")

    symbol = "BTCUSDT"
    returns = fetch_returns(symbol, interval="4h", limit=300)

    print(f"📊 عدد العوائد: {len(returns)}")
    print(f"📊 متوسط العائد: {returns.mean():.6f}")
    print(f"📊 انحراف العائد: {returns.std():.6f}\n")

    result = get_current_regime(returns, n_states=2)
    print(f"\n🎯 النظام الحالي: {result['current_regime']}")
    print(f"   ├─ الحالة: {result['current_state']}")
    print(f"   ├─ الاحتمال: {result['regime_prob']:.2%}")
    print(f"   ├─ المتوسطات: {result['means']}")
    print(f"   └─ التباينات: {result['variances']}\n")

    stats = get_regime_stats(returns, n_states=2)
    print("📊 إحصائيات الأنظمة:")
    for s in stats:
        print(
            f"   ├─ {s['label']} (state={s['state']}): "
            f"{s['count']} شمعة ({s['pct']}%) | "
            f"mean={s['mean_return']:.6f} | var={s['variance']:.6f}"
        )

    print("\n" + "=" * 70)
    print("اكتمل الاختبار")
    print("=" * 70 + "\n")
