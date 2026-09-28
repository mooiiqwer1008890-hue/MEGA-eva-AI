"""
hmm_regime.py
=============
اكتشاف نظام السوق (Regime Detection) باستخدام Hidden Markov Models (HMM).

المصدر:
- كتاب "Advanced Algorithmic Trading" - الفصل 14
- كتاب "Python Trader" - الفصل 31

الفكرة:
- السوق يمر بفترات مختلفة (Bull = هادئ، Bear = متقلب).
- HMM يكتشف هذه الفترات تلقائياً من بيانات العوائد.
- نستخدم النتيجة كفلتر: لا نتداول في الفترات المتقلبة.

الفاصل الزمني الموصى به: 4H (4 ساعات)
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
def fetch_returns(symbol, interval="4h", limit=1000):
    """
    يجلب أسعار الإغلاق من Binance Data API ويحسب العوائد اللوغاريتمية.
    
    Parameters
    ----------
    symbol : str
        مثال: "BTCUSDT"
    interval : str
        "4h" (الموصى به)، "1h", "1d"
    limit : int
        عدد الشموع
    
    Returns
    -------
    pd.Series
        سلسلة العوائد اللوغاريتمية
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
# HMM REGIME DETECTION
# ---------------------------------------------------------------------
def detect_regimes(returns, n_states=2, n_iter=1000, random_state=42):
    """
    يكتشف الأنظمة (Regimes) باستخدام Gaussian Hidden Markov Model.
    
    Parameters
    ----------
    returns : pd.Series
        سلسلة العوائد
    n_states : int
        عدد الأنظمة (2 = Bull/Bear، 3 = Bull/Neutral/Bear)
    n_iter : int
        عدد التكرارات للتدريب
    random_state : int
        للنتائج المتكررة
    
    Returns
    -------
    dict
        {
            "model": GaussianHMM,
            "hidden_states": np.array,
            "means": np.array,
            "variances": np.array,
            "regime_order": list
        }
    """
    if len(returns) < 100:
        raise ValueError("السلسلة قصيرة جداً (تحتاج 100 نقطة على الأقل)")

    # تحويل العوائد إلى شكل (n_samples, 1)
    X = returns.values.reshape(-1, 1)

    # تدريب HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)

    # التنبؤ بالحالات المخفية
    hidden_states = model.predict(X)

    # استخراج المعايير
    means = model.means_.flatten()
    variances = model.covars_.flatten()

    # ترتيب الأنظمة حسب المتوسط (الأقل = Bear، الأعلى = Bull)
    regime_order = list(np.argsort(means))

    log.info(f"HMM trained with {n_states} states")
    log.info(f"Means: {means}")
    log.info(f"Variances: {variances}")
    log.info(f"Regime order (Bear -> Bull): {regime_order}")

    return {
        "model": model,
        "hidden_states": hidden_states,
        "means": means,
        "variances": variances,
        "regime_order": regime_order,
    }


def get_current_regime(returns, n_states=2):
    """
    يحصل على النظام الحالي (آخر حالة).
    
    Returns
    -------
    dict
        {
            "current_state": int,
            "current_regime": str,   # "Bear" أو "Bull"
            "regime_prob": float,    # احتمال النظام الحالي
            "means": np.array,
            "variances": np.array,
            "n_states": int
        }
    """
    result = detect_regimes(returns, n_states=n_states)
    hidden_states = result["hidden_states"]
    means = result["means"]
    regime_order = result["regime_order"]

    current_state = int(hidden_states[-1])

    # تحديد إذا كان Bull أم Bear
    # regime_order[0] = الأقل متوسطاً (Bear)
    # regime_order[-1] = الأعلى متوسطاً (Bull)
    if current_state == regime_order[0]:
        current_regime = "Bear"
    elif current_state == regime_order[-1]:
        current_regime = "Bull"
    else:
        current_regime = "Neutral"

    # حساب احتمال النظام الحالي (تقريبي)
    # نستخدم نسبة الحالات في آخر 20 شمعة
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
    
    القاعدة:
    - في "Bull": نسمح بالتداول.
    - في "Bear": نوقف التداول.
    - في "Neutral": نسمح بحذر.
    
    Parameters
    ----------
    signal : int
        إشارة (1 = شراء، -1 = بيع، 0 = لا شيء)
    current_regime : str
        "Bull"، "Bear"، "Neutral"
    
    Returns
    -------
    int
        إشارة مفلترة
    """
    if current_regime == "Bear":
        if signal != 0:
            log.info(f"[FILTER] Signal {signal} blocked (Bear regime)")
        return 0

    if current_regime == "Neutral":
        # نسمح بإشارات الشراء فقط، ونمنع البيع
        if signal == -1:
            log.info(f"[FILTER] Sell signal blocked (Neutral regime)")
            return 0
        return signal

    # Bull: نسمح بكل شيء
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
    print("اختبار HMM Regime Detection - الفاصل الزمني: 4H")
    print("=" * 70 + "\n")

    # اختبار على BTCUSDT
    symbol = "BTCUSDT"

    # جلب العوائد
    returns = fetch_returns(symbol, interval="4h", limit=1000)
    print(f"📊 عدد العوائد: {len(returns)}")
    print(f"📊 متوسط العائد: {returns.mean():.6f}")
    print(f"📊 انحراف العائد: {returns.std():.6f}\n")

    # اختبار بنظامين (2 states)
    print("=" * 70)
    print("اختبار 1: نظامان (2 States) - Bull vs Bear")
    print("=" * 70)

    result_2 = get_current_regime(returns, n_states=2)
    print(f"\n🎯 النظام الحالي: {result_2['current_regime']}")
    print(f"   ├─ الحالة: {result_2['current_state']}")
    print(f"   ├─ الاحتمال: {result_2['regime_prob']:.2%}")
    print(f"   ├─ المتوسطات: {result_2['means']}")
    print(f"   └─ التباينات: {result_2['variances']}\n")

    stats_2 = get_regime_stats(returns, n_states=2)
    print("📊 إحصائيات الأنظمة (2 States):")
    for s in stats_2:
        print(
            f"   ├─ {s['label']} (state={s['state']}): "
            f"{s['count']} شمعة ({s['pct']}%) | "
            f"mean={s['mean_return']:.6f} | var={s['variance']:.6f}"
        )

    # اختبار بثلاثة أنظمة (3 states)
    print("\n" + "=" * 70)
    print("اختبار 2: ثلاثة أنظمة (3 States) - Bull vs Neutral vs Bear")
    print("=" * 70)

    result_3 = get_current_regime(returns, n_states=3)
    print(f"\n🎯 النظام الحالي: {result_3['current_regime']}")
    print(f"   ├─ الحالة: {result_3['current_state']}")
    print(f"   ├─ الاحتمال: {result_3['regime_prob']:.2%}")
    print(f"   ├─ المتوسطات: {result_3['means']}")
    print(f"   └─ التباينات: {result_3['variances']}\n")

    stats_3 = get_regime_stats(returns, n_states=3)
    print("📊 إحصائيات الأنظمة (3 States):")
    for s in stats_3:
        print(
            f"   ├─ {s['label']} (state={s['state']}): "
            f"{s['count']} شمعة ({s['pct']}%) | "
            f"mean={s['mean_return']:.6f} | var={s['variance']:.6f}"
        )

    # اختبار الفلتر
    print("\n" + "=" * 70)
    print("اختبار الفلتر")
    print("=" * 70)
    test_signal = 1  # إشارة شراء
    filtered = filter_signals_by_regime(test_signal, result_2['current_regime'])
    print(f"   إشارة أصلية: {test_signal}")
    print(f"   إشارة مفلترة: {filtered}")
    print(f"   النظام الحالي: {result_2['current_regime']}")

    print("\n" + "=" * 70)
    print("اكتمل الاختبار")
    print("=" * 70 + "\n")
