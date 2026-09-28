"""
cointegration_pairs.py
======================
أدوات اختبار التكامل المشترك وتداول الأزواج.

المصدر: كتاب "Advanced Algorithmic Trading" - الفصل 12
"""

import logging
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

log = logging.getLogger("cointegration_pairs")


# ---------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------
def fetch_closes(symbol, interval="15m", limit=500):
    """
    يجلب أسعار الإغلاق من Binance Data API (غير محظور جغرافياً).
    
    Parameters
    ----------
    symbol : str
        مثال: "BTCUSDT"
    interval : str
        مثال: "15m", "1h", "1d"
    limit : int
        عدد الشموع
    
    Returns
    -------
    pd.Series
        سلسلة أسعار الإغلاق مع فهرس زمني
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()
    
    timestamps = [int(k[0]) for k in raw]
    closes = [float(k[4]) for k in raw]
    
    index = pd.to_datetime(timestamps, unit="ms")
    series = pd.Series(closes, index=index, name=symbol)
    return series


# ---------------------------------------------------------------------
# COINTEGRATION TEST
# ---------------------------------------------------------------------
def test_cointegration(series1, series2):
    """
    يختبر التكامل المشترك بين سلسلتين.
    
    يستخدم اختبار Engle-Granger (coint) من statsmodels.
    
    Parameters
    ----------
    series1, series2 : pd.Series
        سلسلتان زمنيتان بنفس الطول
    
    Returns
    -------
    dict
        {
            "is_cointegrated": bool,
            "p_value": float,
            "hedge_ratio": float,
            "intercept": float,
            "spread": pd.Series
        }
    """
    if len(series1) != len(series2):
        raise ValueError("السلسلتان يجب أن يكونا بنفس الطول")
    if len(series1) < 30:
        raise ValueError("السلسلة قصيرة جداً (تحتاج 30 نقطة على الأقل)")
    
    # اختبار Engle-Granger
    score, p_value, _ = coint(series1, series2)
    is_cointegrated = p_value < 0.05
    
    # حساب نسبة التحوط (Hedge Ratio) عبر OLS
    # series1 = intercept + hedge_ratio * series2 + spread
    y = series1.values
    x = sm.add_constant(series2.values)
    model = sm.OLS(y, x).fit()
    
    intercept = float(model.params[0])
    hedge_ratio = float(model.params[1])
    
    # حساب السبريد (Spread)
    spread = series1 - (hedge_ratio * series2 + intercept)
    
    log.info(
        f"Cointegration test: p_value={p_value:.4f}, "
        f"is_cointegrated={is_cointegrated}, "
        f"hedge_ratio={hedge_ratio:.4f}"
    )
    
    return {
        "is_cointegrated": is_cointegrated,
        "p_value": float(p_value),
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread": spread,
    }


# ---------------------------------------------------------------------
# Z-SCORE
# ---------------------------------------------------------------------
def calculate_zscore(spread, window=20):
    """
    يحسب Z-Score للسبريد (Spread).
    
    Parameters
    ----------
    spread : pd.Series
    window : int
        نافذة المتوسط المتحرك
    
    Returns
    -------
    pd.Series
        Z-Score
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    # تجنب القسمة على صفر
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore


# ---------------------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------------------
def generate_pairs_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    يولد إشارات تداول الأزواج.
    
    Parameters
    ----------
    zscore : pd.Series
    entry_threshold : float
        عتبة الدخول (افتراضي 2.0)
    exit_threshold : float
        عتبة الخروج (افتراضي 0.5)
    
    Returns
    -------
    pd.Series
        إشارة: 1 (شراء spread)، -1 (بيع spread)، 0 (خروج/انتظار)
    """
    signals = pd.Series(0, index=zscore.index)
    
    # حالة الشراء: Z < -entry_threshold (السبريد منخفض جداً)
    signals[zscore < -entry_threshold] = 1
    
    # حالة البيع: Z > entry_threshold (السبريد مرتفع جداً)
    signals[zscore > entry_threshold] = -1
    
    # حالة الخروج: |Z| < exit_threshold
    signals[zscore.abs() < exit_threshold] = 0
    
    return signals


def generate_stateful_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    يولد إشارات مع حالة (State Machine).
    
    لا يفتح صفقة جديدة إذا كانت هناك صفقة مفتوحة بالفعل.
    """
    signals = pd.Series(0, index=zscore.index)
    position = 0  # 0 = لا شيء، 1 = long spread، -1 = short spread
    
    for i in range(len(zscore)):
        z = zscore.iloc[i]
        
        if pd.isna(z):
            signals.iloc[i] = 0
            continue
        
        if position == 0:
            if z < -entry_threshold:
                position = 1
                signals.iloc[i] = 1  # فتح long spread
            elif z > entry_threshold:
                position = -1
                signals.iloc[i] = -1  # فتح short spread
            else:
                signals.iloc[i] = 0
        elif position == 1:
            if abs(z) < exit_threshold:
                position = 0
                signals.iloc[i] = 0  # إغلاق
            else:
                signals.iloc[i] = 0  # استمرار (لا إشارة جديدة)
        elif position == -1:
            if abs(z) < exit_threshold:
                position = 0
                signals.iloc[i] = 0  # إغلاق
            else:
                signals.iloc[i] = 0
    
    return signals


# ---------------------------------------------------------------------
# PAIRS ANALYSIS
# ---------------------------------------------------------------------
def analyze_pairs(symbol1, symbol2, interval="15m", limit=500):
    """
    تحليل كامل لزوج من العملات.
    
    Returns
    -------
    dict
        يحتوي على كل النتائج: cointegration, spread, zscore, signals
    """
    log.info(f"Analyzing pair: {symbol1} / {symbol2}")
    
    # جلب البيانات
    s1 = fetch_closes(symbol1, interval, limit)
    s2 = fetch_closes(symbol2, interval, limit)
    
    # دمج على نفس الفهرس الزمني
    df = pd.DataFrame({symbol1: s1, symbol2: s2}).dropna()
    if len(df) < 50:
        log.warning(f"بيانات غير كافية للزوج {symbol1}/{symbol2}")
        return None
    
    s1 = df[symbol1]
    s2 = df[symbol2]
    
    # اختبار التكامل المشترك
    result = test_cointegration(s1, s2)
    
    # حساب Z-Score
    zscore = calculate_zscore(result["spread"], window=20)
    
    # توليد الإشارات
    signals = generate_stateful_signals(zscore)
    
    # إحصائيات
    num_buy = (signals == 1).sum()
    num_sell = (signals == -1).sum()
    current_z = zscore.iloc[-1] if not pd.isna(zscore.iloc[-1]) else 0.0
    current_signal = signals.iloc[-1]
    
    summary = {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "is_cointegrated": result["is_cointegrated"],
        "p_value": result["p_value"],
        "hedge_ratio": result["hedge_ratio"],
        "intercept": result["intercept"],
        "spread": result["spread"],
        "zscore": zscore,
        "signals": signals,
        "current_z": current_z,
        "current_signal": current_signal,
        "num_buy_signals": int(num_buy),
        "num_sell_signals": int(num_sell),
        "price1": s1.iloc[-1],
        "price2": s2.iloc[-1],
    }
    
    log.info(
        f"Pair {symbol1}/{symbol2}: cointegrated={result['is_cointegrated']}, "
        f"p={result['p_value']:.4f}, hedge_ratio={result['hedge_ratio']:.4f}, "
        f"current_z={current_z:.2f}, signal={current_signal}"
    )
    
    return summary


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # اختبار على BTCUSDT و ETHUSDT
    pairs_to_test = [
        ("BTCUSDT", "ETHUSDT"),
        ("ETHUSDT", "BNBUSDT"),
        ("BTCUSDT", "SOLUSDT"),
    ]
    
    print("\n" + "=" * 70)
    print("اختبار التكامل المشترك على أزواج العملات")
    print("=" * 70 + "\n")
    
    for s1, s2 in pairs_to_test:
        try:
            result = analyze_pairs(s1, s2, interval="15m", limit=500)
            if result is None:
                continue
            
            print(f"\n📊 زوج: {s1} / {s2}")
            print(f"   ├─ متكامل مشتركياً؟ {'✅ نعم' if result['is_cointegrated'] else '❌ لا'}")
            print(f"   ├─ p-value: {result['p_value']:.4f}")
            print(f"   ├─ Hedge Ratio: {result['hedge_ratio']:.4f}")
            print(f"   ├─ Intercept: {result['intercept']:.4f}")
            print(f"   ├─ Z-Score الحالي: {result['current_z']:.2f}")
            print(f"   ├─ الإشارة الحالية: {result['current_signal']}")
            print(f"   ├─ عدد إشارات الشراء: {result['num_buy_signals']}")
            print(f"   └─ عدد إشارات البيع: {result['num_sell_signals']}")
        except Exception as e:
            print(f"\n❌ خطأ في الزوج {s1}/{s2}: {e}")
    
    print("\n" + "=" * 70)
    print("اكتمل الاختبار")
    print("=" * 70 + "\n")
