"""
garch_model.py
==============
نمذجة التقلب (Volatility) باستخدام GARCH.

المصدر:
- كتاب "Advanced Algorithmic Trading" - الفصل 11
- كتاب "Python Trader" - الفصل 11

الفكرة:
- التقلب في الأسواق المالية يتجمع (Volatility Clustering).
- GARCH يتنبأ بالتقلب المستقبلي بناءً على التقلب السابق.
- نستخدم التنبؤ لتعديل حجم الصفقة (Position Sizing).

الفاصل الزمني الموصى به: 4H
"""

import logging
import warnings
import numpy as np
import pandas as pd
import requests

# تجاهل تحذيرات arch (كثيرة جداً)
warnings.filterwarnings("ignore")

from arch import arch_model

log = logging.getLogger("garch_model")


# ---------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------
def fetch_returns(symbol, interval="4h", limit=1000):
    """
    يجلب أسعار الإغلاق من Binance Data API ويحسب العوائد اللوغاريتمية (بـ %).
    
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
        سلسلة العوائد بنسبة مئوية
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
    log_returns = np.log(closes / closes.shift(1)).dropna() * 100.0  # النسبة المئوية
    return log_returns


# ---------------------------------------------------------------------
# GARCH MODEL
# ---------------------------------------------------------------------
def fit_garch(returns, p=1, q=1, dist="t"):
    """
    يدرب نموذج GARCH(p,q) على العوائد.
    
    Parameters
    ----------
    returns : pd.Series
        سلسلة العوائد (بالنسبة المئوية)
    p : int
        عدد معاملات GARCH (افتراضي 1)
    q : int
        عدد معاملات ARCH (افتراضي 1)
    dist : str
        "normal" أو "t" (افتراضي "t" - أفضر للبيانات المالية)
    
    Returns
    -------
    dict
        {
            "model": arch_model,
            "result": fitted model,
            "aic": float,
            "bic": float,
            "params": dict
        }
    """
    if len(returns) < 100:
        raise ValueError("السلسلة قصيرة جداً (تحتاج 100 نقطة على الأقل)")

    # تدريب GARCH
    model = arch_model(returns, vol="GARCH", p=p, q=q, dist=dist, rescale=False)
    result = model.fit(disp="off", show_warning=False)

    log.info(
        f"GARCH({p},{q}) trained: AIC={result.aic:.2f}, BIC={result.bic:.2f}"
    )
    log.info(f"Parameters: {result.params.to_dict()}")

    return {
        "model": model,
        "result": result,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "params": result.params.to_dict(),
    }


def forecast_volatility(returns, horizon=1, p=1, q=1):
    """
    يتنبأ بالتقلب للفترات القادمة.
    
    Parameters
    ----------
    returns : pd.Series
    horizon : int
        عدد الفترات للتنبؤ (افتراضي 1)
    p, q : int
        معاملات GARCH
    
    Returns
    -------
    dict
        {
            "forecast_vol": float,       # التقلب المتوقع (بالنسبة المئوية)
            "annualized_vol": float,     # التقلب السنوي
            "current_vol": float,        # التقلب الحالي
            "vol_ratio": float           # نسبة التقلب المتوقع إلى الحالي
        }
    """
    result = fit_garch(returns, p=p, q=q)
    forecast = result["result"].forecast(horizon=horizon)
    variance_forecast = forecast.variance.values[-1, :]

    # التقلب المتوقع (الجذر التربيعي)
    forecast_vol = float(np.sqrt(variance_forecast[0]))

    # التقلب السنوي (بافتراض 6 شمعات 4H يومياً × 365 = 2190)
    periods_per_year = 6 * 365
    annualized_vol = forecast_vol * np.sqrt(periods_per_year)

    # التقلب الحالي (آخر قيمة Conditional Volatility)
    conditional_vol = result["result"].conditional_volatility
    current_vol = float(conditional_vol.iloc[-1])

    # نسبة التقلب المتوقع إلى الحالي
    vol_ratio = forecast_vol / current_vol if current_vol > 0 else 1.0

    log.info(
        f"Volatility forecast: {forecast_vol:.4f}% | "
        f"Annualized: {annualized_vol:.2f}% | "
        f"Current: {current_vol:.4f}% | "
        f"Ratio: {vol_ratio:.2f}"
    )

    return {
        "forecast_vol": round(forecast_vol, 4),
        "annualized_vol": round(annualized_vol, 2),
        "current_vol": round(current_vol, 4),
        "vol_ratio": round(vol_ratio, 4),
    }


# ---------------------------------------------------------------------
# VOLATILITY REGIME
# ---------------------------------------------------------------------
def classify_volatility(vol_ratio):
    """
    يصنف التقلب إلى 3 فئات.
    
    Parameters
    ----------
    vol_ratio : float
        نسبة التقلب المتوقع إلى الحالي
    
    Returns
    -------
    str
        "LOW", "NORMAL", "HIGH"
    """
    if vol_ratio < 0.8:
        return "LOW"       # تقلب منخفض - فرصة جيدة
    elif vol_ratio > 1.5:
        return "HIGH"      # تقلب مرتفع - خطر
    else:
        return "NORMAL"    # تقلب طبيعي


# ---------------------------------------------------------------------
# POSITION SIZE ADJUSTMENT
# ---------------------------------------------------------------------
def adjust_position_size_by_volatility(base_size, vol_ratio, min_size=0.3, max_size=1.5):
    """
    يعدل حجم الصفقة بناءً على التقلب.
    
    القاعدة:
    - تقلب منخفض (ratio < 0.8): زيادة الحجم (×1.5).
    - تقلب طبيعي (0.8 ≤ ratio ≤ 1.5): الحجم الطبيعي.
    - تقلب مرتفع (ratio > 1.5): تقليل الحجم (×0.3).
    
    Parameters
    ----------
    base_size : float
        حجم الصفقة الأساسي (من Kelly)
    vol_ratio : float
        نسبة التقلب
    min_size : float
        الحد الأدنى لمعامل التعديل
    max_size : float
        الحد الأعلى لمعامل التعديل
    
    Returns
    -------
    float
        حجم الصفقة المعدل
    """
    if vol_ratio < 0.8:
        multiplier = 1.5  # تقلب منخفض → حجم أكبر
    elif vol_ratio > 1.5:
        multiplier = 0.3  # تقلب مرتفع → حجم أصغر
    else:
        multiplier = 1.0  # تقلب طبيعي

    # تحديد حدود المعامل
    multiplier = max(min_size, min(multiplier, max_size))
    adjusted = base_size * multiplier

    log.info(
        f"Position adjustment: base={base_size:.2f} × {multiplier:.2f} = {adjusted:.2f}"
    )

    return round(adjusted, 2)


# ---------------------------------------------------------------------
# VOLATILITY STATS
# ---------------------------------------------------------------------
def get_volatility_stats(returns):
    """
    يعيد إحصائيات كاملة عن التقلب.
    """
    result = fit_garch(returns)
    forecast = forecast_volatility(returns)

    conditional_vol = result["result"].conditional_volatility
    stats = {
        "current_vol": forecast["current_vol"],
        "forecast_vol": forecast["forecast_vol"],
        "annualized_vol": forecast["annualized_vol"],
        "vol_ratio": forecast["vol_ratio"],
        "vol_regime": classify_volatility(forecast["vol_ratio"]),
        "mean_vol": round(float(conditional_vol.mean()), 4),
        "max_vol": round(float(conditional_vol.max()), 4),
        "min_vol": round(float(conditional_vol.min()), 4),
        "aic": result["aic"],
        "bic": result["bic"],
    }
    return stats


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 70)
    print("اختبار GARCH Model - الفاصل الزمني: 4H")
    print("=" * 70 + "\n")

    # اختبار على BTCUSDT و ETHUSDT
    symbols_to_test = ["BTCUSDT", "ETHUSDT"]

    for symbol in symbols_to_test:
        try:
            print(f"\n{'=' * 70}")
            print(f"📊 العملة: {symbol}")
            print(f"{'=' * 70}")

            returns = fetch_returns(symbol, interval="4h", limit=1000)
            print(f"عدد العوائد: {len(returns)}")
            print(f"متوسط العائد: {returns.mean():.4f}%")
            print(f"انحراف العائد: {returns.std():.4f}%\n")

            # إحصائيات التقلب
            stats = get_volatility_stats(returns)

            print(f"📈 التقلب الحالي: {stats['current_vol']:.4f}%")
            print(f"📈 التقلب المتوقع: {stats['forecast_vol']:.4f}%")
            print(f"📈 التقلب السنوي: {stats['annualized_vol']:.2f}%")
            print(f"📊 نسبة التقلب: {stats['vol_ratio']:.4f}")
            print(f"🎯 النظام: {stats['vol_regime']}")
            print(f"📉 متوسط التقلب: {stats['mean_vol']:.4f}%")
            print(f"📈 أقصى تقلب: {stats['max_vol']:.4f}%")
            print(f"📉 أدنى تقلب: {stats['min_vol']:.4f}%")
            print(f"📊 AIC: {stats['aic']:.2f}")
            print(f"📊 BIC: {stats['bic']:.2f}\n")

            # اختبار تعديل حجم الصفقة
            base_size = 100.0  # حجم صفقة افتراضي
            adjusted = adjust_position_size_by_volatility(base_size, stats['vol_ratio'])
            print(f"💰 حجم الصفقة الأساسي: ${base_size:.2f}")
            print(f"💰 حجم الصفقة المعدل: ${adjusted:.2f}")
            print(f"💰 الفرق: ${adjusted - base_size:+.2f}")

        except Exception as e:
            print(f"\n❌ خطأ في {symbol}: {e}")

    print("\n" + "=" * 70)
    print("اكتمل الاختبار")
    print("=" * 70 + "\n")
