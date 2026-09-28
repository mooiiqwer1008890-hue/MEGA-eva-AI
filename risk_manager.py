"""
risk_manager.py
===============
نظام إدارة مخاطر احترافي مبني على كتاب "Successful Algorithmic Trading" من QuantStart.

يحتوي على:
1. معيار كيلي (Kelly Criterion) مع Half-Kelly لحساب حجم الصفقة الأمثل.
2. قيمة المعرض للخطر (Value at Risk - VaR) لقياس أقصى خسارة متوقعة.
3. أقصى تراجع (Maximum Drawdown) لقياس أسوأ فترة خسارة.
4. نسبة شارب (Sharpe Ratio) لقياس العائد المعدل بالمخاطر.
5. نسبة سورتينو (Sortino Ratio) لقياس العائد المعدل بالتقلب السلبي.
6. تقرير أداء شامل (Performance Report).

الاستخدام:
----------
    from risk_manager import (
        calculate_kelly_position_size,
        calculate_var,
        calculate_max_drawdown,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        generate_performance_report
    )
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# ---------------------------------------------------------------------
# إعداد Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("risk_manager")


# ---------------------------------------------------------------------
# 1. معيار كيلي (Kelly Criterion)
# ---------------------------------------------------------------------
def calculate_kelly_position_size(win_rate, win_loss_ratio, balance, fraction=0.5):
    """
    تحسب حجم الصفقة الأمثل باستخدام معيار كيلي (Kelly Criterion).
    
    معيار كيلي هو معادلة رياضية تحدد النسبة المثالية من رأس المال
    التي يجب المخاطرة بها في كل صفقة لتعظيم النمو على المدى الطويل.
    
    نستخدم هنا "Half-Kelly" (fraction=0.5) لتقليل المخاطر، لأن
    Full Kelly يمكن أن يؤدي إلى تقلبات عنيفة.
    
    Parameters
    ----------
    win_rate : float
        نسبة الصفقات الرابحة (مثلاً 0.55 لـ 55%).
    win_loss_ratio : float
        نسبة متوسط الربح إلى متوسط الخسارة (مثلاً 1.5).
    balance : float
        رأس المال الحالي.
    fraction : float, optional
        معامل الأمان (0.5 للـ Half-Kelly، 1.0 للـ Full Kelly). الافتراضي 0.5.
    
    Returns
    -------
    position_size : float
        حجم الصفقة المقترح بالدولار. إذا كانت النتيجة سالبة، أرجع 0.
    
    Examples
    --------
    >>> calculate_kelly_position_size(0.55, 1.5, 1000, 0.5)
    83.33  # (تقريباً)
    """
    # التحقق من صحة المدخلات
    if win_rate <= 0 or win_rate >= 1:
        log.warning(f"win_rate غير صالح: {win_rate}. يجب أن يكون بين 0 و 1.")
        return 0.0
    if win_loss_ratio <= 0:
        log.warning(f"win_loss_ratio غير صالح: {win_loss_ratio}. يجب أن يكون > 0.")
        return 0.0
    if balance <= 0:
        log.warning(f"balance غير صالح: {balance}. يجب أن يكون > 0.")
        return 0.0
    
    # معادلة كيلي
    f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # إذا كانت f سالبة، فهذا يعني أن الاستراتيجية خاسرة
    if f <= 0:
        log.info(f"Kelly f = {f:.4f} (سالب). لا تتداول.")
        return 0.0
    
    # تطبيق معامل الأمان (Half-Kelly)
    half_kelly = fraction * f
    
    # حساب حجم الصفقة
    position_size = balance * half_kelly
    
    log.info(
        f"Kelly: win_rate={win_rate:.2%}, win_loss_ratio={win_loss_ratio:.2f}, "
        f"f={f:.4f}, half_kelly={half_kelly:.4f}, position_size=${position_size:.2f}"
    )
    
    return round(position_size, 2)


# ---------------------------------------------------------------------
# 2. قيمة المعرض للخطر (Value at Risk - VaR)
# ---------------------------------------------------------------------
def calculate_var(returns, confidence=0.95, portfolio_value=10000):
    """
    تحسب قيمة المعرض للخطر (VaR) باستخدام طريقة التباين-التغاير.
    
    VaR هي مقياس إحصائي يحدد أقصى خسارة متوقعة في فترة زمنية معينة
    عند مستوى ثقة معين. مثلاً، VaR = 500$ عند 95% يعني أن هناك
    احتمال 95% ألا تتجاوز خسارتك 500$ في اليوم التالي.
    
    Parameters
    ----------
    returns : np.array or pd.Series
        سلسلة العوائد (يومية أو ساعية).
    confidence : float, optional
        مستوى الثقة (0.95 أو 0.99). الافتراضي 0.95.
    portfolio_value : float, optional
        قيمة المحفظة. الافتراضي 10000.
    
    Returns
    -------
    var : float
        أقصى خسارة متوقعة بالدولار.
    
    Examples
    --------
    >>> returns = np.random.normal(0, 0.02, 1000)
    >>> calculate_var(returns, 0.95, 10000)
    329.00  # (تقريباً)
    """
    # التحقق من صحة المدخلات
    if len(returns) == 0:
        log.warning("returns فارغة. لا يمكن حساب VaR.")
        return 0.0
    if confidence <= 0 or confidence >= 1:
        log.warning(f"confidence غير صالح: {confidence}. يجب أن يكون بين 0 و 1.")
        return 0.0
    if portfolio_value <= 0:
        log.warning(f"portfolio_value غير صالح: {portfolio_value}.")
        return 0.0
    
    # تحويل إلى numpy array إذا لزم الأمر
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # حساب المتوسط والانحراف المعياري
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    # حساب alpha (النسبة المئوية للخسارة)
    alpha = norm.ppf(1 - confidence, mu, sigma)
    
    # حساب VaR
    var = portfolio_value - portfolio_value * (alpha + 1)
    
    log.info(
        f"VaR: confidence={confidence:.0%}, mu={mu:.4f}, sigma={sigma:.4f}, "
        f"alpha={alpha:.4f}, VaR=${var:.2f}"
    )
    
    return round(var, 2)


# ---------------------------------------------------------------------
# 3. أقصى تراجع (Maximum Drawdown)
# ---------------------------------------------------------------------
def calculate_max_drawdown(equity_curve):
    """
    تحسب أقصى تراجع (Maximum Drawdown) ومدة التراجع.
    
    Maximum Drawdown هو أكبر انخفاض من قمة إلى قاع في منحنى رأس المال.
    مثال: إذا وصل رأس مالك إلى 1000$ ثم انخفض إلى 700$، فإن
    أقصى تراجع هو 30%. هذا المقياس حاسم لتقييم مخاطر الاستراتيجية.
    
    Parameters
    ----------
    equity_curve : pd.Series
        منحنى رأس المال (القيم عبر الزمن).
    
    Returns
    -------
    tuple
        (max_drawdown, drawdown_duration)
        - max_drawdown (float): أقصى تراجع (رقم سالب).
        - drawdown_duration (int): عدد الفترات التي كان فيها التراجع.
    
    Examples
    --------
    >>> equity = pd.Series([100, 110, 105, 90, 95, 100])
    >>> calculate_max_drawdown(equity)
    (-0.1818, 3)
    """
    # التحقق من صحة المدخلات
    if len(equity_curve) == 0:
        log.warning("equity_curve فارغة. لا يمكن حساب Max Drawdown.")
        return (0.0, 0)
    
    # حساب القمة المتراكمة (Running Max)
    running_max = equity_curve.cummax()
    
    # حساب التراجع (Drawdown)
    drawdown = (equity_curve - running_max) / running_max
    
    # أقصى تراجع
    max_drawdown = drawdown.min()
    
    # مدة التراجع (عدد الفترات التي كان فيها التراجع سالباً)
    drawdown_duration = (drawdown < 0).sum()
    
    log.info(
        f"Max Drawdown: {max_drawdown:.2%}, Duration: {drawdown_duration} periods"
    )
    
    return (round(max_drawdown, 4), int(drawdown_duration))


# ---------------------------------------------------------------------
# 4. نسبة شارب (Sharpe Ratio)
# ---------------------------------------------------------------------
def calculate_sharpe_ratio(returns, periods=252, risk_free_rate=0.02):
    """
    تحسب نسبة شارب (Sharpe Ratio).
    
    نسبة شارب هي مقياس للعائد المعدل بالمخاطر. تقيس كم عائد إضافي
    تحصل عليه مقابل كل وحدة مخاطرة (تقلب). نسبة شارب > 1 تعتبر جيدة،
    و > 2 ممتازة، و > 3 استثنائية.
    
    Parameters
    ----------
    returns : pd.Series
        سلسلة العوائد.
    periods : int, optional
        عدد الفترات في السنة (252 لليومي، 1638 للساعي). الافتراضي 252.
    risk_free_rate : float, optional
        سعر الفائدة الخالي من المخاطر (سنوي). الافتراضي 0.02 (2%).
    
    Returns
    -------
    sharpe : float
        نسبة شارب.
    
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    >>> calculate_sharpe_ratio(returns)
    0.79  # (تقريباً)
    """
    # التحقق من صحة المدخلات
    if len(returns) < 2:
        log.warning("returns أقل من قيمتين. لا يمكن حساب Sharpe.")
        return 0.0
    
    # حساب العوائد الزائدة (Excess Returns)
    excess_returns = returns - risk_free_rate / periods
    
    # تجنب القسمة على صفر
    std = excess_returns.std()
    if std == 0:
        log.warning("الانحراف المعياري للعوائد = 0. لا يمكن حساب Sharpe.")
        return 0.0
    
    # حساب نسبة شارب
    sharpe = (excess_returns.mean() / std) * np.sqrt(periods)
    
    log.info(f"Sharpe Ratio: {sharpe:.4f}")
    
    return round(sharpe, 4)


# ---------------------------------------------------------------------
# 5. نسبة سورتينو (Sortino Ratio)
# ---------------------------------------------------------------------
def calculate_sortino_ratio(returns, periods=252, risk_free_rate=0.02):
    """
    تحسب نسبة سورتينو (Sortino Ratio).
    
    نسبة سورتينو هي نسخة محسنة من نسبة شارب. الفرق أنها تركز فقط
    على التقلب السلبي (Downside Volatility)، وليس كل التقلب. هذا
    منطقي لأن المستثمرين يخافون من الخسائر، وليس من الأرباح.
    
    Parameters
    ----------
    returns : pd.Series
        سلسلة العوائد.
    periods : int, optional
        عدد الفترات في السنة. الافتراضي 252.
    risk_free_rate : float, optional
        سعر الفائدة الخالي من المخاطر. الافتراضي 0.02.
    
    Returns
    -------
    sortino : float
        نسبة سورتينو.
    
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    >>> calculate_sortino_ratio(returns)
    1.12  # (تقريباً)
    """
    # التحقق من صحة المدخلات
    if len(returns) < 2:
        log.warning("returns أقل من قيمتين. لا يمكن حساب Sortino.")
        return 0.0
    
    # حساب العوائد الزائدة
    excess_returns = returns - risk_free_rate / periods
    
    # فصل العوائد السلبية فقط
    downside_returns = excess_returns[excess_returns < 0]
    
    # تجنب القسمة على صفر
    if len(downside_returns) == 0:
        log.warning("لا توجد عوائد سلبية. Sortino غير معرف.")
        return 0.0
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        log.warning("الانحراف المعياري السلبي = 0. Sortino غير معرف.")
        return 0.0
    
    # حساب نسبة سورتينو
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods)
    
    log.info(f"Sortino Ratio: {sortino:.4f}")
    
    return round(sortino, 4)


# ---------------------------------------------------------------------
# 6. تقرير الأداء الشامل (Performance Report)
# ---------------------------------------------------------------------
def generate_performance_report(equity_curve, trades_df):
    """
    تولد تقريراً كاملاً بالأداء.
    
    يجمع هذا التقرير كل مقاييس الأداء في مكان واحد، بما في ذلك:
    - العائد الإجمالي
    - نسبة شارب
    - نسبة سورتينو
    - أقصى تراجع
    - نسبة الصفقات الرابحة
    - نسبة الربح/الخسارة
    
    Parameters
    ----------
    equity_curve : pd.Series
        منحنى رأس المال.
    trades_df : pd.DataFrame
        جدول الصفقات. يجب أن يحتوي على عمود 'pnl' (الربح/الخسارة).
    
    Returns
    -------
    report : dict
        قاموس يحتوي على جميع المقاييس.
    
    Examples
    --------
    >>> equity = pd.Series([100, 110, 105, 115, 120])
    >>> trades = pd.DataFrame({'pnl': [10, -5, 10, 5]})
    >>> report = generate_performance_report(equity, trades)
    >>> print(report['total_return'])
    0.20
    """
    # التحقق من صحة المدخلات
    if len(equity_curve) < 2:
        log.warning("equity_curve قصيرة جداً. لا يمكن توليد تقرير.")
        return {}
    if 'pnl' not in trades_df.columns:
        log.warning("trades_df لا يحتوي على عمود 'pnl'.")
        return {}
    
    # 1. العائد الإجمالي
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # 2. العوائد الدورية
    returns = equity_curve.pct_change().dropna()
    
    # 3. نسبة شارب
    sharpe = calculate_sharpe_ratio(returns)
    
    # 4. نسبة سورتينو
    sortino = calculate_sortino_ratio(returns)
    
    # 5. أقصى تراجع
    max_dd, dd_duration = calculate_max_drawdown(equity_curve)
    
    # 6. إحصائيات الصفقات
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
    
    # تجنب القسمة على صفر
    if avg_loss != 0 and not np.isnan(avg_loss):
        win_loss_ratio = abs(avg_win / avg_loss)
    else:
        win_loss_ratio = 1.0  # قيمة افتراضية آمنة
    
    # 7. صافي الربح
    net_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
    
    # تجميع التقرير
    report = {
        'total_return': round(total_return, 4),
        'net_profit': round(net_profit, 2),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'drawdown_duration': dd_duration,
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'win_loss_ratio': round(win_loss_ratio, 4),
    }
    
    log.info("=" * 60)
    log.info("PERFORMANCE REPORT")
    log.info("=" * 60)
    for key, value in report.items():
        log.info(f"  {key}: {value}")
    log.info("=" * 60)
    
    return report


# ---------------------------------------------------------------------
# اختبار الدوال (Test Section)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("اختبار نظام إدارة المخاطر")
    print("=" * 60 + "\n")
    
    # 1. اختبار معيار كيلي
    print("1. اختبار معيار كيلي (Kelly Criterion):")
    size = calculate_kelly_position_size(0.55, 1.5, 1000, 0.5)
    print(f"   Kelly Position Size: ${size:.2f}\n")
    
    # 2. اختبار VaR
    print("2. اختبار قيمة المعرض للخطر (VaR):")
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 1000)
    var = calculate_var(returns, 0.95, 10000)
    print(f"   VaR (95%): ${var:.2f}\n")
    
    # 3. اختبار أقصى تراجع
    print("3. اختبار أقصى تراجع (Max Drawdown):")
    equity = pd.Series(np.cumsum(np.random.normal(0, 0.01, 1000)) + 100)
    max_dd, dd_dur = calculate_max_drawdown(equity)
    print(f"   Max Drawdown: {max_dd:.2%}, Duration: {dd_dur}\n")
    
    # 4. اختبار نسبة شارب
    print("4. اختبار نسبة شارب (Sharpe Ratio):")
    sharpe = calculate_sharpe_ratio(pd.Series(returns))
    print(f"   Sharpe Ratio: {sharpe:.4f}\n")
    
    # 5. اختبار نسبة سورتينو
    print("5. اختبار نسبة سورتينو (Sortino Ratio):")
    sortino = calculate_sortino_ratio(pd.Series(returns))
    print(f"   Sortino Ratio: {sortino:.4f}\n")
    
    # 6. اختبار تقرير الأداء
    print("6. اختبار تقرير الأداء الشامل:")
    trades = pd.DataFrame({
        'pnl': [10, -5, 15, -3, 8, -2, 12, 5, -4, 7]
    })
    report = generate_performance_report(equity, trades)
    print(f"   Total Return: {report['total_return']:.2%}")
    print(f"   Sharpe: {report['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {report['max_drawdown']:.2%}")
    print(f"   Win Rate: {report['win_rate']:.2%}")
    print(f"   Win/Loss Ratio: {report['win_loss_ratio']:.4f}")
    
    print("\n" + "=" * 60)
    print("اكتمل الاختبار بنجاح")
    print("=" * 60 + "\n")
