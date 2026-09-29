"""
sentiment_filter.py
===================
دمج تحليل المشاعر مع إشارات التداول.

يعمل مع CoinMarketCap Keyless API (بدون API Key).
"""

import logging
from typing import Dict, List, Optional

from news_fetcher import (
    update_news_cache,
    get_cached_news,
    COIN_ID_MAP,
)
from sentiment_analysis import SentimentAnalyzer

log = logging.getLogger("sentiment_filter")


# ============================================================
# CONFIG
# ============================================================
BEARISH_THRESHOLD = -0.30
BULLISH_THRESHOLD = 0.30
MIN_NEWS_COUNT = 3


class SentimentFilter:
    """فلتر المشاعر — يدمج الأخبار مع Z-score."""
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self._cache: Dict[str, Dict] = {}
        log.info("[SENTIMENT_FILTER] تم التهيئة")
    
    def get_sentiment(self, symbol: str, force_refresh: bool = False) -> Dict:
        """الحصول على مشاعر عملة معينة."""
        if not force_refresh and symbol in self._cache:
            return self._cache[symbol]
        
        try:
            base = symbol.replace("USDT", "").replace("USD", "").upper()
            
            # تحديث الأخبار
            if force_refresh or not get_cached_news([base]):
                update_news_cache(symbols=[base])
            
            # جلب الأخبار المخزنة
            news = get_cached_news([base])
            
            if not news:
                result = {
                    "score": 0.0, "label": "neutral", "count": 0,
                    "positive": 0, "negative": 0, "neutral": 0,
                }
            else:
                analyzed = self.analyzer.analyze_news_batch(news)
                result = self.analyzer.aggregate_sentiment(analyzed, symbol)
            
            self._cache[symbol] = result
            return result
        
        except Exception as e:
            log.exception(f"[SENTIMENT_FILTER] فشل لـ {symbol}: {e}")
            return {
                "score": 0.0, "label": "neutral", "count": 0,
                "positive": 0, "negative": 0, "neutral": 0,
                "error": str(e),
            }
    
    def filter_signal(self, signal_side: str, symbol: str) -> Dict:
        """فلترة إشارة بناءً على المشاعر."""
        sentiment = self.get_sentiment(symbol)
        score = sentiment["score"]
        label = sentiment["label"]
        count = sentiment["count"]
        
        if count < MIN_NEWS_COUNT:
            return {
                "allowed": True,
                "multiplier": 1.0,
                "reason": f"عدد أخبار قليل ({count}) — لا يمكن الحكم",
                "sentiment": sentiment,
            }
        
        if signal_side == "BUY":
            if label == "bearish" and score < BEARISH_THRESHOLD:
                return {
                    "allowed": False,
                    "multiplier": 0.0,
                    "reason": f"مشاعر سلبية قوية ({score:+.2f}) — تجنب catch a falling knife",
                    "sentiment": sentiment,
                }
            elif label == "bullish" and score > BULLISH_THRESHOLD:
                return {
                    "allowed": True,
                    "multiplier": 1.2,
                    "reason": f"مشاعر إيجابية قوية ({score:+.2f}) — تعزيز الصفقة",
                    "sentiment": sentiment,
                }
            elif label == "bearish":
                return {
                    "allowed": True,
                    "multiplier": 0.7,
                    "reason": f"مشاعر سلبية ({score:+.2f}) — تقليل حجم الصفقة",
                    "sentiment": sentiment,
                }
            else:
                return {
                    "allowed": True,
                    "multiplier": 1.0,
                    "reason": f"مشاعر محايدة ({score:+.2f})",
                    "sentiment": sentiment,
                }
        
        elif signal_side == "SELL":
            if label == "bullish" and score > BULLISH_THRESHOLD:
                return {
                    "allowed": False,
                    "multiplier": 0.0,
                    "reason": f"مشاعر إيجابية قوية ({score:+.2f}) — تجنب short squeeze",
                    "sentiment": sentiment,
                }
            elif label == "bearish" and score < BEARISH_THRESHOLD:
                return {
                    "allowed": True,
                    "multiplier": 1.2,
                    "reason": f"مشاعر سلبية قوية ({score:+.2f}) — تعزيز البيع",
                    "sentiment": sentiment,
                }
            else:
                return {
                    "allowed": True,
                    "multiplier": 1.0,
                    "reason": f"مشاعر محايدة/سلبية ({score:+.2f})",
                    "sentiment": sentiment,
                }
        
        return {
            "allowed": True, "multiplier": 1.0,
            "reason": "إشارة غير معروفة", "sentiment": sentiment,
        }
    
    def summary(self, symbol: str) -> str:
        """ملخص نصي لحالة المشاعر."""
        s = self.get_sentiment(symbol)
        emoji = "🟢" if s["label"] == "bullish" else "🔴" if s["label"] == "bearish" else "⚪"
        return (
            f"{emoji} {symbol}: {s['label']} "
            f"({s['score']:+.3f}) — {s['count']} خبر"
        )


# ============================================================
# Singleton
# ============================================================
_global_filter: Optional[SentimentFilter] = None


def get_sentiment_filter() -> SentimentFilter:
    """الحصول على instance عام."""
    global _global_filter
    if _global_filter is None:
        _global_filter = SentimentFilter()
    return _global_filter


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    sf = get_sentiment_filter()
    
    print("\n" + "=" * 70)
    print("تحديث الأخبار من CoinMarketCap...")
    print("=" * 70)
    
    update_news_cache(symbols=["BTC", "ETH", "BNB", "SOL", "XRP"])
    
    print("\n" + "=" * 70)
    print("اختبار الفلتر على 5 عملات")
    print("=" * 70)
    
    for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]:
        print(f"\n{sf.summary(symbol)}")
        
        result = sf.filter_signal("BUY", symbol)
        emoji = "✅" if result["allowed"] else "❌"
        print(f"   {emoji} BUY: {result['reason']}")
