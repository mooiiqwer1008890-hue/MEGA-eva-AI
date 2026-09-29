"""
sentiment_analysis.py
=====================
تحليل المشاعر (Sentiment Analysis) لعناوين الأخبار.

المصدر:
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- كتاب "Advanced Algorithmic Trading" - Kaabar
- كتاب "Deep Learning for Finance" - Kaabar

الفكرة:
- VADER جيد للأخبار العامة
- لكن لا يعرف مصطلحات الكريبتو (ATH, hodl, rugpull, FUD)
- لذا نضيف قاموساً مخصصاً للكريبتو
- النتيجة: نقاط (-1 إلى +1) لكل عنوان

التصنيف النهائي:
- score > +0.15 → BULLISH 🟢
- score < -0.15 → BEARISH 🔴
- خلاف ذلك → NEUTRAL ⚪
"""

import logging
from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger("sentiment_analysis")


# ============================================================
# قاموس مصطلحات الكريبتو
# ============================================================
CRYPTO_LEXICON = {
    # === إيجابي قوي ===
    "bullish": 2.5,
    "moon": 2.0,
    "mooning": 2.5,
    "ath": 2.5,
    "all-time high": 2.5,
    "rally": 2.0,
    "surge": 2.0,
    "soar": 2.5,
    "soaring": 2.5,
    "breakout": 1.8,
    "adoption": 1.5,
    "institutional": 1.5,
    "etf": 1.5,
    "approval": 2.0,
    "approved": 2.0,
    "halving": 1.5,
    "hodl": 1.5,
    "hodler": 1.5,
    "whale": 1.0,
    "accumulation": 1.5,
    "buy-the-dip": 1.5,
    "inflow": 1.2,
    "partnership": 1.5,
    "upgrade": 1.5,
    "mainstream": 1.2,
    
    # === سلبي قوي ===
    "bearish": -2.5,
    "dump": -2.5,
    "dumping": -2.5,
    "rugpull": -3.5,
    "rug": -2.5,
    "rugged": -3.5,
    "crash": -3.0,
    "crashing": -3.0,
    "plunge": -2.5,
    "plunging": -2.5,
    "plummet": -2.5,
    "hack": -3.0,
    "hacked": -3.0,
    "exploit": -3.0,
    "exploited": -3.0,
    "breach": -2.5,
    "breached": -2.5,
    "fud": -2.0,
    "ban": -2.5,
    "banned": -2.5,
    "crackdown": -2.5,
    "liquidation": -2.0,
    "liquidated": -2.0,
    "outflow": -1.2,
    "correction": -1.0,
    "dump": -2.5,
    "collapse": -3.0,
    "collapsed": -3.0,
    "bankruptcy": -3.0,
    "bankrupt": -3.0,
    "scam": -3.0,
    "ponzi": -3.5,
    "fraud": -3.0,
    "lawsuit": -2.0,
    "sec charges": -2.5,
    "investigation": -1.5,
    "delisting": -2.5,
    "delisted": -2.5,
    
    # === محايد أو مختلط ===
    "fomo": 1.5,
    "regulation": -0.5,
    "regulatory": -0.5,
    "volatile": -0.3,
    "uncertainty": -1.0,
    "uncertain": -1.0,
}


# ============================================================
# المحلل
# ============================================================
class SentimentAnalyzer:
    """
    محلل مشاعر الأخبار
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # إضافة قاموس الكريبتو
        self.vader.lexicon.update(CRYPTO_LEXICON)
        log.info(f"[SENTIMENT] تم تهيئة VADER + {len(CRYPTO_LEXICON)} مصطلح كريبتو")
    
    # ========================================================
    # تحليل نص واحد
    # ========================================================
    def analyze_text(self, text: str) -> Dict:
        """
        تحليل نص واحد
        
        Parameters
        ----------
        text : str
            العنوان أو النص
        
        Returns
        -------
        dict
            {
                "compound": -1.0 to +1.0,
                "pos": 0.0 to 1.0,
                "neu": 0.0 to 1.0,
                "neg": 0.0 to 1.0,
                "label": "positive" / "negative" / "neutral"
            }
        """
        if not text or not isinstance(text, str):
            return {
                "compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0,
                "label": "neutral"
            }
        
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "compound": round(compound, 4),
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4),
            "label": label,
        }
    
    # ========================================================
    # تحليل قائمة أخبار
    # ========================================================
    def analyze_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        """
        تحليل قائمة أخبار وإضافة حقل sentiment لكل خبر
        
        Parameters
        ----------
        news_list : list[dict]
            قائمة أخبار من news_fetcher
            [{"title": "...", "votes": {...}, ...}, ...]
        
        Returns
        -------
        list[dict]
            نفس القائمة + حقل sentiment لكل خبر
        """
        analyzed = []
        for article in news_list:
            title = article.get("title", "")
            sentiment = self.analyze_text(title)
            
            # وزن إضافي حسب تصويتات CryptoPanic
            votes = article.get("votes", {})
            positive_votes = votes.get("positive", 0)
            negative_votes = votes.get("negative", 0)
            
            if positive_votes + negative_votes > 0:
                vote_sentiment = (
                    (positive_votes - negative_votes)
                    / (positive_votes + negative_votes)
                )
                # دمج: 70% من VADER + 30% من التصويتات
                sentiment["compound"] = round(
                    0.7 * sentiment["compound"] + 0.3 * vote_sentiment, 4
                )
                # إعادة تصنيف
                if sentiment["compound"] >= 0.05:
                    sentiment["label"] = "positive"
                elif sentiment["compound"] <= -0.05:
                    sentiment["label"] = "negative"
                else:
                    sentiment["label"] = "neutral"
            
            article["sentiment"] = sentiment
            analyzed.append(article)
        
        return analyzed
    
    # ========================================================
    # تجميع مشاعر عملة معينة
    # ========================================================
    def aggregate_sentiment(
        self, analyzed_news: List[Dict], symbol: str = None
    ) -> Dict:
        """
        تجميع مشاعر عملة معينة
        
        Parameters
        ----------
        analyzed_news : list[dict]
            قائمة أخبار محللة
        symbol : str, optional
            رمز العملة (BTCUSDT, ETHUSDT, ...)
        
        Returns
        -------
        dict
            {
                "score": -1.0 to +1.0,
                "label": "bullish" / "bearish" / "neutral",
                "count": int,
                "positive": int,
                "negative": int,
                "neutral": int
            }
        """
        if not analyzed_news:
            return {
                "score": 0.0, "label": "neutral",
                "count": 0, "positive": 0, "negative": 0, "neutral": 0
            }
        
        # فلترة حسب العملة
        if symbol:
            base = symbol.replace("USDT", "").replace("USD", "").upper()
            filtered = [
                n for n in analyzed_news
                if base in [c.upper() for c in n.get("currencies", [])]
            ]
        else:
            filtered = analyzed_news
        
        if not filtered:
            return {
                "score": 0.0, "label": "neutral",
                "count": 0, "positive": 0, "negative": 0, "neutral": 0
            }
        
        # حساب المتوسط
        compounds = [n["sentiment"]["compound"] for n in filtered]
        avg_score = sum(compounds) / len(compounds)
        
        positive = sum(1 for c in compounds if c >= 0.05)
        negative = sum(1 for c in compounds if c <= -0.05)
        neutral = len(compounds) - positive - negative
        
        # التصنيف
        if avg_score >= 0.15:
            label = "bullish"
        elif avg_score <= -0.15:
            label = "bearish"
        else:
            label = "neutral"
        
        return {
            "score": round(avg_score, 4),
            "label": label,
            "count": len(filtered),
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        }


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    analyzer = SentimentAnalyzer()
    
    test_headlines = [
        # إيجابي
        "Bitcoin surges to new all-time high as institutional adoption grows",
        "SEC approves spot Bitcoin ETF in landmark decision",
        "Ethereum upgrade successful, gas fees drop significantly",
        "MicroStrategy buys another $500M in Bitcoin",
        
        # سلبي
        "Major exchange hacked, millions lost in exploit",
        "Crypto market crashes amid regulatory crackdown fears",
        "Bitcoin plunges below $80K as whales dump holdings",
        "SEC files lawsuit against major crypto exchange",
        
        # محايد
        "Bitcoin price unchanged as traders await Fed decision",
        "Crypto market sees mixed signals this week",
    ]
    
    print("\n" + "=" * 70)
    print("اختبار تحليل المشاعر")
    print("=" * 70)
    
    for h in test_headlines:
        result = analyzer.analyze_text(h)
        emoji = "🟢" if result["label"] == "positive" else "🔴" if result["label"] == "negative" else "⚪"
        print(f"{emoji} [{result['compound']:+.3f}] {h}")
    
    print("\n" + "=" * 70)
    print("اختبار التجميع")
    print("=" * 70)
    
    # محاكاة أخبار محللة
    mock_news = [
        {"title": test_headlines[0], "currencies": ["BTC"], "votes": {"positive": 10, "negative": 0}},
        {"title": test_headlines[1], "currencies": ["BTC"], "votes": {"positive": 15, "negative": 1}},
        {"title": test_headlines[2], "currencies": ["ETH"], "votes": {"positive": 8, "negative": 2}},
        {"title": test_headlines[3], "currencies": ["BTC"], "votes": {"positive": 12, "negative": 3}},
    ]
    
    analyzed = analyzer.analyze_news_batch(mock_news)
    
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        agg = analyzer.aggregate_sentiment(analyzed, symbol)
        emoji = "🟢" if agg["label"] == "bullish" else "🔴" if agg["label"] == "bearish" else "⚪"
        print(f"\n{emoji} {symbol}:")
        print(f"   Score: {agg['score']:+.4f}")
        print(f"   Label: {agg['label']}")
        print(f"   Count: {agg['count']} أخبار")
        print(f"   (إيجابي: {agg['positive']}, سلبي: {agg['negative']}, محايد: {agg['neutral']})")
