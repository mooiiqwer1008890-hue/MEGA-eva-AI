"""
news_fetcher.py
===============
جلب الأخبار من CryptoPanic API وتحليلها.

المصدر: كتاب "Advanced Algorithmic Trading" - الفصل 23

المتطلبات:
- CryptoPanic API Key (مجاني): https://cryptopanic.com/developers/api/
- مكتبة vaderSentiment: pip install vaderSentiment
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta

log = logging.getLogger("news_fetcher")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")
NEWS_CACHE_FILE = "news_cache.json"
NEWS_MAX_AGE_HOURS = 24  # الأخبار الأقدم من 24 ساعة تُحذف


# ---------------------------------------------------------------------
# FETCH NEWS
# ---------------------------------------------------------------------
def fetch_cryptopanic_news(currencies="BTC,ETH", limit=50):
    """
    يجلب الأخبار من CryptoPanic API.
    
    Parameters
    ----------
    currencies : str
        قائمة العملات (مفصولة بفواصل)
    limit : int
        عدد الأخبار
    
    Returns
    -------
    list
        قائمة الأخبار (كل خبر = dict)
    """
    if not CRYPTOPANIC_API_KEY:
        log.warning("CRYPTOPANIC_API_KEY غير مضبوط. لا يمكن جلب الأخبار.")
        return []

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "currencies": currencies,
        "filter": "hot",
        "public": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("results", [])

        news = []
        for post in posts[:limit]:
            news.append({
                "title": post.get("title", ""),
                "url": post.get("url", ""),
                "source": post.get("source", {}).get("title", "Unknown"),
                "published_at": post.get("published_at", ""),
                "currencies": [c.get("code") for c in post.get("currencies", [])],
                "votes": post.get("votes", {}),
            })

        log.info(f"Fetched {len(news)} news items from CryptoPanic.")
        return news
    except requests.RequestException as e:
        log.error(f"Failed to fetch CryptoPanic news: {e}")
        return []


# ---------------------------------------------------------------------
# CACHE MANAGEMENT
# ---------------------------------------------------------------------
def load_news_cache():
    """يقرأ ملف الكاش."""
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load news cache: {e}")
    return []


def save_news_cache(news_list):
    """يحفظ الأخبار في الملف (مع إزالة القديمة)."""
    cutoff = datetime.utcnow() - timedelta(hours=NEWS_MAX_AGE_HOURS)
    filtered = []
    for item in news_list:
        try:
            pub_time = datetime.fromisoformat(item.get("published_at", "").replace("Z", "+00:00"))
            if pub_time.replace(tzinfo=None) >= cutoff:
                filtered.append(item)
        except Exception:
            filtered.append(item)  # احتفظ بالخبر إذا فشل تحويل التاريخ

    with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    log.info(f"News cache saved: {len(filtered)} items.")
    return filtered


def update_news_cache():
    """
    يحدث الكاش: يجلب أخبار جديدة، يدمجها مع القديمة، يحفظ.
    """
    old_news = load_news_cache()
    new_news = fetch_cryptopanic_news()

    # دمج (بدون تكرار حسب URL)
    existing_urls = {item.get("url") for item in old_news}
    for item in new_news:
        if item.get("url") not in existing_urls:
            old_news.append(item)

    return save_news_cache(old_news)


def get_cached_news(currencies=None):
    """
    يقرأ الأخبار من الكاش.
    
    Parameters
    ----------
    currencies : list
        إذا محدد، نرجع الأخبار التي تخص هذه العملات فقط.
    
    Returns
    -------
    list
    """
    news = load_news_cache()
    if currencies:
        currencies_upper = [c.upper() for c in currencies]
        news = [
            item for item in news
            if any(c.upper() in currencies_upper for c in item.get("currencies", []))
        ]
    return news


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 70)
    print("اختبار News Fetcher")
    print("=" * 70 + "\n")

    if not CRYPTOPANIC_API_KEY:
        print("⚠️ لا يوجد API Key. الحصول على مفتاح مجاني من:")
        print("https://cryptopanic.com/developers/api/")
    else:
        news = update_news_cache()
        print(f"✅ تم جلب {len(news)} خبر وحفظها في الكاش.\n")

        btc_news = get_cached_news(["BTC"])
        print(f"📰 أخبار BTC: {len(btc_news)} خبر")
        for item in btc_news[:5]:
            print(f"   ├─ {item['title'][:80]}")
            print(f"   └─ المصدر: {item['source']} | {item['published_at']}")
