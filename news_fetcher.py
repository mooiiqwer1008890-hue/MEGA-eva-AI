"""
news_fetcher.py
===============
جلب الأخبار من CoinMarketCap Keyless API.

المصدر:
- https://api.coinmarketcap.com/content/v3/news
- بدون API Key
- نفس API الذي يستخدمه موقع CoinMarketCap

المزايا:
- مجاني 100%
- لا يحتاج تسجيل
- أخبار مرتبة حسب العملة
- Cache محلي لتقليل الطلبات
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

log = logging.getLogger("news_fetcher")


# ============================================================
# CONFIG
# ============================================================
NEWS_CACHE_FILE = "news_cache.json"
NEWS_MAX_AGE_HOURS = 1  # الأخبار صالحة لمدة ساعة

# CoinMarketCap API (Keyless — بدون API Key)
CMC_BASE_URL = "https://api.coinmarketcap.com"

# خريطة العملات → CMC coinId
# (مطلوبة لأن API يستخدم IDs وليس رموز)
COIN_ID_MAP = {
    "BTC": 1,
    "ETH": 1027,
    "BNB": 1839,
    "SOL": 5426,
    "XRP": 52,
    "ADA": 2010,
    "DOGE": 74,
    "DOT": 6636,
    "LINK": 1975,
    "MATIC": 3890,
    "AVAX": 5805,
    "ATOM": 3794,
    "LTC": 2,
    "BCH": 1831,
    "TRX": 1958,
    "UNI": 7083,
    "AAVE": 7278,
    "MKR": 1518,
    "COMP": 5692,
    "SUSHI": 6758,
}

# أسماء العملات (للطباعة)
COIN_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "Binance Coin",
    "SOL": "Solana",
    "XRP": "XRP",
}


# ============================================================
# جلب الأخبار من CoinMarketCap
# ============================================================
def fetch_cmc_news(
    symbols: List[str] = None,
    limit: int = 50,
    timeout: int = 10
) -> List[Dict]:
    """
    جلب الأخبار من CoinMarketCap Keyless API.
    
    Parameters
    ----------
    symbols : list[str]
        قائمة رموز العملات (مثل ["BTC", "ETH"])
        None = الأخبار العامة
    limit : int
        عدد الأخبار
    timeout : int
        timeout بالثواني
    
    Returns
    -------
    list[dict]
        قائمة أخبار بالشكل الموحد:
        [{
            "title": "...",
            "subtitle": "...",
            "source": "...",
            "url": "...",
            "published_at": "...",
            "currencies": ["BTC", "ETH"],
            "votes": {}
        }]
    """
    url = f"{CMC_BASE_URL}/content/v3/news"
    
    params = {
        "page": 1,
        "size": limit,
    }
    
    # إذا حددنا عملات، نمرر IDsها
    if symbols:
        coin_ids = []
        for sym in symbols:
            base = sym.replace("USDT", "").replace("USD", "").upper()
            if base in COIN_ID_MAP:
                coin_ids.append(str(COIN_ID_MAP[base]))
        
        if coin_ids:
            params["coins"] = ",".join(coin_ids)
            log.info(f"[CMC] جلب أخبار لـ: {symbols} (IDs: {params['coins']})")
    else:
        log.info("[CMC] جلب الأخبار العامة")
    
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        
        articles = []
        for item in data.get("data", []):
            meta = item.get("meta", {})
            
            # استخلاص العملات
            currencies = []
            for asset in item.get("assets", []):
                name = asset.get("name", "").upper()
                # نحاول ربط الاسم بالرمز
                for sym, cname in COIN_NAMES.items():
                    if cname.upper() in name or sym in name:
                        if sym not in currencies:
                            currencies.append(sym)
            
            articles.append({
                "title": meta.get("title", ""),
                "subtitle": meta.get("subtitle", ""),
                "source": meta.get("sourceName", "CMC"),
                "url": meta.get("sourceUrl", ""),
                "published_at": meta.get("releasedAt", ""),
                "currencies": currencies,
                "votes": {},  # CMC لا يوفر تصويتات
            })
        
        log.info(f"[CMC] ✅ تم جلب {len(articles)} خبر")
        return articles
    
    except requests.exceptions.RequestException as e:
        log.error(f"[CMC] ❌ فشل جلب الأخبار: {e}")
        return []
    except json.JSONDecodeError as e:
        log.error(f"[CMC] ❌ فشل تحليل JSON: {e}")
        return []


# ============================================================
# إدارة الكاش
# ============================================================
def load_news_cache() -> List[Dict]:
    """يقرأ ملف الكاش."""
    if not os.path.exists(NEWS_CACHE_FILE):
        return []
    
    try:
        with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            log.info(f"[CACHE] تم تحميل {len(data)} خبر من الكاش")
            return data
    except Exception as e:
        log.warning(f"[CACHE] فشل قراءة الكاش: {e}")
        return []


def save_news_cache(news_list: List[Dict]) -> List[Dict]:
    """يحفظ الأخبار في الملف مع إزالة القديمة."""
    cutoff = datetime.utcnow() - timedelta(hours=NEWS_MAX_AGE_HOURS)
    
    filtered = []
    for item in news_list:
        try:
            pub_str = item.get("published_at", "")
            if pub_str:
                pub_time = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                # إزالة timezone للمقارنة
                pub_time = pub_time.replace(tzinfo=None)
                if pub_time >= cutoff:
                    filtered.append(item)
            else:
                filtered.append(item)
        except Exception:
            filtered.append(item)
    
    try:
        with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        log.info(f"[CACHE] تم حفظ {len(filtered)} خبر")
    except Exception as e:
        log.error(f"[CACHE] فشل حفظ الكاش: {e}")
    
    return filtered


def update_news_cache(symbols: List[str] = None) -> List[Dict]:
    """
    يحدّث الكاش بالأخبار الجديدة.
    
    Parameters
    ----------
    symbols : list[str], optional
        عملات محددة (None = عام)
    
    Returns
    -------
    list[dict]
        الأخبار المحدثة
    """
    old_news = load_news_cache()
    
    # جلب أخبار جديدة
    new_news = fetch_cmc_news(symbols=symbols, limit=50)
    
    # دمج بدون تكرار (حسب URL)
    existing_urls = {item.get("url") for item in old_news}
    for item in new_news:
        if item.get("url") not in existing_urls:
            old_news.append(item)
    
    # حفظ (مع إزالة القديمة)
    return save_news_cache(old_news)


def get_cached_news(currencies: List[str] = None) -> List[Dict]:
    """
    يقرأ الأخبار من الكاش، مع فلترة حسب العملة.
    
    Parameters
    ----------
    currencies : list[str], optional
        عملات محددة (مثل ["BTC", "ETH"])
    
    Returns
    -------
    list[dict]
        الأخبار المفلترة
    """
    news = load_news_cache()
    
    if not currencies:
        return news
    
    # فلترة حسب العملة
    filtered = []
    currencies_upper = [c.upper() for c in currencies]
    
    for item in news:
        item_currencies = [c.upper() for c in item.get("currencies", [])]
        if any(c in currencies_upper for c in item_currencies):
            filtered.append(item)
    
    return filtered


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("\n" + "=" * 70)
    print("اختبار news_fetcher.py (CoinMarketCap Keyless API)")
    print("=" * 70)
    
    # 1. اختبار الأخبار العامة
    print("\n[1] جلب الأخبار العامة...")
    general_news = fetch_cmc_news(limit=5)
    for n in general_news[:5]:
        print(f"  📰 {n['title'][:80]}...")
        print(f"     المصدر: {n['source']} | العملات: {n['currencies']}")
    
    # 2. اختبار فلترة BTC
    print("\n[2] جلب أخبار BTC...")
    btc_news = fetch_cmc_news(symbols=["BTC"], limit=5)
    for n in btc_news[:5]:
        print(f"  🟠 {n['title'][:80]}...")
    
    # 3. اختبار الكاش
    print("\n[3] اختبار الكاش...")
    if btc_news:
        cached = save_news_cache(btc_news)
        print(f"  ✅ تم حفظ {len(cached)} خبر في الكاش")
        
        loaded = get_cached_news(["BTC"])
        print(f"  ✅ تم تحميل {len(loaded)} خبر من الكاش")
    
    print("\n" + "=" * 70)
