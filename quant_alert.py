        return True
    except requests.RequestException as e:
        log.error(f"Telegram send failed: {e}")
        return False


def build_report(symbol: str, interval: str, price: float, z: float, vol: float) -> str:
    signal = classify_signal(z)
    return (
        f"🧠 *تحليل إحصائي - {symbol} ({interval})*\n"
        f"-------------------------------------\n"
        f"💵 *السعر:* ${price:,.2f}\n\n"
        f"📐 *المؤشرات:*\n"
        f"• Z-Score: {z:.2f}\n"
        f"• تقلب Garman-Klass: {vol:.3f}%\n\n"
        f"-------------------------------------\n"
        f"🎯 *الإشارة:* {signal}\n"
        f"-------------------------------------\n"
        f"⚠️ _تذكير: هذه إشارة إحصائية أولية غير مُختبرة تاريخياً (Backtested).  "
        f"لا تُستخدم كقرار تنفيذ مباشر._"
    )


# ---------------------------------------------------------------------
# 4. MAIN LOOP — runs forever, survives individual failures
# ---------------------------------------------------------------------
def run_cycle():
    candles = fetch_candles(SYMBOL, INTERVAL, CANDLE_LIMIT)
    closes = candles[:, 3]
    price = closes[-1]
    z = z_score(closes, LOOKBACK)
    vol = garman_klass_volatility(candles, LOOKBACK)
    report = build_report(SYMBOL, INTERVAL, price, z, vol)
    sent = send_telegram_message(report)
    log.info(
        f"{SYMBOL} price={price:.2f} z={z:.2f} gk_vol={vol:.3f}% "
        f"telegram_sent={sent}"
    )


def main():
    log.info(f"Starting quant_alert for {SYMBOL} ({INTERVAL}), polling every {POLL_SECONDS}s")
    while True:
        try:
            run_cycle()
        except requests.RequestException as e:
            log.error(f"Network error this cycle, will retry next cycle: {e}")
        except Exception as e:
            # Catch-all so a transient bug never kills the whole process —
            # but we still log it loudly so you notice and fix it.
            log.exception(f"Unexpected error this cycle: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
