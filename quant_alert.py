"""
quant_alert.py
===============
Python replacement for the C/Termux BTC analysis bot.

What changed vs. the original C version, and why:

1. SECRETS: Bot token / chat ID are read from environment variables,
   never hardcoded. Set them in Railway's "Variables" tab, not in code.
   NEVER commit a token to git, even in a private repo.

2. NO SHELL INJECTION: The original C code built a curl command by
   string-concatenating the message text, so a literal "$" in the
   message (e.g. "$67234.50") could trigger shell variable expansion
   and silently corrupt or truncate the message. Here, `requests`
   sends the message as a proper HTTP POST body — no shell involved,
   so there is nothing to inject into.

3. NO system()/curl/jq: `requests` talks to Binance and Telegram
   directly over HTTPS. Fewer moving parts, proper error handling,
   no dependency on external CLI tools being installed correctly.

4. RUNS FOREVER, SAFELY: A loop with a sleep interval, wrapped so a
   single failed request (network blip) doesn't crash the whole
   process — it logs the error and tries again next cycle. This is
   the actual requirement for "works all the time", and it has
   nothing to do with C vs Python — it's about where it's hosted
   (Railway, always-on) vs where it isn't (a phone Android kills).

Indicators (same formulas as your C version):
- Z-Score of the last close vs. a rolling mean/std (mean-reversion signal)
- Garman-Klass volatility (uses OHLC, more efficient than close-only vol)

IMPORTANT CAVEAT (carried over from our earlier discussion):
The signal thresholds below (|z| > 2) are the same ones from your C
code — they are a REASONABLE STARTING POINT, not a validated
strategy. Before trusting this for real capital, backtest these
thresholds properly (walk-forward, out-of-sample) using the
feature-engineering pipeline we already built. This script is the
DATA/ALERT layer, not a substitute for that validation step.
"""

import os
import time
import logging
import requests
import numpy as np

# ---------------------------------------------------------------------
# CONFIG — all secrets come from environment variables, set these in
# Railway's dashboard under Variables, never in this file.
# ---------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_CHAT_ID")
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
