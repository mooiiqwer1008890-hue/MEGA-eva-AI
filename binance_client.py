"""
binance_client.py
=================

Professional Binance Spot Execution Engine.

المعمارية:
-----------

Signal Engine
     ↓
Risk Manager
     ↓
Execution Engine  ← هذا الملف
     ↓
Binance Spot API

هذا الملف لا يقرر:
    "هل يجب أن أشتري؟"

بل يقرر:
    "هل يمكن تنفيذ الأمر المطلوب بأمان وبصورة صحيحة
     وفق قواعد Binance والسيولة الحالية؟"

المزايا:
--------
- Spot فقط
- Testnet افتراضيًا
- عدم وضع API keys داخل الكود
- Decimal بدل float في الأموال والأسعار والكميات
- قراءة exchangeInfo تلقائيًا
- احترام:
    PRICE_FILTER
    LOT_SIZE
    MARKET_LOT_SIZE
    MIN_NOTIONAL
    NOTIONAL
    MAX_NUM_ORDERS
    MAX_NUM_ALGO_ORDERS
    MAX_POSITION
    TRAILING_DELTA
- فحص حالة الرمز
- فحص أنواع الأوامر المسموحة
- فحص الرصيد قبل التنفيذ
- فحص Bid/Ask spread
- تقدير الانزلاق السعري من Order Book
- منع تنفيذ Market Order عندما تكون السيولة ضعيفة
- حماية من تكرار الأمر باستخدام clientOrderId
- عدم إعادة إرسال الأمر عشوائيًا بعد timeout
- استرجاع حالة الأمر عند حدوث مشكلة نقل
- Stop Loss Limit متوافق مع tick size
- دعم Test Order بدون تنفيذ فعلي
- مراقبة حالة الأمر
- إلغاء الأوامر
- سجلات آمنة بدون API Secret

مهم:
-----
هذا الملف Execution Layer.
لا يضع استراتيجية أو Alpha Model داخله.

يجب أن يأتي القرار من ملفات مثل:
    zscore_engine.py
    regime_detector.py
    risk_manager.py
    meta_labeling.py

ثم يرسل القرار إلى هذا الملف.

المراجع:
---------
- Binance Spot API Documentation
- Binance Spot Filters
- Binance Spot Testnet Documentation
- python-binance documentation
- مبادئ Market Microstructure
- مبادئ Execution Risk / Slippage Control

"""

from __future__ import annotations

import logging
import os
import time
import uuid

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

from binance.client import Client
from binance.exceptions import (
    BinanceAPIException,
    BinanceOrderException,
    BinanceRequestException,
)


# ============================================================================
# Logging
# ============================================================================

log = logging.getLogger("binance_client")


# ============================================================================
# Constants
# ============================================================================

ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_LIMIT = "LIMIT"
ORDER_TYPE_STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"

SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

TIME_IN_FORCE_GTC = "GTC"

ORDER_STATUS_NEW = "NEW"
ORDER_STATUS_PARTIALLY_FILLED = "PARTIALLY_FILLED"
ORDER_STATUS_FILLED = "FILLED"
ORDER_STATUS_CANCELED = "CANCELED"
ORDER_STATUS_REJECTED = "REJECTED"
ORDER_STATUS_EXPIRED = "EXPIRED"

SUPPORTED_TESTNET_ONLY_ENV = "BINANCE_TESTNET"


# ============================================================================
# Exceptions
# ============================================================================

class BinanceClientError(Exception):
    """Base execution-engine exception."""


class BinanceConfigurationError(BinanceClientError):
    """Invalid configuration or missing credentials."""


class BinanceValidationError(BinanceClientError):
    """Local validation failed before sending the order."""


class BinanceInsufficientBalance(BinanceClientError):
    """Not enough free balance."""


class BinanceMarketSafetyError(BinanceClientError):
    """Market conditions fail execution safety checks."""


class BinanceOrderRejected(BinanceClientError):
    """Binance rejected the order."""


# ============================================================================
# Decimal helpers
# ============================================================================

def D(value: Any) -> Decimal:
    """
    Convert safely to Decimal.

    Never use:
        Decimal(float)

    because binary floating-point representation can introduce
    undesirable rounding errors.

    Instead:
        Decimal(str(value))
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise BinanceValidationError(
            f"Invalid numeric value: {value!r}"
        ) from exc


def decimal_to_str(value: Decimal) -> str:
    """
    Convert Decimal to a non-scientific string accepted by Binance.
    """
    if not value.is_finite():
        raise BinanceValidationError(
            f"Non-finite Decimal value: {value}"
        )

    return format(value, "f")


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """
    Round DOWN to the nearest step.

    Used for:
        quantity
        price

    We intentionally use ROUND_DOWN because rounding upward could make
    an order larger than the intended risk/notional.
    """
    if step <= 0:
        return value

    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return units * step


def floor_to_filter(
    value: Decimal,
    minimum: Decimal,
    step: Decimal,
) -> Decimal:
    """
    Align value to:
        minimum + N * step

    This handles filters where minPrice/minQty is non-zero.
    """
    if step <= 0:
        return value

    if value < minimum:
        return minimum

    steps = ((value - minimum) / step).to_integral_value(
        rounding=ROUND_DOWN
    )

    return minimum + (steps * step)


# ============================================================================
# Data classes
# ============================================================================

@dataclass(frozen=True)
class MarketSnapshot:
    """
    Snapshot of top-of-book conditions.
    """

    symbol: str
    bid: Decimal
    ask: Decimal
    mid: Decimal
    spread: Decimal
    spread_bps: Decimal
    last_price: Decimal


@dataclass(frozen=True)
class ExecutionEstimate:
    """
    Estimated execution quality from order-book depth.
    """

    side: str
    quantity: Decimal
    estimated_vwap: Decimal
    best_price: Decimal
    slippage_bps: Decimal
    quote_value: Decimal


@dataclass
class SymbolRules:
    """
    Parsed Binance symbol rules.
    """

    symbol: str
    status: str
    base_asset: str
    quote_asset: str

    min_price: Decimal = D("0")
    max_price: Decimal = D("0")
    tick_size: Decimal = D("0")

    min_qty: Decimal = D("0")
    max_qty: Decimal = D("0")
    step_size: Decimal = D("0")

    market_min_qty: Decimal = D("0")
    market_max_qty: Decimal = D("0")
    market_step_size: Decimal = D("0")

    min_notional: Decimal = D("0")
    max_notional: Decimal = D("0")

    min_notional_apply_to_market: bool = False
    max_notional_apply_to_market: bool = False

    max_num_orders: int = 0
    max_num_algo_orders: int = 0
    max_position: Decimal = D("0")

    min_trailing_below_delta: int = 0
    max_trailing_below_delta: int = 0
    min_trailing_above_delta: int = 0
    max_trailing_above_delta: int = 0

    order_types: Tuple[str, ...] = ()


# ============================================================================
# Main Client
# ============================================================================

class BinanceClient:
    """
    Professional Binance Spot execution client.

    Parameters
    ----------
    testnet:
        True  -> Binance Spot Testnet
        False -> Binance Live

    dry_run:
        When True, validation and safety checks run but no real order
        is submitted.
    """

    def __init__(
        self,
        testnet: bool = True,
        dry_run: bool = False,
        cache_ttl_seconds: int = 300,
    ) -> None:

        self.testnet = bool(testnet)
        self.dry_run = bool(dry_run)
        self.cache_ttl_seconds = max(30, int(cache_ttl_seconds))

        # ------------------------------------------------------------------
        # Safety configuration
        # ------------------------------------------------------------------

        self.max_spread_bps = D(
            os.getenv("BINANCE_MAX_SPREAD_BPS", "30")
        )

        self.max_slippage_bps = D(
            os.getenv("BINANCE_MAX_SLIPPAGE_BPS", "100")
        )

        self.balance_reserve_pct = D(
            os.getenv("BINANCE_BALANCE_RESERVE_PCT", "0.20")
        )

        self.stop_limit_offset_bps = D(
            os.getenv("BINANCE_STOP_LIMIT_OFFSET_BPS", "25")
        )

        self.recv_window = int(
            os.getenv("BINANCE_RECV_WINDOW", "5000")
        )

        self.request_timeout = int(
            os.getenv("BINANCE_REQUEST_TIMEOUT", "15")
        )

        # Maximum quote notional for one order.
        #
        # Example:
        #   25
        #
        # means the local engine will refuse an order above 25 USDT,
        # even if Binance itself would allow it.
        max_order_notional_raw = os.getenv(
            "BINANCE_MAX_ORDER_NOTIONAL",
            "0",
        )

        self.max_order_notional = D(max_order_notional_raw)

        # ------------------------------------------------------------------
        # LIVE trading requires explicit opt-in.
        # ------------------------------------------------------------------

        if not self.testnet:

            live_confirmation = os.getenv(
                "BINANCE_LIVE_TRADING",
                "",
            ).strip().upper()

            if live_confirmation != "YES":

                raise BinanceConfigurationError(
                    "Live trading is disabled by safety policy. "
                    "Set BINANCE_LIVE_TRADING=YES explicitly."
                )

        # ------------------------------------------------------------------
        # Credentials
        # ------------------------------------------------------------------

        if self.testnet:

            api_key = os.getenv(
                "BINANCE_TESTNET_API_KEY"
            )

            secret_key = os.getenv(
                "BINANCE_TESTNET_SECRET_KEY"
            )

        else:

            api_key = os.getenv(
                "BINANCE_API_KEY"
            )

            secret_key = os.getenv(
                "BINANCE_SECRET_KEY"
            )

        if not api_key or not secret_key:

            environment = (
                "TESTNET"
                if self.testnet
                else "LIVE"
            )

            raise BinanceConfigurationError(
                f"Missing Binance {environment} API credentials."
            )

        # ------------------------------------------------------------------
        # Create Binance client.
        #
        # Important:
        # We intentionally do NOT manually overwrite API_URL.
        # python-binance handles the testnet endpoint via testnet=True.
        # ------------------------------------------------------------------

        self.client = Client(
            api_key=api_key,
            api_secret=secret_key,
            testnet=self.testnet,
            requests_params={
                "timeout": self.request_timeout,
            },
            ping=True,
        )

        # ------------------------------------------------------------------
        # Internal caches
        # ------------------------------------------------------------------

        self._symbol_cache: Dict[
            str,
            Tuple[float, Dict[str, Any]]
        ] = {}

        self._rules_cache: Dict[
            str,
            Tuple[float, SymbolRules]
        ] = {}

        # ------------------------------------------------------------------
        # Connectivity
        # ------------------------------------------------------------------

        self._verify_connection()

    # ======================================================================
    # Connection
    # ======================================================================

    def _verify_connection(self) -> None:
        """
        Verify connectivity and authenticated account access.
        """

        try:

            self.client.ping()

            server_time = self.client.get_server_time()

            account = self.client.get_account(
                recvWindow=self.recv_window
            )

            balances = {
                item["asset"]: D(item["free"])
                for item in account.get("balances", [])
            }

            usdt_balance = balances.get(
                "USDT",
                D("0"),
            )

            environment = (
                "TESTNET"
                if self.testnet
                else "LIVE"
            )

            log.info(
                "[BINANCE] Connected | "
                "Environment=%s | "
                "USDT=%s | "
                "ServerTime=%s",
                environment,
                decimal_to_str(usdt_balance),
                server_time.get("serverTime"),
            )

        except Exception as exc:

            log.exception(
                "[BINANCE] Connection verification failed"
            )

            raise BinanceClientError(
                "Could not verify Binance connection."
            ) from exc

    # ======================================================================
    # Symbol information
    # ======================================================================

    def _get_symbol_info(
        self,
        symbol: str,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        now = time.monotonic()

        cached = self._symbol_cache.get(symbol)

        if cached:

            cached_at, data = cached

            if (
                now - cached_at
                < self.cache_ttl_seconds
            ):
                return data

        try:

            info = self.client.get_exchange_info()

            for item in info.get("symbols", []):

                if item.get("symbol") == symbol:

                    self._symbol_cache[symbol] = (
                        now,
                        item,
                    )

                    return item

        except (
            BinanceAPIException,
            BinanceRequestException,
        ) as exc:

            raise BinanceClientError(
                f"Failed to load exchange information "
                f"for {symbol}"
            ) from exc

        raise BinanceValidationError(
            f"Unknown Binance symbol: {symbol}"
        )

    def _get_symbol_rules(
        self,
        symbol: str,
    ) -> SymbolRules:

        symbol = symbol.upper()

        now = time.monotonic()

        cached = self._rules_cache.get(symbol)

        if cached:

            cached_at, rules = cached

            if (
                now - cached_at
                < self.cache_ttl_seconds
            ):
                return rules

        info = self._get_symbol_info(symbol)

        filters = {
            item["filterType"]: item
            for item in info.get("filters", [])
        }

        price_filter = filters.get(
            "PRICE_FILTER",
            {},
        )

        lot_filter = filters.get(
            "LOT_SIZE",
            {},
        )

        market_lot_filter = filters.get(
            "MARKET_LOT_SIZE",
            {},
        )

        min_notional_filter = filters.get(
            "MIN_NOTIONAL",
            {},
        )

        notional_filter = filters.get(
            "NOTIONAL",
            {},
        )

        max_orders_filter = filters.get(
            "MAX_NUM_ORDERS",
            {},
        )

        max_algo_filter = filters.get(
            "MAX_NUM_ALGO_ORDERS",
            {},
        )

        max_position_filter = filters.get(
            "MAX_POSITION",
            {},
        )

        trailing_filter = filters.get(
            "TRAILING_DELTA",
            {},
        )

        min_notional = D("0")
        max_notional = D("0")

        apply_min_market = False
        apply_max_market = False

        # --------------------------------------------------------------
        # MIN_NOTIONAL
        # --------------------------------------------------------------

        if min_notional_filter:

            min_notional = D(
                min_notional_filter.get(
                    "minNotional",
                    "0",
                )
            )

            apply_min_market = bool(
                min_notional_filter.get(
                    "applyToMarket",
                    False,
                )
            )

        # --------------------------------------------------------------
        # NOTIONAL
        #
        # Prefer NOTIONAL when Binance exposes it.
        # --------------------------------------------------------------

        if notional_filter:

            min_notional = D(
                notional_filter.get(
                    "minNotional",
                    min_notional,
                )
            )

            max_notional = D(
                notional_filter.get(
                    "maxNotional",
                    "0",
                )
            )

            apply_min_market = bool(
                notional_filter.get(
                    "applyMinToMarket",
                    apply_min_market,
                )
            )

            apply_max_market = bool(
                notional_filter.get(
                    "applyMaxToMarket",
                    False,
                )
            )

        rules = SymbolRules(
            symbol=symbol,
            status=str(
                info.get("status", "")
            ),
            base_asset=str(
                info.get("baseAsset", "")
            ),
            quote_asset=str(
                info.get("quoteAsset", "")
            ),

            min_price=D(
                price_filter.get(
                    "minPrice",
                    "0",
                )
            ),

            max_price=D(
                price_filter.get(
                    "maxPrice",
                    "0",
                )
            ),

            tick_size=D(
                price_filter.get(
                    "tickSize",
                    "0",
                )
            ),

            min_qty=D(
                lot_filter.get(
                    "minQty",
                    "0",
                )
            ),

            max_qty=D(
                lot_filter.get(
                    "maxQty",
                    "0",
                )
            ),

            step_size=D(
                lot_filter.get(
                    "stepSize",
                    "0",
                )
            ),

            market_min_qty=D(
                market_lot_filter.get(
                    "minQty",
                    lot_filter.get(
                        "minQty",
                        "0",
                    ),
                )
            ),

            market_max_qty=D(
                market_lot_filter.get(
                    "maxQty",
                    lot_filter.get(
                        "maxQty",
                        "0",
                    ),
                )
            ),

            market_step_size=D(
                market_lot_filter.get(
                    "stepSize",
                    lot_filter.get(
                        "stepSize",
                        "0",
                    ),
                )
            ),

            min_notional=min_notional,
            max_notional=max_notional,

            min_notional_apply_to_market=apply_min_market,
            max_notional_apply_to_market=apply_max_market,

            max_num_orders=int(
                max_orders_filter.get(
                    "maxNumOrders",
                    0,
                )
            ),

            max_num_algo_orders=int(
                max_algo_filter.get(
                    "maxNumAlgoOrders",
                    0,
                )
            ),

            max_position=D(
                max_position_filter.get(
                    "maxPosition",
                    "0",
                )
            ),

            min_trailing_below_delta=int(
                trailing_filter.get(
                    "minTrailingBelowDelta",
                    0,
                )
            ),

            max_trailing_below_delta=int(
                trailing_filter.get(
                    "maxTrailingBelowDelta",
                    0,
                )
            ),

            min_trailing_above_delta=int(
                trailing_filter.get(
                    "minTrailingAboveDelta",
                    0,
                )
            ),

            max_trailing_above_delta=int(
                trailing_filter.get(
                    "maxTrailingAboveDelta",
                    0,
                )
            ),

            order_types=tuple(
                str(x)
                for x in info.get(
                    "orderTypes",
                    [],
                )
            ),
        )

        self._rules_cache[symbol] = (
            now,
            rules,
        )

        return rules

    # ======================================================================
    # Balance
    # ======================================================================

    def get_balance_decimal(
        self,
        asset: str = "USDT",
    ) -> Decimal:

        asset = asset.upper()

        try:

            balance = self.client.get_asset_balance(
                asset=asset,
                recvWindow=self.recv_window,
            )

            if not balance:

                return D("0")

            return D(
                balance.get(
                    "free",
                    "0",
                )
            )

        except Exception as exc:

            log.error(
                "[BINANCE] Balance read failed: asset=%s error=%s",
                asset,
                exc,
            )

            raise BinanceClientError(
                f"Failed to read {asset} balance."
            ) from exc

    def get_balance(
        self,
        asset: str = "USDT",
    ) -> float:
        """
        Backward-compatible float API.
        """

        return float(
            self.get_balance_decimal(asset)
        )

    # ======================================================================
    # Prices
    # ======================================================================

    def get_price_decimal(
        self,
        symbol: str,
    ) -> Decimal:

        symbol = symbol.upper()

        try:

            ticker = self.client.get_symbol_ticker(
                symbol=symbol
            )

            return D(
                ticker["price"]
            )

        except Exception as exc:

            log.error(
                "[BINANCE] Price read failed: symbol=%s error=%s",
                symbol,
                exc,
            )

            raise BinanceClientError(
                f"Failed to get price for {symbol}"
            ) from exc

    def get_price(
        self,
        symbol: str,
    ) -> float:
        """
        Backward-compatible API.
        """

        return float(
            self.get_price_decimal(symbol)
        )

    # ======================================================================
    # Order book / market microstructure
    # ======================================================================

    def get_market_snapshot(
        self,
        symbol: str,
        depth_limit: int = 20,
    ) -> MarketSnapshot:

        symbol = symbol.upper()

        try:

            book = self.client.get_order_book(
                symbol=symbol,
                limit=max(5, min(depth_limit, 100)),
            )

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            if not bids or not asks:

                raise BinanceMarketSafetyError(
                    f"Empty order book for {symbol}"
                )

            bid = D(bids[0][0])
            ask = D(asks[0][0])

            if bid <= 0 or ask <= 0:

                raise BinanceMarketSafetyError(
                    f"Invalid best bid/ask for {symbol}"
                )

            if ask < bid:

                raise BinanceMarketSafetyError(
                    f"Crossed order book detected for {symbol}"
                )

            mid = (
                bid + ask
            ) / D("2")

            spread = ask - bid

            spread_bps = (
                spread / mid
            ) * D("10000")

            last_price = (
                self.get_price_decimal(symbol)
            )

            return MarketSnapshot(
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                spread=spread,
                spread_bps=spread_bps,
                last_price=last_price,
            )

        except BinanceMarketSafetyError:
            raise

        except Exception as exc:

            raise BinanceClientError(
                f"Failed to read order book for {symbol}"
            ) from exc

    def _estimate_market_execution(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        depth_limit: int = 100,
    ) -> ExecutionEstimate:

        symbol = symbol.upper()
        side = side.upper()

        if side not in {
            SIDE_BUY,
            SIDE_SELL,
        }:
            raise BinanceValidationError(
                f"Unsupported side: {side}"
            )

        if quantity <= 0:

            raise BinanceValidationError(
                "Quantity must be > 0"
            )

        try:

            book = self.client.get_order_book(
                symbol=symbol,
                limit=max(20, min(depth_limit, 1000)),
            )

            levels = (
                book.get("asks", [])
                if side == SIDE_BUY
                else book.get("bids", [])
            )

            if not levels:

                raise BinanceMarketSafetyError(
                    f"No liquidity available for {symbol}"
                )

            remaining = quantity
            total_qty = D("0")
            total_quote = D("0")

            best_price = D(
                levels[0][0]
            )

            for level_price_raw, level_qty_raw in levels:

                level_price = D(level_price_raw)
                level_qty = D(level_qty_raw)

                if level_price <= 0 or level_qty <= 0:
                    continue

                fill_qty = min(
                    remaining,
                    level_qty,
                )

                total_qty += fill_qty
                total_quote += (
                    fill_qty * level_price
                )

                remaining -= fill_qty

                if remaining <= 0:
                    break

            if remaining > 0:

                raise BinanceMarketSafetyError(
                    f"Insufficient visible order-book liquidity "
                    f"for {symbol}; remaining={remaining}"
                )

            vwap = (
                total_quote / total_qty
            )

            if side == SIDE_BUY:

                slippage_bps = (
                    (vwap - best_price)
                    / best_price
                ) * D("10000")

            else:

                slippage_bps = (
                    (best_price - vwap)
                    / best_price
                ) * D("10000")

            return ExecutionEstimate(
                side=side,
                quantity=total_qty,
                estimated_vwap=vwap,
                best_price=best_price,
                slippage_bps=max(
                    D("0"),
                    slippage_bps,
                ),
                quote_value=total_quote,
            )

        except BinanceMarketSafetyError:
            raise

        except Exception as exc:

            raise BinanceClientError(
                f"Failed to estimate market execution for {symbol}"
            ) from exc

    def _check_market_conditions(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
    ) -> ExecutionEstimate:

        snapshot = self.get_market_snapshot(
            symbol
        )

        if snapshot.spread_bps > self.max_spread_bps:

            raise BinanceMarketSafetyError(
                f"Spread too wide for {symbol}: "
                f"{snapshot.spread_bps:.2f} bps > "
                f"{self.max_spread_bps:.2f} bps"
            )

        estimate = self._estimate_market_execution(
            symbol=symbol,
            side=side,
            quantity=quantity,
        )

        if estimate.slippage_bps > self.max_slippage_bps:

            raise BinanceMarketSafetyError(
                f"Estimated slippage too high for {symbol}: "
                f"{estimate.slippage_bps:.2f} bps > "
                f"{self.max_slippage_bps:.2f} bps"
            )

        return estimate

    # ======================================================================
    # Quantity / Price normalization
    # ======================================================================

    def normalize_quantity(
        self,
        symbol: str,
        quantity: Any,
        market: bool = False,
    ) -> Decimal:

        rules = self._get_symbol_rules(symbol)

        value = D(quantity)

        if value <= 0:

            raise BinanceValidationError(
                "Quantity must be greater than zero."
            )

        if market:

            minimum = rules.market_min_qty
            maximum = rules.market_max_qty
            step = rules.market_step_size

        else:

            minimum = rules.min_qty
            maximum = rules.max_qty
            step = rules.step_size

        normalized = floor_to_step(
            value,
            step,
        )

        if maximum > 0 and normalized > maximum:

            normalized = floor_to_step(
                maximum,
                step,
            )

        if normalized < minimum:

            raise BinanceValidationError(
                f"Quantity {normalized} is below "
                f"minimum {minimum} for {symbol}"
            )

        if normalized <= 0:

            raise BinanceValidationError(
                "Normalized quantity became zero."
            )

        return normalized

    def normalize_price(
        self,
        symbol: str,
        price: Any,
    ) -> Decimal:

        rules = self._get_symbol_rules(symbol)

        value = D(price)

        if value <= 0:

            raise BinanceValidationError(
                "Price must be greater than zero."
            )

        if (
            rules.min_price > 0
            and value < rules.min_price
        ):
            raise BinanceValidationError(
                f"Price {value} is below "
                f"minimum {rules.min_price}"
            )

        normalized = floor_to_filter(
            value,
            rules.min_price,
            rules.tick_size,
        )

        if (
            rules.max_price > 0
            and normalized > rules.max_price
        ):

            normalized = floor_to_filter(
                rules.max_price,
                rules.min_price,
                rules.tick_size,
            )

        return normalized

    # ======================================================================
    # Notional / account risk checks
    # ======================================================================

    def _check_notional(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        market: bool = False,
    ) -> Decimal:

        rules = self._get_symbol_rules(symbol)

        notional = quantity * price

        if self.max_order_notional > 0:

            if notional > self.max_order_notional:

                raise BinanceValidationError(
                    f"Local max order notional exceeded: "
                    f"{notional} > {self.max_order_notional}"
                )

        # --------------------------------------------------------------
        # Minimum notional
        # --------------------------------------------------------------

        if rules.min_notional > 0:

            must_apply = (
                rules.min_notional_apply_to_market
                if market
                else True
            )

            if (
                must_apply
                and notional < rules.min_notional
            ):

                raise BinanceValidationError(
                    f"Order notional {notional} is below "
                    f"minimum {rules.min_notional} for {symbol}"
                )

        # --------------------------------------------------------------
        # Maximum notional
        # --------------------------------------------------------------

        if rules.max_notional > 0:

            must_apply = (
                rules.max_notional_apply_to_market
                if market
                else True
            )

            if (
                must_apply
                and notional > rules.max_notional
            ):

                raise BinanceValidationError(
                    f"Order notional {notional} exceeds "
                    f"maximum {rules.max_notional} for {symbol}"
                )

        return notional

    def _check_buy_balance(
        self,
        symbol: str,
        required_quote: Decimal,
    ) -> None:

        rules = self._get_symbol_rules(symbol)

        free_quote = self.get_balance_decimal(
            rules.quote_asset
        )

        reserve_multiplier = (
            D("1")
            + (
                self.balance_reserve_pct
                / D("100")
            )
        )

        required_with_reserve = (
            required_quote
            * reserve_multiplier
        )

        if free_quote < required_with_reserve:

            raise BinanceInsufficientBalance(
                f"Insufficient {rules.quote_asset}: "
                f"free={free_quote}, "
                f"required≈{required_with_reserve}"
            )

    def _check_sell_balance(
        self,
        symbol: str,
        quantity: Decimal,
    ) -> None:

        rules = self._get_symbol_rules(symbol)

        free_base = self.get_balance_decimal(
            rules.base_asset
        )

        if free_base < quantity:

            raise BinanceInsufficientBalance(
                f"Insufficient {rules.base_asset}: "
                f"free={free_base}, requested={quantity}"
            )

    # ======================================================================
    # Open order / algorithmic order protection
    # ======================================================================

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        try:

            if symbol:

                return self.client.get_open_orders(
                    symbol=symbol.upper()
                )

            return self.client.get_open_orders()

        except Exception as exc:

            raise BinanceClientError(
                "Failed to read open orders."
            ) from exc

    def _check_open_order_limits(
        self,
        symbol: str,
        new_order_type: str,
    ) -> None:

        rules = self._get_symbol_rules(symbol)

        open_orders = self.get_open_orders(symbol)

        if (
            rules.max_num_orders > 0
            and len(open_orders)
            >= rules.max_num_orders
        ):

            raise BinanceMarketSafetyError(
                f"MAX_NUM_ORDERS reached for {symbol}"
            )

        algo_types = {
            "STOP_LOSS",
            "STOP_LOSS_LIMIT",
            "TAKE_PROFIT",
            "TAKE_PROFIT_LIMIT",
        }

        if (
            new_order_type in algo_types
            and rules.max_num_algo_orders > 0
        ):

            algo_count = sum(
                1
                for order in open_orders
                if order.get("type") in algo_types
            )

            if (
                algo_count
                >= rules.max_num_algo_orders
            ):

                raise BinanceMarketSafetyError(
                    f"MAX_NUM_ALGO_ORDERS reached for {symbol}"
                )

    # ======================================================================
    # Order ID
    # ======================================================================

    @staticmethod
    def _client_order_id(prefix: str) -> str:
        """
        Short unique client order ID.

        Example:
            BOT_MKT_A13F8C92
        """

        clean_prefix = (
            "".join(
                ch
                for ch in prefix.upper()
                if ch.isalnum()
            )[:8]
        )

        return (
            f"BOT_{clean_prefix}_"
            f"{uuid.uuid4().hex[:12].upper()}"
        )

    # ======================================================================
    # Raw order submission
    # ======================================================================

    def _submit_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        **params: Any,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        client_order_id = params.get(
            "newClientOrderId"
        )

        if not client_order_id:

            client_order_id = self._client_order_id(
                order_type
            )

            params["newClientOrderId"] = (
                client_order_id
            )

        # --------------------------------------------------------------
        # DRY RUN
        # --------------------------------------------------------------

        if self.dry_run:

            log.warning(
                "[BINANCE] DRY-RUN order skipped | "
                "%s %s %s | params=%s",
                side,
                symbol,
                order_type,
                params,
            )

            return {
                "dryRun": True,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "clientOrderId": client_order_id,
                "status": "DRY_RUN",
                **params,
            }

        # --------------------------------------------------------------
        # Live/Testnet submission
        # --------------------------------------------------------------

        try:

            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                recvWindow=self.recv_window,
                **params,
            )

            log.info(
                "[BINANCE] Order accepted | "
                "symbol=%s side=%s type=%s orderId=%s clientOrderId=%s",
                symbol,
                side,
                order_type,
                order.get("orderId"),
                order.get(
                    "clientOrderId",
                    client_order_id,
                ),
            )

            return order

        except BinanceRequestException as exc:

            # ----------------------------------------------------------
            # IMPORTANT:
            # A network timeout does NOT prove that Binance rejected
            # the order.
            #
            # Therefore we NEVER blindly resend the order.
            # We first attempt reconciliation using clientOrderId.
            # ----------------------------------------------------------

            log.warning(
                "[BINANCE] Transport error after order submission; "
                "attempting reconciliation | symbol=%s clientOrderId=%s",
                symbol,
                client_order_id,
            )

            recovered = self._reconcile_order(
                symbol=symbol,
                client_order_id=client_order_id,
            )

            if recovered:

                return recovered

            raise BinanceClientError(
                "Order submission status is uncertain. "
                "No blind retry was performed."
            ) from exc

        except BinanceAPIException as exc:

            log.error(
                "[BINANCE] API rejected order | "
                "symbol=%s side=%s type=%s code=%s message=%s",
                symbol,
                side,
                order_type,
                getattr(exc, "code", None),
                getattr(exc, "message", str(exc)),
            )

            raise BinanceOrderRejected(
                f"Binance rejected order: {exc}"
            ) from exc

        except BinanceOrderException as exc:

            log.error(
                "[BINANCE] Order exception | "
                "symbol=%s error=%s",
                symbol,
                exc,
            )

            raise BinanceOrderRejected(
                f"Binance order exception: {exc}"
            ) from exc

    def _reconcile_order(
        self,
        symbol: str,
        client_order_id: str,
    ) -> Optional[Dict[str, Any]]:

        delays = (
            0.25,
            0.75,
            1.50,
        )

        for delay in delays:

            time.sleep(delay)

            try:

                order = self.client.get_order(
                    symbol=symbol,
                    origClientOrderId=client_order_id,
                    recvWindow=self.recv_window,
                )

                if order:

                    log.info(
                        "[BINANCE] Reconciled order | "
                        "symbol=%s orderId=%s status=%s",
                        symbol,
                        order.get("orderId"),
                        order.get("status"),
                    )

                    return order

            except Exception:
                continue

        return None

    # ======================================================================
    # Market BUY
    # ======================================================================

    def place_market_buy(
        self,
        symbol: str,
        quantity: Any,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        rules = self._get_symbol_rules(
            symbol
        )

        if rules.status != "TRADING":

            raise BinanceValidationError(
                f"{symbol} is not in TRADING status: "
                f"{rules.status}"
            )

        if (
            ORDER_TYPE_MARKET
            not in rules.order_types
        ):

            raise BinanceValidationError(
                f"MARKET orders are not supported for {symbol}"
            )

        normalized_qty = self.normalize_quantity(
            symbol,
            quantity,
            market=True,
        )

        estimate = self._check_market_conditions(
            symbol=symbol,
            side=SIDE_BUY,
            quantity=normalized_qty,
        )

        estimated_notional = self._check_notional(
            symbol=symbol,
            quantity=normalized_qty,
            price=estimate.estimated_vwap,
            market=True,
        )

        self._check_buy_balance(
            symbol=symbol,
            required_quote=estimated_notional,
        )

        self._check_open_order_limits(
            symbol=symbol,
            new_order_type=ORDER_TYPE_MARKET,
        )

        log.info(
            "[BINANCE] Market BUY validation passed | "
            "symbol=%s quantity=%s estVWAP=%s spread=%s bps slippage=%s bps",
            symbol,
            decimal_to_str(normalized_qty),
            decimal_to_str(estimate.estimated_vwap),
            decimal_to_str(
                self.get_market_snapshot(
                    symbol
                ).spread_bps
            ),
            decimal_to_str(
                estimate.slippage_bps
            ),
        )

        return self._submit_order(
            symbol=symbol,
            side=SIDE_BUY,
            order_type=ORDER_TYPE_MARKET,
            quantity=decimal_to_str(
                normalized_qty
            ),
            newOrderRespType="FULL",
        )

    # ======================================================================
    # Market SELL
    # ======================================================================

    def place_market_sell(
        self,
        symbol: str,
        quantity: Any,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        rules = self._get_symbol_rules(
            symbol
        )

        if rules.status != "TRADING":

            raise BinanceValidationError(
                f"{symbol} is not in TRADING status: "
                f"{rules.status}"
            )

        if (
            ORDER_TYPE_MARKET
            not in rules.order_types
        ):

            raise BinanceValidationError(
                f"MARKET orders are not supported for {symbol}"
            )

        normalized_qty = self.normalize_quantity(
            symbol,
            quantity,
            market=True,
        )

        estimate = self._check_market_conditions(
            symbol=symbol,
            side=SIDE_SELL,
            quantity=normalized_qty,
        )

        estimated_notional = self._check_notional(
            symbol=symbol,
            quantity=normalized_qty,
            price=estimate.estimated_vwap,
            market=True,
        )

        self._check_sell_balance(
            symbol=symbol,
            quantity=normalized_qty,
        )

        self._check_open_order_limits(
            symbol=symbol,
            new_order_type=ORDER_TYPE_MARKET,
        )

        log.info(
            "[BINANCE] Market SELL validation passed | "
            "symbol=%s quantity=%s estVWAP=%s slippage=%s bps",
            symbol,
            decimal_to_str(normalized_qty),
            decimal_to_str(estimate.estimated_vwap),
            decimal_to_str(estimate.slippage_bps),
        )

        return self._submit_order(
            symbol=symbol,
            side=SIDE_SELL,
            order_type=ORDER_TYPE_MARKET,
            quantity=decimal_to_str(
                normalized_qty
            ),
            newOrderRespType="FULL",
        )

    # ======================================================================
    # Limit order
    # ======================================================================

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Any,
        price: Any,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()
        side = side.upper()

        if side not in {
            SIDE_BUY,
            SIDE_SELL,
        }:

            raise BinanceValidationError(
                f"Unsupported side: {side}"
            )

        rules = self._get_symbol_rules(
            symbol
        )

        if rules.status != "TRADING":

            raise BinanceValidationError(
                f"{symbol} is not tradable: {rules.status}"
            )

        if (
            ORDER_TYPE_LIMIT
            not in rules.order_types
        ):

            raise BinanceValidationError(
                f"LIMIT orders are not supported for {symbol}"
            )

        normalized_qty = self.normalize_quantity(
            symbol,
            quantity,
            market=False,
        )

        normalized_price = self.normalize_price(
            symbol,
            price,
        )

        notional = self._check_notional(
            symbol=symbol,
            quantity=normalized_qty,
            price=normalized_price,
            market=False,
        )

        if side == SIDE_BUY:

            self._check_buy_balance(
                symbol=symbol,
                required_quote=notional,
            )

        else:

            self._check_sell_balance(
                symbol=symbol,
                quantity=normalized_qty,
            )

        self._check_open_order_limits(
            symbol=symbol,
            new_order_type=ORDER_TYPE_LIMIT,
        )

        return self._submit_order(
            symbol=symbol,
            side=side,
            order_type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=decimal_to_str(
                normalized_qty
            ),
            price=decimal_to_str(
                normalized_price
            ),
            newOrderRespType="RESULT",
        )

    # ======================================================================
    # Stop Loss
    # ======================================================================

    def place_stop_loss(
        self,
        symbol: str,
        quantity: Any,
        stop_price: Any,
        limit_price: Optional[Any] = None,
        limit_offset_bps: Optional[Any] = None,
    ) -> Dict[str, Any]:

        """
        Place SELL STOP_LOSS_LIMIT.

        المنطق:
            current price
                 ↓
              stop_price
                 ↓
              limit_price

        لا نستخدم:
            price = stop_price * 0.99

        بشكل ثابت.

        بل:
            1. نأخذ tick size الحقيقي.
            2. نطبع stopPrice إلى tick صحيح.
            3. نطبع limitPrice إلى tick صحيح.
            4. نضمن أن limitPrice < stopPrice.
            5. نرفض stop أعلى من السعر الحالي في Stop Loss sell.
        """

        symbol = symbol.upper()

        rules = self._get_symbol_rules(
            symbol
        )

        if rules.status != "TRADING":

            raise BinanceValidationError(
                f"{symbol} is not tradable: {rules.status}"
            )

        if (
            ORDER_TYPE_STOP_LOSS_LIMIT
            not in rules.order_types
        ):

            raise BinanceValidationError(
                f"STOP_LOSS_LIMIT is not supported for {symbol}"
            )

        normalized_qty = self.normalize_quantity(
            symbol,
            quantity,
            market=False,
        )

        normalized_stop = self.normalize_price(
            symbol,
            stop_price,
        )

        current_price = self.get_price_decimal(
            symbol
        )

        # --------------------------------------------------------------
        # Prevent an invalid immediate stop trigger.
        # --------------------------------------------------------------

        if normalized_stop >= current_price:

            raise BinanceValidationError(
                f"SELL stop price {normalized_stop} "
                f"is not below current price {current_price}"
            )

        # --------------------------------------------------------------
        # Dynamic limit price.
        # --------------------------------------------------------------

        if limit_price is not None:

            normalized_limit = self.normalize_price(
                symbol,
                limit_price,
            )

        else:

            offset_bps = (
                D(limit_offset_bps)
                if limit_offset_bps is not None
                else self.stop_limit_offset_bps
            )

            if offset_bps <= 0:

                raise BinanceValidationError(
                    "limit_offset_bps must be > 0"
                )

            offset_fraction = (
                offset_bps
                / D("10000")
            )

            raw_limit = (
                normalized_stop
                * (
                    D("1")
                    - offset_fraction
                )
            )

            normalized_limit = self.normalize_price(
                symbol,
                raw_limit,
            )

        # --------------------------------------------------------------
        # Critical relationship:
        #
        # SELL STOP_LOSS_LIMIT:
        #
        #     current price
        #          >
        #     stop price
        #          >
        #     limit price
        #
        # otherwise the order structure can be invalid.
        # --------------------------------------------------------------

        if normalized_limit >= normalized_stop:

            tick = rules.tick_size

            if tick <= 0:

                raise BinanceValidationError(
                    "Cannot adjust stop-limit price without tickSize."
                )

            normalized_limit = (
                normalized_stop - tick
            )

            normalized_limit = self.normalize_price(
                symbol,
                normalized_limit,
            )

        if normalized_limit <= 0:

            raise BinanceValidationError(
                f"Invalid calculated limit price: "
                f"{normalized_limit}"
            )

        notional = self._check_notional(
            symbol=symbol,
            quantity=normalized_qty,
            price=normalized_limit,
            market=False,
        )

        self._check_sell_balance(
            symbol=symbol,
            quantity=normalized_qty,
        )

        self._check_open_order_limits(
            symbol=symbol,
            new_order_type=ORDER_TYPE_STOP_LOSS_LIMIT,
        )

        log.info(
            "[BINANCE] Stop Loss prepared | "
            "symbol=%s qty=%s stop=%s limit=%s notional=%s",
            symbol,
            decimal_to_str(normalized_qty),
            decimal_to_str(normalized_stop),
            decimal_to_str(normalized_limit),
            decimal_to_str(notional),
        )

        return self._submit_order(
            symbol=symbol,
            side=SIDE_SELL,
            order_type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=decimal_to_str(
                normalized_qty
            ),
            price=decimal_to_str(
                normalized_limit
            ),
            stopPrice=decimal_to_str(
                normalized_stop
            ),
            newOrderRespType="RESULT",
        )

    # ======================================================================
    # Trailing Stop
    # ======================================================================

    def place_trailing_stop_loss(
        self,
        symbol: str,
        quantity: Any,
        trailing_delta_bips: int,
        limit_offset_bps: Optional[Any] = None,
    ) -> Dict[str, Any]:

        """
        Advanced Spot trailing stop.

        trailing_delta_bips:
            Binance trailingDelta value in BIPS.

        Example:
            100 BIPS = 1%

        The symbol must support TRAILING_DELTA.
        """

        symbol = symbol.upper()

        rules = self._get_symbol_rules(
            symbol
        )

        if (
            ORDER_TYPE_STOP_LOSS_LIMIT
            not in rules.order_types
        ):

            raise BinanceValidationError(
                f"STOP_LOSS_LIMIT is not supported for {symbol}"
            )

        delta = int(
            trailing_delta_bips
        )

        if delta <= 0:

            raise BinanceValidationError(
                "trailing_delta_bips must be > 0"
            )

        if (
            rules.min_trailing_below_delta > 0
            and delta
            < rules.min_trailing_below_delta
        ):

            raise BinanceValidationError(
                f"Trailing delta {delta} is below "
                f"minimum {rules.min_trailing_below_delta}"
            )

        if (
            rules.max_trailing_below_delta > 0
            and delta
            > rules.max_trailing_below_delta
        ):

            raise BinanceValidationError(
                f"Trailing delta {delta} exceeds "
                f"maximum {rules.max_trailing_below_delta}"
            )

        normalized_qty = self.normalize_quantity(
            symbol,
            quantity,
            market=False,
        )

        current_price = self.get_price_decimal(
            symbol
        )

        # A reference stop price is needed when using
        # a STOP_LOSS_LIMIT implementation.
        #
        # We place a reference price slightly below current price.
        offset_bps = (
            D(limit_offset_bps)
            if limit_offset_bps is not None
            else self.stop_limit_offset_bps
        )

        stop_reference = current_price * (
            D("1")
            - (
                offset_bps
                / D("10000")
            )
        )

        stop_reference = self.normalize_price(
            symbol,
            stop_reference,
        )

        limit_price = self.normalize_price(
            symbol,
            stop_reference * (
                D("1")
                - (
                    offset_bps
                    / D("10000")
                )
            ),
        )

        if limit_price >= stop_reference:

            limit_price = self.normalize_price(
                symbol,
                stop_reference
                - rules.tick_size,
            )

        self._check_notional(
            symbol=symbol,
            quantity=normalized_qty,
            price=limit_price,
            market=False,
        )

        self._check_sell_balance(
            symbol=symbol,
            quantity=normalized_qty,
        )

        self._check_open_order_limits(
            symbol=symbol,
            new_order_type=ORDER_TYPE_STOP_LOSS_LIMIT,
        )

        return self._submit_order(
            symbol=symbol,
            side=SIDE_SELL,
            order_type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=decimal_to_str(
                normalized_qty
            ),
            price=decimal_to_str(
                limit_price
            ),
            stopPrice=decimal_to_str(
                stop_reference
            ),
            trailingDelta=delta,
            newOrderRespType="RESULT",
        )

    # ======================================================================
    # Safe test order
    # ======================================================================

    def test_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        **params: Any,
    ) -> Dict[str, Any]:

        """
        Binance validation endpoint.

        It validates the order without placing it.

        Useful during development/Testnet.
        """

        symbol = symbol.upper()
        side = side.upper()
        order_type = order_type.upper()

        try:

            result = self.client.create_test_order(
                symbol=symbol,
                side=side,
                type=order_type,
                **params,
            )

            log.info(
                "[BINANCE] Test order accepted | "
                "symbol=%s side=%s type=%s",
                symbol,
                side,
                order_type,
            )

            return result or {}

        except Exception as exc:

            log.error(
                "[BINANCE] Test order rejected | "
                "symbol=%s side=%s type=%s error=%s",
                symbol,
                side,
                order_type,
                exc,
            )

            raise BinanceOrderRejected(
                f"Test order rejected: {exc}"
            ) from exc

    # ======================================================================
    # Order state
    # ======================================================================

    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        if (
            order_id is None
            and not client_order_id
        ):

            raise BinanceValidationError(
                "Either order_id or client_order_id is required."
            )

        params: Dict[str, Any] = {
            "symbol": symbol,
            "recvWindow": self.recv_window,
        }

        if order_id is not None:

            params["orderId"] = int(
                order_id
            )

        else:

            params[
                "origClientOrderId"
            ] = client_order_id

        try:

            return self.client.get_order(
                **params
            )

        except Exception as exc:

            raise BinanceClientError(
                f"Failed to query order for {symbol}"
            ) from exc

    def wait_for_order(
        self,
        symbol: str,
        order_id: int,
        timeout_seconds: float = 30.0,
        poll_seconds: float = 0.5,
    ) -> Dict[str, Any]:

        start = time.monotonic()

        while (
            time.monotonic() - start
            < timeout_seconds
        ):

            order = self.get_order(
                symbol=symbol,
                order_id=order_id,
            )

            status = order.get(
                "status"
            )

            if status in {
                ORDER_STATUS_FILLED,
                ORDER_STATUS_CANCELED,
                ORDER_STATUS_REJECTED,
                ORDER_STATUS_EXPIRED,
            }:

                return order

            time.sleep(
                max(0.1, poll_seconds)
            )

        return self.get_order(
            symbol=symbol,
            order_id=order_id,
        )

    # ======================================================================
    # Cancel
    # ======================================================================

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        symbol = symbol.upper()

        if (
            order_id is None
            and not client_order_id
        ):

            raise BinanceValidationError(
                "Either order_id or client_order_id is required."
            )

        params: Dict[str, Any] = {
            "symbol": symbol,
            "recvWindow": self.recv_window,
        }

        if order_id is not None:

            params["orderId"] = int(
                order_id
            )

        else:

            params[
                "origClientOrderId"
            ] = client_order_id

        try:

            result = self.client.cancel_order(
                **params
            )

            log.info(
                "[BINANCE] Order canceled | "
                "symbol=%s orderId=%s",
                symbol,
                result.get("orderId"),
            )

            return result

        except BinanceAPIException as exc:

            raise BinanceOrderRejected(
                f"Cancel rejected: {exc}"
            ) from exc

        except Exception as exc:

            raise BinanceClientError(
                f"Failed to cancel order for {symbol}"
            ) from exc

    # ======================================================================
    # Cancel all open orders for symbol
    # ======================================================================

    def cancel_all_orders(
        self,
        symbol: str,
    ) -> List[Dict[str, Any]]:

        symbol = symbol.upper()

        try:

            open_orders = self.get_open_orders(
                symbol
            )

            canceled = []

            for order in open_orders:

                order_id = order.get(
                    "orderId"
                )

                if order_id is None:
                    continue

                try:

                    result = self.cancel_order(
                        symbol=symbol,
                        order_id=order_id,
                    )

                    canceled.append(
                        result
                    )

                except Exception as exc:

                    log.error(
                        "[BINANCE] Failed to cancel "
                        "orderId=%s: %s",
                        order_id,
                        exc,
                    )

            return canceled

        except Exception as exc:

            raise BinanceClientError(
                f"Failed to cancel all orders for {symbol}"
            ) from exc

    # ======================================================================
    # Utility: position
    # ======================================================================

    def get_position(
        self,
        symbol: str,
    ) -> Decimal:

        rules = self._get_symbol_rules(
            symbol
        )

        return self.get_balance_decimal(
            rules.base_asset
        )

    # ======================================================================
    # Utility: execution diagnostics
    # ======================================================================

    def execution_diagnostics(
        self,
        symbol: str,
        quantity: Any,
        side: str = SIDE_BUY,
    ) -> Dict[str, Any]:

        """
        Diagnostic snapshot.

        لا ينفذ أي أمر.

        Returns:
            - best bid
            - best ask
            - spread
            - spread bps
            - estimated VWAP
            - estimated slippage
            - estimated notional
        """

        symbol = symbol.upper()
        side = side.upper()

        qty = self.normalize_quantity(
            symbol,
            quantity,
            market=True,
        )

        snapshot = self.get_market_snapshot(
            symbol
        )

        estimate = self._estimate_market_execution(
            symbol=symbol,
            side=side,
            quantity=qty,
        )

        return {
            "symbol": symbol,
            "side": side,
            "quantity": decimal_to_str(qty),
            "bid": decimal_to_str(snapshot.bid),
            "ask": decimal_to_str(snapshot.ask),
            "mid": decimal_to_str(snapshot.mid),
            "spread": decimal_to_str(snapshot.spread),
            "spread_bps": float(snapshot.spread_bps),
            "last_price": decimal_to_str(
                snapshot.last_price
            ),
            "estimated_vwap": decimal_to_str(
                estimate.estimated_vwap
            ),
            "estimated_slippage_bps": float(
                estimate.slippage_bps
            ),
            "estimated_notional": decimal_to_str(
                estimate.quote_value
            ),
            "market_safe": (
                snapshot.spread_bps
                <= self.max_spread_bps
                and
                estimate.slippage_bps
                <= self.max_slippage_bps
            ),
        }


# ============================================================================
# Example
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s - "
            "%(message)s"
        ),
    )

    # ------------------------------------------------------------------------
    # Testnet by default.
    #
    # Required environment variables:
    #
    # BINANCE_TESTNET_API_KEY
    # BINANCE_TESTNET_SECRET_KEY
    #
    # Optional:
    #
    # BINANCE_MAX_SPREAD_BPS=30
    # BINANCE_MAX_SLIPPAGE_BPS=100
    # BINANCE_MAX_ORDER_NOTIONAL=20
    # BINANCE_BALANCE_RESERVE_PCT=0.20
    # BINANCE_STOP_LIMIT_OFFSET_BPS=25
    # ------------------------------------------------------------------------

    client = BinanceClient(
        testnet=True,
        dry_run=True,
    )

    symbol = "BTCUSDT"

    print(
        client.execution_diagnostics(
            symbol=symbol,
            quantity="0.0001",
            side=SIDE_BUY,
        )
    )

    print(
        "Current price:",
        client.get_price_decimal(symbol),
    )

    print(
        "USDT balance:",
        client.get_balance_decimal("USDT"),
    )

    # ------------------------------------------------------------------------
    # Real Testnet market order example:
    #
    # client = BinanceClient(
    #     testnet=True,
    #     dry_run=False,
    # )
    #
    # order = client.place_market_buy(
    #     symbol="BTCUSDT",
    #     quantity="0.0001",
    # )
    #
    # print(order)
    # ------------------------------------------------------------------------
