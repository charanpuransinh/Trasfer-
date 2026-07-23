"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              UNIFIED STOCK SCANNER — Shakti + Core Strategy Fusion            ║
║                                                                                ║
║  This file merges two previously separate modules into one deployable        ║
║  scanner:                                                                     ║
║                                                                                ║
║  PART 1 (from shakti_scanner_v1.py):                                         ║
║    Raw OHLCV -> IndicatorEngine (65+ indicators) -> ScoringEngine ->          ║
║    ShaktiScanner.scan() -> ScannerResult (technical score, smart-money        ║
║    signals, ADX trend, etc.)                                                  ║
║                                                                                ║
║  PART 2 (from stock_scanner_core-10.py):                                     ║
║    Decision/fusion layer — takes already-scored inputs (chart, flow,          ║
║    option_chain, greeks, mtf, price action) and runs them through            ║
║    AutoTradeValidator, InstitutionalFlowEngine, PriceActionEngine,            ║
║    AIExecutionGuardian, and the master HighAccuracyFusionEngine.             ║
║                                                                                ║
║  PART 3 (new — UnifiedScannerEngine):                                        ║
║    The bridge. Runs ShaktiScanner on your OHLCV data, converts its output     ║
║    into the dict shapes Part 2's engines expect, and drives the full          ║
║    Core decision pipeline end-to-end from a single async call.               ║
║                                                                                ║
║  ── KNOWN LIMITATION (flagging, not hiding) ──────────────────────────────    ║
║  ShaktiScanner.engine.multi_timeframe_alignment() returns HARDCODED demo      ║
║  values (Monthly/Weekly/Daily/... all fixed BULLISH/BEARISH labels) — it      ║
║  was never wired to real multi-timeframe candle data in the original file.   ║
║  UnifiedScannerEngine uses this as-is for now, so `mtf_confluence` in the     ║
║  final output is NOT reliable yet. Core's own MultiTimeframeConfluenceEngine  ║
║  needs real per-timeframe (ema20/50/200, price, vwap, rsi) data per symbol    ║
║  which isn't available from a single-timeframe df — wire this up when you     ║
║  have a real multi-timeframe data source (see run_full_scan() docstring).    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SHAKTI SCANNER v1.0                                   ║
║              High-Accuracy Multi-Timeframe Stock Market Scanner              ║
║                                                                              ║
║  Features:                                                                   ║
║  • 65+ Technical Indicators (Trend, Momentum, Volatility, Volume)          ║
║  • ADX/DMI with Trend Strength Classification                                ║
║  • Smart Money Concepts (Order Blocks, FVG, Liquidity Sweeps, BOS/CHoCH)    ║
║  • Volume Profile with Point of Control (POC)                                ║
║  • Order Book Imbalance & Monopoly Money Flow                                ║
║  • Multi-Timeframe Confluence Engine (1m to Monthly)                         ║
║  • Composite Scoring System (0-100)                                          ║
║  • Session Analysis & Kill Zone Detection                                     ║
║                                                                              ║
║  Built for: NSE (India), Forex, Crypto, US Equities                         ║
║  Language: Python 3.10+ | Dependencies: pandas, numpy                        ║
║  License: Institutional Use | Zero Tolerance for Bugs                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class TrendStrength(Enum):
    NO_TREND = "NO_TREND"
    WEAK_TREND = "WEAK_TREND"
    MODERATE_TREND = "MODERATE_TREND"
    STRONG_TREND = "STRONG_TREND"
    VERY_STRONG_TREND = "VERY_STRONG_TREND"

@dataclass
class ScannerResult:
    """Container for scanner output"""
    symbol: str
    timestamp: str
    composite_score: float
    signal: SignalType
    action: str
    trend_score: float
    momentum_score: float
    volume_score: float
    smart_money_score: float
    mtf_confluence: float
    adx_value: float
    adx_trend: str
    indicators: Dict[str, float]
    smart_money_signals: Dict[str, any]
    mtf_alignment: Dict[str, Dict]

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'composite_score': round(self.composite_score, 2),
            'signal': self.signal.value,
            'action': self.action,
            'trend_score': round(self.trend_score, 2),
            'momentum_score': round(self.momentum_score, 2),
            'volume_score': round(self.volume_score, 2),
            'smart_money_score': round(self.smart_money_score, 2),
            'mtf_confluence': round(self.mtf_confluence, 2),
            'adx_value': round(self.adx_value, 2),
            'adx_trend': self.adx_trend,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SAFE MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def safe_divide(a, b, default=0.0):
    """Safe division to prevent division by zero - handles both scalar and Series"""
    # Handle pandas Series
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        result = a.div(b.replace(0, np.nan)).fillna(default)
        return result.replace([np.inf, -np.inf], default)
    elif isinstance(a, pd.Series):
        b_safe = b if b != 0 else np.nan
        result = a / b_safe
        return result.replace([np.inf, -np.inf, np.nan], default)
    elif isinstance(b, pd.Series):
        result = a / b.replace(0, np.nan)
        return result.replace([np.inf, -np.inf, np.nan], default)

    # Handle numpy arrays
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b)
            result = np.where(np.isfinite(result), result, default)
            return result

    # Handle scalars
    try:
        if b == 0 or b == 0.0:
            return default
        result = a / b
        return default if not np.isfinite(result) else result
    except:
        return default

def safe_sqrt(x, default=0.0):
    """Safe square root"""
    if isinstance(x, pd.Series):
        return np.sqrt(x.clip(lower=0)).fillna(default).replace([np.inf, -np.inf], default)
    if isinstance(x, np.ndarray):
        return np.sqrt(np.clip(x, 0, None))
    return np.sqrt(max(x, 0)) if x >= 0 else default

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CORE INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """Calculates all 65+ technical indicators"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_data()

    def _validate_data(self):
        """Validate OHLCV data integrity"""
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Data integrity checks
        assert (self.df['high'] >= self.df['low']).all(), "High must be >= Low"
        assert (self.df['close'] >= self.df['low']).all(), "Close must be >= Low"
        assert (self.df['close'] <= self.df['high']).all(), "Close must be <= High"
        assert (self.df['volume'] > 0).all(), "Volume must be positive"

    # ─── TREND INDICATORS (10) ───
    def sma(self, series, period):
        return series.rolling(window=period).mean()

    def ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def wma(self, series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def hma(self, series, period):
        half = int(period / 2)
        sqrt_p = int(np.sqrt(period))
        return self.wma(2 * self.wma(series, half) - self.wma(series, period), sqrt_p)

    def dema(self, series, period):
        e1 = self.ema(series, period)
        e2 = self.ema(e1, period)
        return 2 * e1 - e2

    def tema(self, series, period):
        e1 = self.ema(series, period)
        e2 = self.ema(e1, period)
        e3 = self.ema(e2, period)
        return 3 * e1 - 3 * e2 + e3

    def kama(self, series, n=10, fast=2, slow=30):
        change = abs(series - series.shift(n))
        volatility = abs(series - series.shift(1)).rolling(n).sum()
        er = safe_divide(change, volatility, 0)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = series.copy()
        for i in range(n, len(series)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
        return kama

    def atr(self, period=14):
        tr1 = self.df['high'] - self.df['low']
        tr2 = abs(self.df['high'] - self.df['close'].shift(1))
        tr3 = abs(self.df['low'] - self.df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def parabolic_sar(self, af=0.02, max_af=0.2):
        high, low, close = self.df['high'], self.df['low'], self.df['close']
        n = len(close)
        sar = np.zeros(n)
        trend = np.zeros(n)
        ep = np.zeros(n)
        af_vals = np.zeros(n)

        trend[0] = 1 if close.iloc[0] >= close.iloc[1] else -1
        sar[0] = low.iloc[0] if trend[0] == 1 else high.iloc[0]
        ep[0] = high.iloc[0] if trend[0] == 1 else low.iloc[0]
        af_vals[0] = af

        for i in range(1, n):
            sar[i] = sar[i-1] + af_vals[i-1] * (ep[i-1] - sar[i-1])

            if trend[i-1] == 1:
                if low.iloc[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af_vals[i] = af
                else:
                    trend[i] = 1
                    ep[i] = max(high.iloc[i], ep[i-1])
                    af_vals[i] = min(af_vals[i-1] + af, max_af) if ep[i] != ep[i-1] else af_vals[i-1]
            else:
                if high.iloc[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af_vals[i] = af
                else:
                    trend[i] = -1
                    ep[i] = min(low.iloc[i], ep[i-1])
                    af_vals[i] = min(af_vals[i-1] + af, max_af) if ep[i] != ep[i-1] else af_vals[i-1]

        return pd.Series(sar, index=close.index), pd.Series(trend, index=close.index)

    def supertrend(self, period=10, multiplier=3):
        hl2 = (self.df['high'] + self.df['low']) / 2
        atr_val = self.atr(period)
        upper = hl2 + multiplier * atr_val
        lower = hl2 - multiplier * atr_val

        st = np.zeros(len(self.df))
        direction = np.zeros(len(self.df))

        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > upper.iloc[i-1]:
                direction[i] = 1
            elif self.df['close'].iloc[i] < lower.iloc[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]

            if direction[i] == 1:
                st[i] = max(lower.iloc[i], st[i-1] if i > 0 else lower.iloc[i])
            else:
                st[i] = min(upper.iloc[i], st[i-1] if i > 0 else upper.iloc[i])

        return pd.Series(st, index=self.df.index), pd.Series(direction, index=self.df.index)

    def ichimoku(self):
        h, l, c = self.df['high'], self.df['low'], self.df['close']
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        chikou = c.shift(-26)
        return tenkan, kijun, senkou_a, senkou_b, chikou

    # ─── MOMENTUM INDICATORS (10) ───
    def rsi(self, period=14):
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = safe_divide(avg_gain, avg_loss, 0)
        return 100 - safe_divide(100, (1 + rs), 100)

    def stochastic(self, k_period=14, d_period=3):
        lowest = self.df['low'].rolling(k_period).min()
        highest = self.df['high'].rolling(k_period).max()
        k = 100 * safe_divide((self.df['close'] - lowest), (highest - lowest), 50)
        d = k.rolling(d_period).mean()
        return k, d

    def macd(self, fast=12, slow=26, signal=9):
        macd_line = self.ema(self.df['close'], fast) - self.ema(self.df['close'], slow)
        signal_line = self.ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def cci(self, period=20):
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma_tp = self.sma(tp, period)
        mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return safe_divide((tp - sma_tp), (0.015 * mean_dev), 0)

    def williams_r(self, period=14):
        highest = self.df['high'].rolling(period).max()
        lowest = self.df['low'].rolling(period).min()
        return -100 * safe_divide((highest - self.df['close']), (highest - lowest), 0)

    def momentum(self, period=10):
        return self.df['close'] - self.df['close'].shift(period)

    def roc(self, period=12):
        return safe_divide((self.df['close'] - self.df['close'].shift(period)), 
                          self.df['close'].shift(period), 0) * 100

    def ultimate_oscillator(self, p1=7, p2=14, p3=28):
        bp = self.df['close'] - np.minimum(self.df['low'], self.df['close'].shift(1))
        tr = np.maximum(self.df['high'], self.df['close'].shift(1)) - np.minimum(self.df['low'], self.df['close'].shift(1))

        avg1 = safe_divide(bp.rolling(p1).sum(), tr.rolling(p1).sum(), 0)
        avg2 = safe_divide(bp.rolling(p2).sum(), tr.rolling(p2).sum(), 0)
        avg3 = safe_divide(bp.rolling(p3).sum(), tr.rolling(p3).sum(), 0)

        return 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)

    def trix(self, period=15):
        e1 = self.ema(self.df['close'], period)
        e2 = self.ema(e1, period)
        e3 = self.ema(e2, period)
        return safe_divide((e3 - e3.shift(1)), e3.shift(1), 0) * 100

    def awesome_oscillator(self, fast=5, slow=34):
        median = (self.df['high'] + self.df['low']) / 2
        return self.sma(median, fast) - self.sma(median, slow)

    # ─── VOLATILITY INDICATORS (10) ───
    def bollinger_bands(self, period=20, std_dev=2):
        middle = self.sma(self.df['close'], period)
        std = self.df['close'].rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def keltner_channels(self, period=20, multiplier=2):
        middle = self.ema(self.df['close'], period)
        atr_val = self.atr(period)
        upper = middle + multiplier * atr_val
        lower = middle - multiplier * atr_val
        return upper, middle, lower

    def donchian_channels(self, period=20):
        upper = self.df['high'].rolling(period).max()
        lower = self.df['low'].rolling(period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower

    def std_dev(self, period=20):
        return self.df['close'].rolling(period).std()

    def chaikin_volatility(self, period=10):
        hl_range = self.df['high'] - self.df['low']
        ema_range = self.ema(hl_range, period)
        return safe_divide((ema_range - ema_range.shift(period)), ema_range.shift(period), 0) * 100

    def historical_volatility(self, period=20):
        log_returns = np.log(safe_divide(self.df['close'], self.df['close'].shift(1), 1))
        return log_returns.rolling(period).std() * np.sqrt(252) * 100

    def volatility_index(self, period=14):
        log_hl = np.log(safe_divide(self.df['high'], self.df['low'], 1))
        sum_sq = (log_hl ** 2).rolling(period).sum()
        denom = 4 * period * np.log(2)
        return np.sqrt(safe_divide(sum_sq, denom, 0)) * 100

    # ─── VOLUME INDICATORS (10) ───
    def obv(self):
        obv = pd.Series(index=self.df.index, dtype=float)
        obv.iloc[0] = self.df['volume'].iloc[0]
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.df['volume'].iloc[i]
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    def vwap(self):
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        return (tp * self.df['volume']).cumsum() / self.df['volume'].cumsum()

    def ad_line(self):
        mfm = safe_divide((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close']), 
                         (self.df['high'] - self.df['low']), 0)
        mfv = mfm * self.df['volume']
        return mfv.cumsum()

    def chaikin_money_flow(self, period=20):
        mfm = safe_divide((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close']),
                         (self.df['high'] - self.df['low']), 0)
        return safe_divide((mfm * self.df['volume']).rolling(period).sum(),
                          self.df['volume'].rolling(period).sum(), 0)

    def money_flow_index(self, period=14):
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        raw_mf = tp * self.df['volume']
        delta = tp.diff()
        pos_mf = raw_mf.where(delta > 0, 0)
        neg_mf = raw_mf.where(delta < 0, 0)
        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()
        mfi = 100 - safe_divide(100, (1 + safe_divide(pos_sum, neg_sum, 0)), 100)
        return mfi

    def force_index(self, period=13):
        return self.ema((self.df['close'].diff()) * self.df['volume'], period)

    def ease_of_movement(self, period=14):
        distance = ((self.df['high'] + self.df['low']) / 2) - ((self.df['high'].shift(1) + self.df['low'].shift(1)) / 2)
        box_ratio = safe_divide(self.df['volume'] / 1e9, (self.df['high'] - self.df['low']), 0)
        eom = safe_divide(distance, box_ratio, 0)
        return self.sma(eom, period)

    def negative_volume_index(self):
        nvi = pd.Series(index=self.df.index, dtype=float)
        nvi.iloc[0] = 1000
        for i in range(1, len(self.df)):
            if self.df['volume'].iloc[i] < self.df['volume'].iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] + safe_divide((self.df['close'].iloc[i] - self.df['close'].iloc[i-1]),
                                                          self.df['close'].iloc[i-1], 0) * nvi.iloc[i-1]
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        return nvi

    def positive_volume_index(self):
        pvi = pd.Series(index=self.df.index, dtype=float)
        pvi.iloc[0] = 1000
        for i in range(1, len(self.df)):
            if self.df['volume'].iloc[i] > self.df['volume'].iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] + safe_divide((self.df['close'].iloc[i] - self.df['close'].iloc[i-1]),
                                                          self.df['close'].iloc[i-1], 0) * pvi.iloc[i-1]
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        return pvi

    # ─── ADX/DMI (5) ───
    def adx_dmi(self, period=14):
        tr1 = self.df['high'] - self.df['low']
        tr2 = abs(self.df['high'] - self.df['close'].shift(1))
        tr3 = abs(self.df['low'] - self.df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = tr.rolling(period).mean()

        plus_dm = self.df['high'].diff()
        minus_dm = -self.df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * safe_divide(plus_dm.rolling(period).mean(), atr_val, 0)
        minus_di = 100 * safe_divide(minus_dm.rolling(period).mean(), atr_val, 0)

        dx = 100 * safe_divide(abs(plus_di - minus_di), (plus_di + minus_di), 0)
        adx = dx.rolling(period).mean()

        return adx, plus_di, minus_di

    def trend_strength_classify(self, adx):
        conditions = [
            (adx < 20, TrendStrength.NO_TREND),
            ((adx >= 20) & (adx < 40), TrendStrength.MODERATE_TREND),
            ((adx >= 40) & (adx < 60), TrendStrength.STRONG_TREND),
            (adx >= 60, TrendStrength.VERY_STRONG_TREND),
        ]
        result = pd.Series(index=adx.index, dtype=object)
        for cond, label in conditions:
            result = result.where(~cond, label)
        return result

    # ─── SMART MONEY CONCEPTS (10) ───
    def detect_order_blocks(self, lookback=5, displacement_threshold=0.015):
        bullish_ob = pd.Series(index=self.df.index, dtype=float)
        bearish_ob = pd.Series(index=self.df.index, dtype=float)

        for i in range(lookback, len(self.df)):
            if (self.df['close'].iloc[i] > self.df['open'].iloc[i] and 
                self.df['close'].iloc[i-1] < self.df['open'].iloc[i-1]):
                disp = safe_divide((self.df['close'].iloc[i] - self.df['close'].iloc[i-1]), 
                                  self.df['close'].iloc[i-1], 0)
                if disp > displacement_threshold:
                    bullish_ob.iloc[i] = self.df['low'].iloc[i-1]

            if (self.df['close'].iloc[i] < self.df['open'].iloc[i] and 
                self.df['close'].iloc[i-1] > self.df['open'].iloc[i-1]):
                disp = safe_divide((self.df['close'].iloc[i-1] - self.df['close'].iloc[i]), 
                                  self.df['close'].iloc[i-1], 0)
                if disp > displacement_threshold:
                    bearish_ob.iloc[i] = self.df['high'].iloc[i-1]

        return bullish_ob, bearish_ob

    def detect_fvg(self):
        bullish_fvg = pd.Series(index=self.df.index, dtype=float)
        bearish_fvg = pd.Series(index=self.df.index, dtype=float)

        for i in range(2, len(self.df)):
            if self.df['high'].iloc[i-2] < self.df['low'].iloc[i]:
                bullish_fvg.iloc[i] = self.df['high'].iloc[i-2]
            if self.df['low'].iloc[i-2] > self.df['high'].iloc[i]:
                bearish_fvg.iloc[i] = self.df['low'].iloc[i-2]

        return bullish_fvg, bearish_fvg

    def detect_liquidity_sweeps(self, lookback=10):
        sweep_high = pd.Series(index=self.df.index, dtype=float)
        sweep_low = pd.Series(index=self.df.index, dtype=float)

        for i in range(lookback, len(self.df)):
            recent_high = self.df['high'].iloc[i-lookback:i].max()
            recent_low = self.df['low'].iloc[i-lookback:i].min()

            if self.df['high'].iloc[i] > recent_high and self.df['close'].iloc[i] < recent_high:
                sweep_high.iloc[i] = recent_high
            if self.df['low'].iloc[i] < recent_low and self.df['close'].iloc[i] > recent_low:
                sweep_low.iloc[i] = recent_low

        return sweep_high, sweep_low

    def detect_structure_breaks(self, lookback=5):
        bos_bullish = pd.Series(index=self.df.index, dtype=float)
        bos_bearish = pd.Series(index=self.df.index, dtype=float)
        choch = pd.Series(index=self.df.index, dtype=object)

        for i in range(lookback + 5, len(self.df)):
            swing_high = self.df['high'].iloc[i-lookback-5:i-5].max()
            swing_low = self.df['low'].iloc[i-lookback-5:i-5].min()

            if self.df['close'].iloc[i] > swing_high and self.df['close'].iloc[i-1] <= swing_high:
                bos_bullish.iloc[i] = swing_high
            if self.df['close'].iloc[i] < swing_low and self.df['close'].iloc[i-1] >= swing_low:
                bos_bearish.iloc[i] = swing_low

            if (self.df['close'].iloc[i] > self.df['high'].iloc[i-5] and 
                self.df['close'].iloc[i-1] < self.df['low'].iloc[i-5]):
                choch.iloc[i] = "BULLISH_CHOCH"
            elif (self.df['close'].iloc[i] < self.df['low'].iloc[i-5] and 
                  self.df['close'].iloc[i-1] > self.df['high'].iloc[i-5]):
                choch.iloc[i] = "BEARISH_CHOCH"

        return bos_bullish, bos_bearish, choch

    def smart_money_flow_index(self):
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        raw_mf = tp * self.df['volume']
        smfi = pd.Series(index=self.df.index, dtype=float)
        smfi.iloc[0] = 100000

        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                smfi.iloc[i] = smfi.iloc[i-1] + raw_mf.iloc[i]
            else:
                smfi.iloc[i] = smfi.iloc[i-1]
        return smfi

    def volume_profile(self, bins=20):
        min_p = self.df['close'].min()
        max_p = self.df['close'].max()
        bin_size = safe_divide((max_p - min_p), bins, 1)

        profile = {}
        for i in range(bins):
            low_bin = min_p + i * bin_size
            high_bin = min_p + (i + 1) * bin_size
            mask = (self.df['close'] >= low_bin) & (self.df['close'] < high_bin)
            profile[f"{low_bin:.2f}-{high_bin:.2f}"] = self.df['volume'][mask].sum()

        poc_bin = max(profile, key=profile.get)
        poc = (float(poc_bin.split('-')[0]) + float(poc_bin.split('-')[1])) / 2
        return poc

    def order_book_imbalance(self, period=20):
        up_vol = self.df['volume'].where(self.df['close'] > self.df['close'].shift(1), 0).rolling(period).sum()
        down_vol = self.df['volume'].where(self.df['close'] < self.df['close'].shift(1), 0).rolling(period).sum()
        return safe_divide((up_vol - down_vol), (up_vol + down_vol), 0) * 100

    def monopoly_money_flow(self, period=20):
        position = safe_divide((self.df['close'] - self.df['low']), (self.df['high'] - self.df['low']), 0.5)
        mmf = (2 * position - 1) * self.df['volume']
        return self.sma(mmf, period)

    def detect_smart_money_traps(self, lookback=10):
        bull_trap = pd.Series(index=self.df.index, dtype=float)
        bear_trap = pd.Series(index=self.df.index, dtype=float)

        for i in range(lookback + 5, len(self.df)):
            recent_high = self.df['high'].iloc[i-lookback:i].max()
            recent_low = self.df['low'].iloc[i-lookback:i].min()

            if (self.df['high'].iloc[i-2] > recent_high and 
                self.df['close'].iloc[i] < self.df['open'].iloc[i-2]):
                drop = safe_divide((self.df['high'].iloc[i-2] - self.df['close'].iloc[i]), 
                                  self.df['high'].iloc[i-2], 0)
                if drop > 0.02:
                    bull_trap.iloc[i] = self.df['high'].iloc[i-2]

            if (self.df['low'].iloc[i-2] < recent_low and 
                self.df['close'].iloc[i] > self.df['open'].iloc[i-2]):
                bounce = safe_divide((self.df['close'].iloc[i] - self.df['low'].iloc[i-2]), 
                                    self.df['low'].iloc[i-2], 0)
                if bounce > 0.02:
                    bear_trap.iloc[i] = self.df['low'].iloc[i-2]

        return bull_trap, bear_trap

    def detect_displacement(self, lookback=3, threshold=0.015):
        displacement = pd.Series(index=self.df.index, dtype=float)

        for i in range(lookback, len(self.df)):
            body = abs(self.df['close'].iloc[i] - self.df['open'].iloc[i])
            avg_body = self.df[['open', 'close']].diff(axis=1).abs().iloc[i-lookback:i].mean().mean()
            if body > avg_body * 2 and safe_divide(body, self.df['close'].iloc[i-1], 0) > threshold:
                displacement.iloc[i] = safe_divide(body, self.df['close'].iloc[i-1], 0)

        return displacement

    # ─── MULTI-TIMEFRAME (5) ───
    def multi_timeframe_alignment(self):
        return {
            'Monthly': {'trend': 'BULLISH', 'strength': 0.85, 'signal': 'STRONG_BUY'},
            'Weekly': {'trend': 'BULLISH', 'strength': 0.78, 'signal': 'BUY'},
            'Daily': {'trend': 'BULLISH', 'strength': 0.72, 'signal': 'BUY'},
            '4H': {'trend': 'BULLISH', 'strength': 0.65, 'signal': 'BUY'},
            '1H': {'trend': 'NEUTRAL', 'strength': 0.52, 'signal': 'HOLD'},
            '15m': {'trend': 'BEARISH', 'strength': 0.35, 'signal': 'SELL'},
            '5m': {'trend': 'BEARISH', 'strength': 0.28, 'signal': 'SELL'},
            '1m': {'trend': 'BEARISH', 'strength': 0.20, 'signal': 'SELL'},
        }

    def mtf_confluence_score(self, mtf_data):
        bullish = sum(1 for d in mtf_data.values() if d['trend'] == 'BULLISH')
        bearish = sum(1 for d in mtf_data.values() if d['trend'] == 'BEARISH')
        return (bullish - bearish) / len(mtf_data) * 100

    def mtf_volume_confluence(self, periods=[5, 10, 20, 50]):
        confluence = pd.Series(index=self.df.index, dtype=float)

        for i in range(max(periods), len(self.df)):
            scores = []
            for p in periods:
                current = self.df['volume'].iloc[i]
                avg = self.df['volume'].iloc[i-p:i].mean()
                if current > avg * 1.5:
                    scores.append(1.0)
                elif current > avg:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            confluence.iloc[i] = np.mean(scores)

        return confluence

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMPOSITE SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ScoringEngine:
    """Calculates composite scores from all indicators"""

    def __init__(self, engine: IndicatorEngine):
        self.engine = engine
        self.df = engine.df

    def calculate_trend_score(self) -> float:
        score = 0

        if self.df['SMA_20'].iloc[-1] > self.df['SMA_50'].iloc[-1] > self.df['SMA_200'].iloc[-1]:
            score += 25
        elif self.df['SMA_20'].iloc[-1] > self.df['SMA_50'].iloc[-1]:
            score += 15

        if self.df['EMA_12'].iloc[-1] > self.df['EMA_26'].iloc[-1]:
            score += 15

        if self.df['Supertrend_Dir'].iloc[-1] == 1:
            score += 15

        if (self.df['close'].iloc[-1] > self.df['Senkou_A'].iloc[-1] and 
            self.df['close'].iloc[-1] > self.df['Senkou_B'].iloc[-1]):
            score += 20

        if self.df['PSAR_Trend'].iloc[-1] == 1:
            score += 10

        if self.df['ADX_14'].iloc[-1] > 25:
            score += 15

        return min(score, 100)

    def calculate_momentum_score(self) -> float:
        score = 0

        if 40 < self.df['RSI_14'].iloc[-1] < 60:
            score += 10
        elif self.df['RSI_14'].iloc[-1] > 60:
            score += 20

        if self.df['MACD_Hist'].iloc[-1] > 0:
            score += 20

        if self.df['Stoch_K'].iloc[-1] > self.df['Stoch_D'].iloc[-1]:
            score += 15

        if self.df['CCI_20'].iloc[-1] > 0:
            score += 15

        if self.df['Momentum_10'].iloc[-1] > 0:
            score += 15

        if self.df['ROC_12'].iloc[-1] > 0:
            score += 15

        return min(score, 100)

    def calculate_volume_score(self) -> float:
        score = 0

        if self.df['OBV'].iloc[-1] > self.df['OBV'].iloc[-20]:
            score += 20

        if self.df['close'].iloc[-1] > self.df['VWAP'].iloc[-1]:
            score += 20

        if self.df['CMF_20'].iloc[-1] > 0:
            score += 20

        if self.df['MFI_14'].iloc[-1] > 50:
            score += 20

        if self.df['volume'].iloc[-1] > self.df['volume'].iloc[-20:].mean() * 1.5:
            score += 20

        return min(score, 100)

    def calculate_smart_money_score(self) -> float:
        score = 0

        if self.df['OB_Imbalance'].iloc[-1] > 0:
            score += 25

        if self.df['Monopoly_MF'].iloc[-1] > 0:
            score += 25

        if self.df['SMFI'].iloc[-1] > self.df['SMFI'].iloc[-50]:
            score += 25

        if self.df['MTF_Volume'].iloc[-1] > 0.5:
            score += 25

        return min(score, 100)

    def calculate_composite(self, trend, momentum, volume, smart_money, mtf) -> float:
        weights = {'trend': 0.30, 'momentum': 0.25, 'volume': 0.20, 
                   'smart_money': 0.15, 'mtf': 0.10}
        return (trend * weights['trend'] + momentum * weights['momentum'] +
                volume * weights['volume'] + smart_money * weights['smart_money'] +
                abs(mtf) * weights['mtf'])

    def classify_signal(self, score: float) -> Tuple[SignalType, str]:
        if score >= 80:
            return SignalType.STRONG_BUY, "IMMEDIATE_ENTRY"
        elif score >= 65:
            return SignalType.BUY, "ENTRY_ZONE"
        elif score >= 50:
            return SignalType.WEAK_BUY, "WAIT_FOR_CONFIRMATION"
        elif score >= 40:
            return SignalType.NEUTRAL, "NO_TRADE"
        elif score >= 25:
            return SignalType.WEAK_SELL, "WAIT_FOR_CONFIRMATION"
        elif score >= 10:
            return SignalType.SELL, "ENTRY_ZONE"
        else:
            return SignalType.STRONG_SELL, "IMMEDIATE_ENTRY"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN SCANNER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ShaktiScanner:
    """
    SHAKTI SCANNER - Main Entry Point

    Usage:
        scanner = ShaktiScanner()
        result = scanner.scan(df, symbol="RELIANCE")
        print(result.to_dict())
    """

    def __init__(self):
        self.engine = None
        self.scorer = None

    def scan(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> ScannerResult:
        """Run complete scan on OHLCV data"""

        self.engine = IndicatorEngine(df)
        self.scorer = ScoringEngine(self.engine)

        self._calculate_all_indicators()

        trend_score = self.scorer.calculate_trend_score()
        momentum_score = self.scorer.calculate_momentum_score()
        volume_score = self.scorer.calculate_volume_score()
        sm_score = self.scorer.calculate_smart_money_score()

        mtf_data = self.engine.multi_timeframe_alignment()
        mtf_confluence = self.engine.mtf_confluence_score(mtf_data)

        composite = self.scorer.calculate_composite(
            trend_score, momentum_score, volume_score, sm_score, mtf_confluence
        )

        signal, action = self.scorer.classify_signal(composite)

        adx_trend = self.engine.trend_strength_classify(self.df['ADX_14']).iloc[-1]

        indicators = {
            'SMA_20': round(self.df['SMA_20'].iloc[-1], 4),
            'SMA_50': round(self.df['SMA_50'].iloc[-1], 4),
            'SMA_200': round(self.df['SMA_200'].iloc[-1], 4),
            'EMA_12': round(self.df['EMA_12'].iloc[-1], 4),
            'EMA_26': round(self.df['EMA_26'].iloc[-1], 4),
            'RSI_14': round(self.df['RSI_14'].iloc[-1], 4),
            'MACD': round(self.df['MACD'].iloc[-1], 4),
            'MACD_Signal': round(self.df['MACD_Signal'].iloc[-1], 4),
            'MACD_Hist': round(self.df['MACD_Hist'].iloc[-1], 4),
            'ADX_14': round(self.df['ADX_14'].iloc[-1], 4),
            'DI_Plus': round(self.df['DI_Plus'].iloc[-1], 4),
            'DI_Minus': round(self.df['DI_Minus'].iloc[-1], 4),
            'BB_Upper': round(self.df['BB_Upper'].iloc[-1], 4),
            'BB_Lower': round(self.df['BB_Lower'].iloc[-1], 4),
            'ATR_14': round(self.df['ATR_14'].iloc[-1], 4),
            'VWAP': round(self.df['VWAP'].iloc[-1], 4),
            'OBV': round(self.df['OBV'].iloc[-1], 0),
            'MFI_14': round(self.df['MFI_14'].iloc[-1], 4),
            'Volume': int(self.df['volume'].iloc[-1]),
        }

        smart_money = {
            'POC': round(self.df['POC'].iloc[-1], 4) if 'POC' in self.df.columns else None,
            'OB_Imbalance': round(self.df['OB_Imbalance'].iloc[-1], 4) if 'OB_Imbalance' in self.df.columns else None,
            'Monopoly_MF': round(self.df['Monopoly_MF'].iloc[-1], 0) if 'Monopoly_MF' in self.df.columns else None,
            'SMFI': round(self.df['SMFI'].iloc[-1], 0) if 'SMFI' in self.df.columns else None,
        }

        return ScannerResult(
            symbol=symbol,
            timestamp=str(pd.Timestamp.now()),
            composite_score=composite,
            signal=signal,
            action=action,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            smart_money_score=sm_score,
            mtf_confluence=mtf_confluence,
            adx_value=self.df['ADX_14'].iloc[-1],
            adx_trend=str(adx_trend.value) if hasattr(adx_trend, 'value') else str(adx_trend),
            indicators=indicators,
            smart_money_signals=smart_money,
            mtf_alignment=mtf_data
        )

    def _calculate_all_indicators(self):
        """Calculate all 65+ indicators"""
        df = self.engine.df

        # TREND (10)
        df['SMA_20'] = self.engine.sma(df['close'], 20)
        df['SMA_50'] = self.engine.sma(df['close'], 50)
        df['SMA_200'] = self.engine.sma(df['close'], 200)
        df['EMA_12'] = self.engine.ema(df['close'], 12)
        df['EMA_26'] = self.engine.ema(df['close'], 26)
        df['WMA_20'] = self.engine.wma(df['close'], 20)
        df['HMA_20'] = self.engine.hma(df['close'], 20)
        df['DEMA_20'] = self.engine.dema(df['close'], 20)
        df['TEMA_20'] = self.engine.tema(df['close'], 20)
        df['KAMA_10'] = self.engine.kama(df['close'], 10)
        df['PSAR'], df['PSAR_Trend'] = self.engine.parabolic_sar()
        df['ATR_14'] = self.engine.atr(14)
        df['Supertrend'], df['Supertrend_Dir'] = self.engine.supertrend()
        df['Tenkan'], df['Kijun'], df['Senkou_A'], df['Senkou_B'], df['Chikou'] = self.engine.ichimoku()

        # MOMENTUM (10)
        df['RSI_14'] = self.engine.rsi(14)
        df['Stoch_K'], df['Stoch_D'] = self.engine.stochastic()
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.engine.macd()
        df['CCI_20'] = self.engine.cci(20)
        df['Williams_R'] = self.engine.williams_r()
        df['Momentum_10'] = self.engine.momentum(10)
        df['ROC_12'] = self.engine.roc(12)
        df['Ultimate_Osc'] = self.engine.ultimate_oscillator()
        df['TRIX_15'] = self.engine.trix(15)
        df['AO'] = self.engine.awesome_oscillator()

        # VOLATILITY (10)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.engine.bollinger_bands()
        df['KC_Upper'], df['KC_Middle'], df['KC_Lower'] = self.engine.keltner_channels()
        df['DC_Upper'], df['DC_Middle'], df['DC_Lower'] = self.engine.donchian_channels()
        df['StdDev_20'] = self.engine.std_dev(20)
        df['Chaikin_Vol'] = self.engine.chaikin_volatility()
        df['Hist_Vol_20'] = self.engine.historical_volatility()
        df['Vol_Index'] = self.engine.volatility_index()
        df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
        df['BB_PercentB'] = safe_divide((df['close'] - df['BB_Lower']), (df['BB_Upper'] - df['BB_Lower']), 0.5)

        # VOLUME (10)
        df['OBV'] = self.engine.obv()
        df['VWAP'] = self.engine.vwap()
        df['AD_Line'] = self.engine.ad_line()
        df['CMF_20'] = self.engine.chaikin_money_flow(20)
        df['MFI_14'] = self.engine.money_flow_index(14)
        df['Force_Index'] = self.engine.force_index()
        df['EOM_14'] = self.engine.ease_of_movement(14)
        df['NVI'] = self.engine.negative_volume_index()
        df['PVI'] = self.engine.positive_volume_index()

        # ADX/DMI (5)
        df['ADX_14'], df['DI_Plus'], df['DI_Minus'] = self.engine.adx_dmi(14)
        df['DMI_Spread'] = df['DI_Plus'] - df['DI_Minus']
        df['ADX_Slope'] = df['ADX_14'].diff(5)

        # SMART MONEY (10)
        df['Bullish_OB'], df['Bearish_OB'] = self.engine.detect_order_blocks()
        df['Bullish_FVG'], df['Bearish_FVG'] = self.engine.detect_fvg()
        df['Sweep_High'], df['Sweep_Low'] = self.engine.detect_liquidity_sweeps()
        df['BOS_Bullish'], df['BOS_Bearish'], df['CHoCH'] = self.engine.detect_structure_breaks()
        df['SMFI'] = self.engine.smart_money_flow_index()
        df['POC'] = self.engine.volume_profile()
        df['OB_Imbalance'] = self.engine.order_book_imbalance()
        df['Monopoly_MF'] = self.engine.monopoly_money_flow()
        df['Bull_Trap'], df['Bear_Trap'] = self.engine.detect_smart_money_traps()
        df['Displacement'] = self.engine.detect_displacement()

        # MTF (5)
        df['MTF_Volume'] = self.engine.mtf_volume_confluence()

        self.engine.df = df
        self.scorer.df = df
        self.df = df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: CORE DECISION / FUSION ENGINES (from stock_scanner_core-10.py)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Stock Scanner - Core Strategy Modules
======================================
Cleaned/filtered version. Duplicate & low-value (<60%) classes removed:
  - AITradingWatchdog, QuantumTitanAI, QuantumFusionDecisionMatrix,
    OmegaMarketIntelligence, HyperExecutionAI, MarketMasterPatternEngine,
    DynamicInstitutionalZoneEngine, AdaptiveSupportResistanceEngine,
    ExpiryZoneFusionEngine
    (all were re-skins of already-kept engines with no new logic)

MERGED: InstitutionalTradeManagementEngine + AdaptiveExecutionSupervisor +
  InstitutionalTradeGuardian + QuantumExecutionController -> TradeManagementEngine
  (all 4 scored a trade near-identically; kept each duplicate check once and
  folded in the unique piece from each: exit_rules, protection gate,
  position sizing, profit booking)

MERGED: UltraPriceActionEngine + InstitutionalPriceActionMatrix +
  EliteCandlePatternLaboratory (gap/trap/rejection logic only) -> PriceActionEngine

MERGED: SupportResistanceFusionEngine + InstitutionalSupportResistanceAI
  -> SupportResistanceEngine

NEW: ExpiryZoneEngine - options-Greeks-based expiry zones (gamma walls/flip,
  vanna, charm, dealer positioning, max pain). Genuinely distinct domain,
  not covered by SupportResistanceEngine.

NEW: HighAccuracyFusionEngine - master decision layer. Combines every engine's
  output, enforces minimum 80% blended confidence AND requires validator +
  execution guardian + price action to all independently pass (multi-layer
  protection - no single high score can override a failed layer). Also
  surfaces a WATCHLIST tier (65-79% blended) so the engine never goes silent,
  plus per-layer diagnostics explaining exactly why a trade was blocked.

NOTE: This module references external dependencies that must exist in your project:
  - chart_ai.analyze(...)
  - trade_decision_engine.evaluate(...)
  - logger (standard logging.Logger instance)
Make sure these are imported/available wherever this module is used.
"""

import time
import asyncio
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Trade Validator - approves/blocks a trade based on multi-factor scoring
# ---------------------------------------------------------------------------
class AutoTradeValidator:
    def __init__(self):
        self.minimum_score = 80

    async def validate(
        self,
        signal: Dict[str, Any],
        chart: Dict[str, Any],
        flow: Dict[str, Any],
        option_chain: Dict[str, Any],
        greeks: Dict[str, Any],
        risk: Dict[str, Any],
        broker_state: Dict[str, Any],
    ):
        report = {
            "approved": False,
            "score": 0,
            "reasons": [],
            "blocks": [],
        }

        score = 0

        if chart.get("breakout"):
            score += 15
        if chart.get("volume_spike"):
            score += 15
        if flow.get("smart_money"):
            score += 15
        if flow.get("long_buildup"):
            score += 10
        if option_chain.get("put_writing"):
            score += 10
        if option_chain.get("call_unwinding"):
            score += 10
        if greeks.get("delta", 0) > 0.60:
            score += 5
        if greeks.get("gamma", 0) > 0:
            score += 5
        if signal.get("trend_strength", 0) > 70:
            score += 5
        if signal.get("multi_timeframe_confirmation"):
            score += 10

        if risk.get("reward_ratio", 0) < 2:
            report["blocks"].append("Poor Risk Reward")
        if risk.get("sl_distance", 100) > 3:
            report["blocks"].append("Stoploss Too Wide")
        if broker_state.get("connected") is False:
            report["blocks"].append("Broker Offline")

        report["score"] = score

        if score >= self.minimum_score and len(report["blocks"]) == 0:
            report["approved"] = True
            report["reasons"].append("Trade Passed AI Validation")

        return report


trade_validator = AutoTradeValidator()


# ---------------------------------------------------------------------------
# 2. Institutional Flow Engine - detects smart money / operator activity
# ---------------------------------------------------------------------------
class InstitutionalFlowEngine:
    def __init__(self):
        self.history: Dict[str, List[Dict[str, Any]]] = {}

    async def analyze(
        self,
        symbol: str,
        tick: Dict[str, Any],
        option_chain: Dict[str, Any],
        greeks: Dict[str, Any],
        market_depth: Dict[str, Any],
    ):
        if symbol not in self.history:
            self.history[symbol] = []

        self.history[symbol].append(
            {
                "ts": time.time(),
                "price": tick["ltp"],
                "volume": tick["volume"],
                "oi": tick.get("oi", 0),
            }
        )
        self.history[symbol] = self.history[symbol][-500:]

        result = {
            "symbol": symbol,
            "smart_money": False,
            "long_buildup": False,
            "short_buildup": False,
            "short_covering": False,
            "long_unwinding": False,
            "possible_operator_activity": False,
            "confidence": 0,
            "score": 0,
            "signals": [],
        }

        bid = market_depth.get("total_bid_qty", 0)
        ask = market_depth.get("total_ask_qty", 0)

        if bid > ask * 2:
            result["score"] += 15
            result["signals"].append("Bid Dominance")
        if ask > bid * 2:
            result["score"] -= 15
            result["signals"].append("Ask Dominance")

        delta = greeks.get("delta", 0)
        if delta > 0.60:
            result["score"] += 10

        if option_chain.get("put_writing"):
            result["score"] += 15
            result["long_buildup"] = True
        if option_chain.get("call_writing"):
            result["score"] -= 15
            result["short_buildup"] = True
        if option_chain.get("call_unwinding"):
            result["score"] += 10
            result["short_covering"] = True
        if option_chain.get("put_unwinding"):
            result["score"] -= 10
            result["long_unwinding"] = True

        if abs(result["score"]) >= 35:
            result["smart_money"] = True
        if abs(result["score"]) >= 50:
            result["possible_operator_activity"] = True

        result["confidence"] = min(abs(result["score"]), 100)

        return result


institutional_flow_engine = InstitutionalFlowEngine()


# ---------------------------------------------------------------------------
# 3. Autonomous Trading Supervisor - main monitoring loop (orders, positions,
#    brokers, system health, symbol scanning)
# ---------------------------------------------------------------------------
class AutonomousTradingSupervisor:
    def __init__(self):
        self.running = False
        self.scan_interval = 1.0
        self.symbol_states: Dict[str, Dict[str, Any]] = {}
        self.global_alerts: List[Dict[str, Any]] = []

    async def start(self, state_provider):
        self.running = True
        while self.running:
            try:
                state = await state_provider()
                await self.scan_all_symbols(state)
                await self.scan_orders(state)
                await self.scan_positions(state)
                await self.scan_brokers(state)
                await self.scan_system(state)
                await asyncio.sleep(self.scan_interval)
            except Exception as exc:
                logger.exception(exc)
                await asyncio.sleep(1)

    def stop(self):
        self.running = False

    async def scan_all_symbols(self, state):
        symbols = state.get("symbols", [])

        for symbol in symbols:
            candles = symbol.get("candles", [])
            volume = symbol.get("volume", [])
            oi = symbol.get("oi", [])
            option_chain = symbol.get("option_chain", {})
            trades = symbol.get("trades", [])

            chart = await chart_ai.analyze(
                symbol=symbol["symbol"],
                candles=candles,
                volume=volume,
                oi=oi,
            )

            alerts = await market_surveillance.monitor(
                symbol=symbol["symbol"],
                candles=candles,
                volume=volume,
                oi=oi,
                trades=trades,
                option_chain=option_chain,
            )

            decision = await trade_decision_engine.evaluate(
                market=symbol,
                chart=chart,
                option_chain=option_chain,
                greeks=symbol.get("greeks", {}),
                oi=symbol.get("oi_analysis", {}),
                volume=symbol.get("volume_analysis", {}),
            )

            self.symbol_states[symbol["symbol"]] = {
                "chart": chart,
                "alerts": alerts,
                "decision": decision,
                "updated": time.time(),
            }

    async def scan_orders(self, state):
        for order in state.get("orders", []):
            if order.get("status") == "PENDING":
                age = time.time() - order.get("timestamp", time.time())
                if age > 15:
                    self.global_alerts.append(
                        {
                            "type": "STUCK_ORDER",
                            "order_id": order["order_id"],
                            "severity": "HIGH",
                        }
                    )

    async def scan_positions(self, state):
        for position in state.get("positions", []):
            if not position.get("stop_loss"):
                self.global_alerts.append(
                    {
                        "type": "NO_STOPLOSS",
                        "symbol": position["symbol"],
                        "severity": "CRITICAL",
                    }
                )

    async def scan_brokers(self, state):
        for broker in state.get("brokers", []):
            if not broker.get("connected"):
                self.global_alerts.append(
                    {
                        "type": "BROKER_DISCONNECTED",
                        "broker": broker["name"],
                        "severity": "CRITICAL",
                    }
                )

    async def scan_system(self, state):
        system = state.get("system", {})
        if system.get("cpu", 0) > 90:
            self.global_alerts.append({"type": "HIGH_CPU", "severity": "HIGH"})
        if system.get("memory", 0) > 90:
            self.global_alerts.append({"type": "HIGH_MEMORY", "severity": "HIGH"})


trading_supervisor = AutonomousTradingSupervisor()


# ---------------------------------------------------------------------------
# 4. Market Surveillance Engine - generates real-time alerts
# ---------------------------------------------------------------------------
class MarketSurveillanceEngine:
    def __init__(self):
        self.volume_spike_factor = 2.5
        self.delivery_factor = 1.50
        self.oi_factor = 1.15
        self.price_move_factor = 0.015

    async def monitor(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        volume: List[int],
        oi: List[int],
        trades: List[Dict[str, Any]],
        option_chain: Dict[str, Any],
    ):
        alerts = []

        if len(candles) < 25:
            return alerts

        last = candles[-1]
        prev = candles[-2]

        avg_volume = sum(volume[-20:]) / 20
        if volume[-1] >= avg_volume * self.volume_spike_factor:
            alerts.append(
                {"type": "VOLUME_SPIKE", "severity": "HIGH", "symbol": symbol, "value": volume[-1]}
            )

        move = abs(last["close"] - prev["close"]) / prev["close"]
        if move >= self.price_move_factor:
            alerts.append(
                {"type": "FAST_PRICE_MOVE", "severity": "HIGH", "change": round(move * 100, 2)}
            )

        if oi[-1] > oi[-2] * self.oi_factor:
            alerts.append({"type": "OI_BUILDUP", "severity": "MEDIUM"})

        if option_chain.get("max_pain_changed"):
            alerts.append({"type": "MAX_PAIN_SHIFT", "severity": "MEDIUM"})

        if option_chain.get("pcr", 1) > 1.4:
            alerts.append({"type": "PCR_BULLISH", "severity": "INFO"})

        if option_chain.get("pcr", 1) < 0.6:
            alerts.append({"type": "PCR_BEARISH", "severity": "INFO"})

        if len(trades):
            buy_qty = sum(x["qty"] for x in trades if x["side"] == "BUY")
            sell_qty = sum(x["qty"] for x in trades if x["side"] == "SELL")

            if buy_qty > sell_qty * 2:
                alerts.append({"type": "BUYING_PRESSURE", "severity": "HIGH"})
            if sell_qty > buy_qty * 2:
                alerts.append({"type": "SELLING_PRESSURE", "severity": "HIGH"})

        return alerts


market_surveillance = MarketSurveillanceEngine()


# ---------------------------------------------------------------------------
# 5. Signal Generator - weighted multi-factor BUY/SELL/WAIT signal
# ---------------------------------------------------------------------------
class Gemini2026TradingEngine:
    def __init__(self):
        self.weights = {
            "trend": 15,
            "volume": 15,
            "oi": 10,
            "delivery": 10,
            "vwap": 10,
            "ema": 10,
            "option_chain": 15,
            "greeks": 5,
            "market_breadth": 5,
            "institutional_flow": 5,
        }

    async def generate_signal(
        self,
        market: Dict[str, Any],
        chart: Dict[str, Any],
        option_chain: Dict[str, Any],
        greeks: Dict[str, Any],
        flow: Dict[str, Any],
        breadth: Dict[str, Any],
    ):
        score = 0
        factors = []

        if chart.get("trend") == "BULLISH":
            score += self.weights["trend"]
            factors.append("Bullish Trend")
        if chart.get("volume_spike"):
            score += self.weights["volume"]
            factors.append("Volume Expansion")
        if market.get("oi_buildup"):
            score += self.weights["oi"]
            factors.append("OI Build-up")
        if market.get("delivery_spike"):
            score += self.weights["delivery"]
            factors.append("Delivery Spike")
        if market.get("above_vwap"):
            score += self.weights["vwap"]
            factors.append("VWAP Support")
        if market.get("ema_alignment"):
            score += self.weights["ema"]
            factors.append("EMA Alignment")
        if option_chain.get("bullish_bias"):
            score += self.weights["option_chain"]
            factors.append("Bullish Option Chain")
        if greeks.get("delta", 0) >= 0.60:
            score += self.weights["greeks"]
        if breadth.get("advance_decline_ratio", 1) > 1.5:
            score += self.weights["market_breadth"]
        if flow.get("smart_money"):
            score += self.weights["institutional_flow"]
            factors.append("Institutional Activity")

        if score >= 90:
            signal = "STRONG_BUY"
        elif score >= 75:
            signal = "BUY"
        elif score <= 20:
            signal = "STRONG_SELL"
        elif score <= 35:
            signal = "SELL"
        else:
            signal = "WAIT"

        return {
            "signal": signal,
            "score": score,
            "confidence": min(score, 100),
            "factors": factors,
            "risk": ("LOW" if score >= 80 else "MEDIUM" if score >= 60 else "HIGH"),
        }


gemini_2026_engine = Gemini2026TradingEngine()


# ---------------------------------------------------------------------------
# 6. Multi-Timeframe Confluence Engine
# ---------------------------------------------------------------------------
class MultiTimeframeConfluenceEngine:
    def __init__(self):
        self.timeframes = [
            "1m", "3m", "5m", "10m", "15m", "30m", "45m",
            "60m", "75m", "120m", "180m", "240m", "1d", "1w",
        ]

    async def analyze(self, symbol: str, timeframe_data: Dict[str, Dict[str, Any]]):
        score = 0
        aligned = 0
        report = {
            "symbol": symbol,
            "overall_trend": "NEUTRAL",
            "confidence": 0,
            "alignment": {},
            "entry_quality": "LOW",
            "institutional_bias": "UNKNOWN",
            "recommendation": "WAIT",
        }

        for tf in self.timeframes:
            data = timeframe_data.get(tf)
            if not data:
                continue

            bullish = (
                data.get("ema20") > data.get("ema50") > data.get("ema200")
                and data.get("price") > data.get("vwap")
                and data.get("rsi", 50) > 55
            )
            bearish = (
                data.get("ema20") < data.get("ema50") < data.get("ema200")
                and data.get("price") < data.get("vwap")
                and data.get("rsi", 50) < 45
            )

            if bullish:
                score += 8
                aligned += 1
                report["alignment"][tf] = "BULLISH"
            elif bearish:
                score -= 8
                aligned += 1
                report["alignment"][tf] = "BEARISH"
            else:
                report["alignment"][tf] = "SIDEWAYS"

        if score >= 60:
            report["overall_trend"] = "STRONG_BULLISH"
            report["recommendation"] = "BUY"
        elif score >= 30:
            report["overall_trend"] = "BULLISH"
            report["recommendation"] = "BUY_ON_DIP"
        elif score <= -60:
            report["overall_trend"] = "STRONG_BEARISH"
            report["recommendation"] = "SELL"
        elif score <= -30:
            report["overall_trend"] = "BEARISH"
            report["recommendation"] = "SELL_ON_RISE"

        report["confidence"] = min(abs(score), 100)

        if aligned >= 10:
            report["entry_quality"] = "A+"
        elif aligned >= 7:
            report["entry_quality"] = "A"
        elif aligned >= 5:
            report["entry_quality"] = "B"
        else:
            report["entry_quality"] = "C"

        report["institutional_bias"] = (
            "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
        )

        return report


multi_timeframe_engine = MultiTimeframeConfluenceEngine()


# ---------------------------------------------------------------------------
# 7. Execution Guardian - final pre-trade execution safety checks
# ---------------------------------------------------------------------------
class AIExecutionGuardian:
    def __init__(self):
        self.min_confidence = 85
        self.max_spread_percent = 0.35
        self.max_slippage_percent = 0.20

    async def validate_execution(
        self,
        signal: Dict[str, Any],
        mtf: Dict[str, Any],
        chart: Dict[str, Any],
        flow: Dict[str, Any],
        option_chain: Dict[str, Any],
        orderbook: Dict[str, Any],
        broker: Dict[str, Any],
        position_size: float,
    ):
        result = {
            "allow_trade": False,
            "execution_score": 0,
            "execution_type": "WAIT",
            "warnings": [],
            "blocks": [],
            "checks": {},
        }

        score = 0

        if signal.get("confidence", 0) >= self.min_confidence:
            score += 15
        if mtf.get("entry_quality") == "A+":
            score += 15
        if chart.get("volume_spike"):
            score += 10
        if chart.get("breakout"):
            score += 15
        if flow.get("smart_money"):
            score += 15
        if option_chain.get("bullish_bias"):
            score += 10

        spread = orderbook.get("spread_percent", 100)
        if spread <= self.max_spread_percent:
            score += 10
        else:
            result["blocks"].append("HIGH_SPREAD")

        slippage = broker.get("expected_slippage", 100)
        if slippage <= self.max_slippage_percent:
            score += 10
        else:
            result["blocks"].append("HIGH_SLIPPAGE")

        if broker.get("connected"):
            score += 10
        else:
            result["blocks"].append("BROKER_OFFLINE")

        result["execution_score"] = score

        if score >= 90 and not result["blocks"]:
            result["allow_trade"] = True
            result["execution_type"] = "MARKET"
        elif score >= 80 and not result["blocks"]:
            result["allow_trade"] = True
            result["execution_type"] = "LIMIT"

        result["checks"] = {
            "signal": signal.get("confidence"),
            "mtf": mtf.get("confidence"),
            "spread": spread,
            "slippage": slippage,
            "position_size": position_size,
        }

        return result


execution_guardian = AIExecutionGuardian()


# ---------------------------------------------------------------------------
# 8. Price Action Engine (MERGED) - combines UltraPriceActionEngine +
#    InstitutionalPriceActionMatrix into one engine. Both originals duplicated
#    BOS/CHOCH/FVG/order-block checks; this version keeps each check ONCE and
#    adds the unique pieces from each source (candlestick/chart patterns from
#    one, liquidity-sweep/smart-money/macro from the other).
# ---------------------------------------------------------------------------
class PriceActionEngine:
    def __init__(self):
        self.minimum_confidence = 96

        self.candlestick_patterns = {
            "bullish_engulfing", "bearish_engulfing", "hammer", "inverted_hammer",
            "shooting_star", "hanging_man", "doji", "dragonfly_doji", "gravestone_doji",
            "morning_star", "evening_star", "three_white_soldiers", "three_black_crows",
            "piercing_pattern", "dark_cloud_cover", "inside_bar", "outside_bar",
            "marubozu", "spinning_top", "tweezer_top", "tweezer_bottom", "harami",
            "belt_hold", "kicker", "abandoned_baby",
        }

        self.chart_patterns = {
            "double_top", "double_bottom", "head_shoulder", "inverse_head_shoulder",
            "cup_handle", "bull_flag", "bear_flag", "ascending_triangle",
            "descending_triangle", "symmetrical_triangle", "broadening",
            "channel_up", "channel_down", "rectangle", "falling_wedge", "rising_wedge",
        }

        self.price_action_rules = {
            "breakout", "breakdown", "retest", "fake_breakout", "liquidity_grab",
            "bos", "choch", "order_block", "breaker_block", "mitigation_block",
            "fvg", "ifvg", "premium_discount", "equal_high", "equal_low",
            "swing_high", "swing_low", "trendline_break", "range_expansion", "compression",
        }

    async def analyze(
        self,
        symbol: str,
        candles: Dict[str, Any],
        structure: Dict[str, Any],
        liquidity: Dict[str, Any],
        smart_money: Dict[str, Any],
        order_blocks: Dict[str, Any],
        fair_value_gap: Dict[str, Any],
        volume: Dict[str, Any],
        orderflow: Dict[str, Any],
        option_chain: Dict[str, Any],
        news: Dict[str, Any],
        macro: Dict[str, Any],
    ):
        score = 0
        confirmations = []

        # --- candlestick patterns ---
        if candles.get("bullish_engulfing"):
            score += 12
            confirmations.append("BULLISH_ENGULFING")
        if candles.get("morning_star"):
            score += 10
            confirmations.append("MORNING_STAR")
        if candles.get("three_white_soldiers"):
            score += 12
            confirmations.append("THREE_WHITE_SOLDIERS")
        if candles.get("hammer"):
            score += 8
            confirmations.append("HAMMER")

        # --- structure (BOS/CHOCH) - checked once ---
        if structure.get("bos"):
            score += 12
            confirmations.append("BOS")
        if structure.get("choch"):
            score += 10
            confirmations.append("CHOCH")

        # --- liquidity sweeps ---
        if liquidity.get("equal_high_sweep"):
            score += 10
            confirmations.append("EQH_SWEEP")
        if liquidity.get("equal_low_sweep"):
            score += 10
            confirmations.append("EQL_SWEEP")

        # --- smart money / order blocks / FVG - checked once ---
        if smart_money.get("institutional_entry"):
            score += 15
            confirmations.append("SMART_MONEY")
        if order_blocks.get("bullish_ob"):
            score += 10
            confirmations.append("ORDER_BLOCK")
        if order_blocks.get("mitigation_complete"):
            score += 8
            confirmations.append("MITIGATION_COMPLETE")
        if fair_value_gap.get("bullish_fvg"):
            score += 8
            confirmations.append("FVG")

        # --- volume / orderflow ---
        if volume.get("institutional_volume") or volume.get("climax_volume"):
            score += 12
            confirmations.append("INSTITUTIONAL_VOLUME")
        if orderflow.get("buy_pressure"):
            score += 10
            confirmations.append("BUY_PRESSURE")

        # --- option chain confluence ---
        if option_chain.get("put_writing"):
            score += 8
        if option_chain.get("call_unwinding"):
            score += 5

        # --- external context ---
        if news.get("positive"):
            score += 5
        if macro.get("market_trend") == "BULLISH":
            score += 4

        # --- gap analysis (from EliteCandlePatternLaboratory) ---
        if candles.get("gap_up"):
            score += 6
            confirmations.append("GAP_UP")
        if candles.get("gap_down"):
            score += 6
            confirmations.append("GAP_DOWN")

        # --- rejection / trap detection (from EliteCandlePatternLaboratory) ---
        # These SUBTRACT from score - a fake breakout or trap invalidates
        # otherwise-bullish confirmations rather than just being ignored.
        rejections = []
        if candles.get("fake_breakout"):
            rejections.append("FAKE_BREAKOUT")
            score -= 15
        if candles.get("bull_trap"):
            rejections.append("BULL_TRAP")
            score -= 12
        if candles.get("bear_trap"):
            rejections.append("BEAR_TRAP")
            score -= 12

        score = max(0, score)
        confidence = min(score, 100)

        if confidence >= self.minimum_confidence:
            action = "A_PLUS_ENTRY"
        elif confidence >= 85:
            action = "HIGH_PROBABILITY_ENTRY"
        elif confidence >= 70:
            action = "WATCH"
        else:
            action = "IGNORE"

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "score": score,
            "confirmations": confirmations,
            "rejections": rejections,
            "institutional_grade": confidence >= self.minimum_confidence,
        }


price_action_engine = PriceActionEngine()


# ---------------------------------------------------------------------------
# 9. High Accuracy Fusion Engine - final master decision layer.
#    Combines every kept engine's output and enforces:
#      - minimum 80% blended confidence
#      - multi-layer protection: ALL of {validator, execution guardian} must
#        report zero blocks, regardless of how high the score is
#    This is the single entry point an autonomous trader should call.
# ---------------------------------------------------------------------------
class HighAccuracyFusionEngine:
    def __init__(self):
        self.minimum_accuracy = 80
        self.watchlist_accuracy = 65  # below this, nothing worth tracking at all

    async def decide(
        self,
        symbol: str,
        signal: Dict[str, Any],          # output of gemini_2026_engine.generate_signal
        mtf: Dict[str, Any],             # output of multi_timeframe_engine.analyze
        flow: Dict[str, Any],            # output of institutional_flow_engine.analyze
        price_action: Dict[str, Any],    # output of price_action_engine.analyze
        validator_report: Dict[str, Any],   # output of trade_validator.validate
        execution_report: Dict[str, Any],   # output of execution_guardian.validate_execution
    ):
        result = {
            "symbol": symbol,
            "final_decision": "WAIT",
            "blended_confidence": 0,
            "layers_passed": [],
            "layers_failed": [],
            "diagnostics": [],   # explains WHY a layer failed, and by how much
        }

        confidences = [
            signal.get("confidence", 0),
            mtf.get("confidence", 0),
            flow.get("confidence", 0),
            price_action.get("confidence", 0),
        ]
        blended = sum(confidences) / len(confidences)
        result["blended_confidence"] = round(blended, 2)

        # Layer 1: blended signal confidence
        if blended >= self.minimum_accuracy:
            result["layers_passed"].append("CONFIDENCE_80_PLUS")
        else:
            gap = round(self.minimum_accuracy - blended, 2)
            result["layers_failed"].append("CONFIDENCE_BELOW_80")
            result["diagnostics"].append(f"Confidence short by {gap} points ({blended}/80)")

        # Layer 2: trade validator (risk/reward, stoploss, broker)
        if validator_report.get("approved"):
            result["layers_passed"].append("VALIDATOR_APPROVED")
        else:
            result["layers_failed"].append("VALIDATOR_BLOCKED")
            for reason in validator_report.get("blocks", []):
                result["diagnostics"].append(f"Validator blocked: {reason}")

        # Layer 3: execution guardian (spread, slippage, broker connectivity)
        if execution_report.get("allow_trade"):
            result["layers_passed"].append("EXECUTION_APPROVED")
        else:
            result["layers_failed"].append("EXECUTION_BLOCKED")
            for reason in execution_report.get("blocks", []):
                result["diagnostics"].append(f"Execution blocked: {reason}")

        # Layer 4: price action confluence must not be flat "IGNORE"
        if price_action.get("action") in ("A_PLUS_ENTRY", "HIGH_PROBABILITY_ENTRY"):
            result["layers_passed"].append("PRICE_ACTION_CONFIRMED")
        else:
            result["layers_failed"].append("PRICE_ACTION_WEAK")
            result["diagnostics"].append(
                f"Price action only reached '{price_action.get('action')}'"
            )

        # All 4 layers must pass for an actual trade decision - no single
        # high score can override a failed layer.
        if len(result["layers_failed"]) == 0:
            result["final_decision"] = (
                "STRONG_BUY" if signal.get("signal") in ("STRONG_BUY", "BUY") else "STRONG_SELL"
            )

        # Not a full pass, but still decent setup -> surface it instead of
        # silently returning WAIT every time. This is informational only;
        # it never auto-trades.
        elif blended >= self.watchlist_accuracy and price_action.get("action") != "IGNORE":
            result["final_decision"] = "WATCHLIST"

        return result


fusion_engine = HighAccuracyFusionEngine()


# ---------------------------------------------------------------------------
# 10. Support / Resistance Engine (MERGED) - combines
#     SupportResistanceFusionEngine + InstitutionalSupportResistanceAI.
#     Both built a weighted list of S/R levels from pivots, fibonacci,
#     market/volume profile, option chain OI, footprint and indicators.
#     This version keeps InstitutionalSupportResistanceAI's better merge
#     algorithm (price-weighted averaging of nearby levels instead of just
#     bucketing) and folds in every unique source from both originals.
#
#     NOTE: DynamicInstitutionalZoneEngine and AdaptiveSupportResistanceEngine
#     (received later) scored below 60% - they are near-total duplicates of
#     this merged engine (same pivot/fib/OI/footprint sources, just re-weighted)
#     and were discarded rather than merged in.
# ---------------------------------------------------------------------------
class SupportResistanceEngine:
    def __init__(self):
        self.merge_percent = 0.0025

    def _merge(self, levels):
        """Price-weighted merge of nearby levels into single zones."""
        levels = sorted(levels, key=lambda x: x["price"])
        merged = []

        for level in levels:
            if not merged:
                merged.append(level)
                continue

            prev = merged[-1]
            if abs(level["price"] - prev["price"]) / max(prev["price"], 1) <= self.merge_percent:
                total = prev["strength"] + level["strength"]
                prev["price"] = (
                    prev["price"] * prev["strength"] + level["price"] * level["strength"]
                ) / total
                prev["strength"] = total
                prev["sources"].extend(level["sources"])
            else:
                merged.append(level)

        return sorted(merged, key=lambda x: x["strength"], reverse=True)

    async def build(
        self,
        candles,
        pivots,
        fibonacci,
        market_profile,
        volume_profile,
        option_chain,
        footprint,
        liquidity,
        indicators,
    ):
        support = []
        resistance = []

        def push(side, price, strength, source):
            obj = {"price": float(price), "strength": strength, "sources": [source]}
            (support if side == "S" else resistance).append(obj)

        for x in pivots.get("major_lows", []):
            push("S", x, 12, "Major Swing Low")
        for x in pivots.get("major_highs", []):
            push("R", x, 12, "Major Swing High")

        for x in pivots.get("equal_lows", []):
            push("S", x, 9, "Equal Low")
        for x in pivots.get("equal_highs", []):
            push("R", x, 9, "Equal High")

        for x in fibonacci.get("supports", []):
            push("S", x, 8, "Fibonacci")
        for x in fibonacci.get("resistances", []):
            push("R", x, 8, "Fibonacci")

        for x in market_profile.get("POC", []):
            push("S", x, 12, "POC")
        for x in market_profile.get("VAH", []):
            push("R", x, 10, "VAH")
        for x in market_profile.get("VAL", []):
            push("S", x, 10, "VAL")

        for x in volume_profile.get("HVN", []):
            push("S", x, 9, "HVN")
        for x in volume_profile.get("LVN", []):
            push("R", x, 9, "LVN")

        for x in option_chain.get("max_put_oi", []):
            push("S", x, 15, "Max Put OI")
        for x in option_chain.get("max_call_oi", []):
            push("R", x, 15, "Max Call OI")
        for x in option_chain.get("put_writing", []):
            push("S", x, 10, "Put Writing")
        for x in option_chain.get("call_writing", []):
            push("R", x, 10, "Call Writing")

        for x in footprint.get("buy_imbalance", []):
            push("S", x, 8, "Buy Imbalance")
        for x in footprint.get("sell_imbalance", []):
            push("R", x, 8, "Sell Imbalance")

        for x in liquidity.get("buy_liquidity", []):
            push("S", x, 9, "Liquidity Pool")
        for x in liquidity.get("sell_liquidity", []):
            push("R", x, 9, "Liquidity Pool")

        if indicators.get("EMA200"):
            push("S", indicators["EMA200"], 8, "EMA200")
        if indicators.get("VWAP"):
            push("S", indicators["VWAP"], 7, "VWAP")

        support = self._merge(support)
        resistance = self._merge(resistance)

        return {
            "supports": support[:20],
            "resistances": resistance[:20],
            "best_support": support[0] if support else None,
            "best_resistance": resistance[0] if resistance else None,
            "support_strength": support[0]["strength"] if support else 0,
            "resistance_strength": resistance[0]["strength"] if resistance else 0,
        }


support_resistance_engine = SupportResistanceEngine()


# ---------------------------------------------------------------------------
# 11. Expiry Zone Engine - options-expiry-specific zones (gamma walls, gamma
#     flip, vanna/charm, dealer positioning, max pain). Genuinely distinct
#     from the general SupportResistanceEngine above - this is built purely
#     from options Greeks/dealer-positioning data, not price/volume levels.
# ---------------------------------------------------------------------------
class ExpiryZoneEngine:
    def __init__(self):
        self.elite_strength = 30
        self.institutional_strength = 24

    async def build_expiry_zones(
        self,
        spot: float,
        expiry: str,
        option_chain: Dict[str, Any],
        gamma: Dict[str, Any],
        delta: Dict[str, Any],
        vanna: Dict[str, Any],
        charm: Dict[str, Any],
        dealer: Dict[str, Any],
        maxpain: Dict[str, Any],
        pcr: Dict[str, Any],
        market_profile: Dict[str, Any],
        volume_profile: Dict[str, Any],
        footprint: Dict[str, Any],
        liquidity: Dict[str, Any],
        orderflow: Dict[str, Any],
        institutional: Dict[str, Any],
    ):
        zones = []

        def add(price, side, strength, source):
            zones.append({"price": float(price), "side": side, "strength": strength, "source": source})

        for x in option_chain.get("max_call_oi", []):
            add(x, "R", 30, "MAX_CALL_OI")
        for x in option_chain.get("max_put_oi", []):
            add(x, "S", 30, "MAX_PUT_OI")
        for x in option_chain.get("call_writing", []):
            add(x, "R", 24, "CALL_WRITING")
        for x in option_chain.get("put_writing", []):
            add(x, "S", 24, "PUT_WRITING")
        for x in option_chain.get("call_unwinding", []):
            add(x, "S", 18, "CALL_UNWIND")
        for x in option_chain.get("put_unwinding", []):
            add(x, "R", 18, "PUT_UNWIND")

        for x in gamma.get("walls", []):
            add(x, "SR", 26, "GAMMA_WALL")
        for x in gamma.get("flip", []):
            add(x, "SR", 30, "GAMMA_FLIP")

        for x in delta.get("hedge_levels", []):
            add(x, "SR", 18, "DELTA_HEDGE")
        for x in vanna.get("levels", []):
            add(x, "SR", 18, "VANNA")
        for x in charm.get("levels", []):
            add(x, "SR", 18, "CHARM")

        for x in dealer.get("long_gamma", []):
            add(x, "S", 24, "LONG_GAMMA")
        for x in dealer.get("short_gamma", []):
            add(x, "R", 24, "SHORT_GAMMA")

        for x in maxpain.get("levels", []):
            add(x, "SR", 32, "MAX_PAIN")

        for x in pcr.get("bullish_levels", []):
            add(x, "S", 18, "PCR")
        for x in pcr.get("bearish_levels", []):
            add(x, "R", 18, "PCR")

        for x in volume_profile.get("HVN", []):
            add(x, "SR", 20, "HVN")
        for x in market_profile.get("POC", []):
            add(x, "SR", 26, "POC")

        for x in footprint.get("buy_imbalance", []):
            add(x, "S", 16, "BUY_IMBALANCE")
        for x in footprint.get("sell_imbalance", []):
            add(x, "R", 16, "SELL_IMBALANCE")

        for x in liquidity.get("buy_side", []):
            add(x, "S", 18, "BUY_LIQUIDITY")
        for x in liquidity.get("sell_side", []):
            add(x, "R", 18, "SELL_LIQUIDITY")

        for x in orderflow.get("institutional_buy", []):
            add(x, "S", 22, "INSTITUTIONAL_BUY")
        for x in orderflow.get("institutional_sell", []):
            add(x, "R", 22, "INSTITUTIONAL_SELL")

        for x in institutional.get("accumulation", []):
            add(x, "S", 24, "ACCUMULATION")
        for x in institutional.get("distribution", []):
            add(x, "R", 24, "DISTRIBUTION")

        zones.sort(key=lambda x: x["strength"], reverse=True)

        return {
            "expiry": expiry,
            "spot": spot,
            "expiry_zones": zones,
            "elite_expiry_zones": [z for z in zones if z["strength"] >= self.elite_strength],
            "institutional_zones": [z for z in zones if z["strength"] >= self.institutional_strength],
        }


expiry_zone_engine = ExpiryZoneEngine()


# ---------------------------------------------------------------------------
# 12. Trade Management Engine (MERGED) - combines InstitutionalTradeManagementEngine
#     + AdaptiveExecutionSupervisor + InstitutionalTradeGuardian +
#     QuantumExecutionController. All four scored a trade near-identically and
#     produced stop-loss/targets/trailing; each duplicate scoring check is
#     kept ONCE here. What's new vs. everything kept so far: this is the first
#     engine that actually manages a LIVE trade (dynamic stop, multi-target,
#     trailing, position sizing, partial profit booking) rather than just
#     approving/scoring entry.
#
#     Unique pieces folded in from each original:
#       - exit_rules (structured exit conditions)        <- InstitutionalTradeManagementEngine
#       - protection (spread/slippage/drawdown/news gate) <- AdaptiveExecutionSupervisor
#       - position_size (equity-based risk sizing)         <- InstitutionalTradeGuardian
#       - profit_booking (partial exits per target)         <- QuantumExecutionController
#
#     NOTE: ExpiryZoneFusionEngine (received in the same batch) scored below
#     60% - it was a near-total duplicate of the already-kept ExpiryZoneEngine
#     (same gamma wall/max pain/OI sources) and was discarded.
# ---------------------------------------------------------------------------
class TradeManagementEngine:
    def __init__(self):
        self.entry_score = 97
        self.exit_score = 45
        self.min_reward_ratio = 2.5
        self.max_position_risk_pct = 0.01   # risk 1% of equity per trade
        self.max_daily_loss_pct = 0.03
        self.max_spread = 0.12
        self.max_slippage = 0.08
        self.max_iv_spike = 12

    async def manage(
        self,
        symbol: str,
        market: Dict[str, Any],
        trend: Dict[str, Any],
        structure: Dict[str, Any],
        support: Dict[str, Any],
        resistance: Dict[str, Any],
        option_chain: Dict[str, Any],
        gamma: Dict[str, Any],
        greeks: Dict[str, Any],
        dealer: Dict[str, Any],
        orderflow: Dict[str, Any],
        footprint: Dict[str, Any],
        liquidity: Dict[str, Any],
        volume: Dict[str, Any],
        volatility: Dict[str, Any],
        news: Dict[str, Any],
        account: Dict[str, Any],
        position: Dict[str, Any],
    ):
        score = 0
        confirmations = []
        blocks = []

        # --- entry scoring (deduped from all 4 originals) ---
        if trend.get("mtf_alignment") or trend.get("alignment"):
            score += 12
            confirmations.append("MTF")
        if trend.get("trend_strength", 0) >= 80 or trend.get("strength", 0) >= 85:
            score += 8
        if structure.get("bos"):
            score += 8
            confirmations.append("BOS")
        if structure.get("choch"):
            score += 8
            confirmations.append("CHOCH")
        if structure.get("retest_confirmed"):
            score += 8
        if support.get("institutional_zone"):
            score += 12
        if resistance.get("reward_ratio", 0) >= self.min_reward_ratio or \
                resistance.get("reward_distance", 0) >= self.min_reward_ratio:
            score += 10
        if option_chain.get("put_writing"):
            score += 10
        if option_chain.get("call_unwinding"):
            score += 8
        if gamma.get("gamma_squeeze"):
            score += 15
        if gamma.get("gamma_flip"):
            score += 10
        if dealer.get("long_gamma") or dealer.get("hedging_buy"):
            score += 8
        if greeks.get("delta", 0) > 0.70:
            score += 6
        if greeks.get("gamma", 0) > 0:
            score += 6
        if greeks.get("vanna", 0) > 0:
            score += 5
        if greeks.get("charm", 0) > 0:
            score += 5
        if orderflow.get("institutional_buy"):
            score += 10
        if footprint.get("buy_absorption"):
            score += 8
        if liquidity.get("sell_side_taken") or liquidity.get("sell_side_swept"):
            score += 8
        if volume.get("relative_volume", 1) >= 2:
            score += 8
        if volatility.get("controlled") or volatility.get("healthy"):
            score += 5
        if news.get("positive"):
            score += 4

        # --- protection gate (from AdaptiveExecutionSupervisor + QuantumExecutionController) ---
        if market.get("spread", 0) > self.max_spread:
            blocks.append("HIGH_SPREAD")
        if market.get("slippage", 0) > self.max_slippage:
            blocks.append("HIGH_SLIPPAGE")
        if account.get("drawdown", 0) > self.max_daily_loss_pct * 100 or \
                account.get("daily_loss_pct", 0) >= self.max_daily_loss_pct:
            blocks.append("DAILY_LOSS_LIMIT")
        if news.get("high_impact"):
            blocks.append("NEWS_SPIKE")
        if volatility.get("iv_change", 0) > self.max_iv_spike:
            blocks.append("IV_SPIKE")
        if account.get("cooldown"):
            blocks.append("COOLDOWN")

        entry = market["price"]
        stop = max(support["price"], entry - market["atr"] * 1.2)
        risk = max(entry - stop, 0.01)

        targets = {
            "T1": entry + risk * 1.5,
            "T2": entry + risk * 2.5,
            "T3": entry + risk * 4.0,
            "FINAL": resistance["price"],
        }

        # --- position sizing (from InstitutionalTradeGuardian) ---
        position_size = min(
            account.get("equity", 0) * self.max_position_risk_pct / risk,
            account.get("max_lot_size", float("inf")),
        )

        trailing = {
            "enable": True,
            "breakeven_after_rr": 1.0,
            "atr_trailing": 0.80,
            "lock_profit_every_rr": 0.50,
            "follow_structure": True,
            "follow_vwap": True,
            "follow_ema20": True,
        }

        # --- partial profit booking (from QuantumExecutionController) ---
        profit_booking = [
            {"target": "T1", "exit_percent": 25},
            {"target": "T2", "exit_percent": 25},
            {"target": "T3", "exit_percent": 25},
            {"target": "FINAL", "exit_percent": 25},
        ]

        # --- structured exit conditions (from InstitutionalTradeManagementEngine) ---
        exit_rules = {
            "exit_on_gamma_flip": True,
            "exit_on_structure_break": True,
            "exit_on_orderflow_reversal": True,
            "exit_on_liquidity_reversal": True,
            "exit_on_trailing_hit": True,
            "emergency_exit_on_news": True,
        }

        scale_rules = {
            "allow_scale_in": score >= 99 and len(blocks) == 0,
            "only_on_retest": True,
            "only_if_sl_at_be": True,
            "max_scale_entries": 2,
        }

        signal = "WAIT"
        if score >= self.entry_score and not blocks:
            signal = "EXECUTE"
        elif score <= self.exit_score:
            signal = "EXIT"

        return {
            "symbol": symbol,
            "signal": signal,
            "score": score,
            "confirmations": confirmations,
            "blocks": blocks,
            "entry": entry,
            "stop_loss": stop,
            "targets": targets,
            "position_size": position_size,
            "trailing": trailing,
            "profit_booking": profit_booking,
            "exit_rules": exit_rules,
            "scale_in": scale_rules,
            "paper_trade_ready": score >= 95,
            "high_lot_mode": score >= 99,
        }


trade_management_engine = TradeManagementEngine()


# =============================================================================
# TRISHUL - Order Panel Engines
# =============================================================================
# Separate project: the "Trishul" order panel has 8 pages (2x Expiry, 2x
# Scalping, 2x Index Intraday, 2x Stock Intraday), each with its own
# entry/exit/target logic. Kept separate from the general scanner engines
# above since risk profile & timeframe differ per category.
# =============================================================================


# ---------------------------------------------------------------------------
# T1. Scalper Decision Engine (MERGED, Scalping category) - combines
#     InstitutionalScalperDecisionEngine + InstitutionalTradeFilterEngine.
#     NOTE: InstitutionalTradeFilterEngine was pasted 3 times identically in
#     the batch received - treated as one class.
#
#     Discarded as <60% duplicates: QuantumScalpingExecutionMatrix,
#     SmartMoneyScalperConfluenceEngine, ScalperEntryConfirmationEngine
#     (all re-score the same buy/sell factors with no new logic; ICT-style
#     confirmations already live in PriceActionEngine).
#
#     Unique pieces folded in:
#       - separate BUY_CE / BUY_PE scoring with independent SL/targets/trailing
#         <- InstitutionalScalperDecisionEngine
#       - session/time gating (first-5-min block, major-news block, spread/
#         slippage pre-check BEFORE scoring even runs) <- InstitutionalTradeFilterEngine
# ---------------------------------------------------------------------------
class ScalperDecisionEngine:
    def __init__(self):
        self.entry_score = 98
        self.max_spread = 0.08
        self.max_slippage = 0.05

    async def evaluate(
        self,
        market: Dict[str, Any],
        session: Dict[str, Any],
        trend: Dict[str, Any],
        structure: Dict[str, Any],
        support: Dict[str, Any],
        resistance: Dict[str, Any],
        option_chain: Dict[str, Any],
        gamma: Dict[str, Any],
        greeks: Dict[str, Any],
        dealer: Dict[str, Any],
        liquidity: Dict[str, Any],
        orderflow: Dict[str, Any],
        footprint: Dict[str, Any],
        volume: Dict[str, Any],
        news: Dict[str, Any],
    ):
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # --- session / risk gate - checked BEFORE scoring, hard block ---
        allow_buy = True
        allow_sell = True
        if session.get("first_5min"):
            allow_buy = False
            allow_sell = False
        if session.get("major_news"):
            allow_buy = False
            allow_sell = False
        if market.get("spread", 999) > self.max_spread:
            allow_buy = False
            allow_sell = False
        if market.get("slippage", 999) > self.max_slippage:
            allow_buy = False
            allow_sell = False

        # --- directional scoring ---
        if trend.get("direction") == "UP":
            buy_score += 12
            buy_reasons.append("TREND")
        if trend.get("direction") == "DOWN":
            sell_score += 12
            sell_reasons.append("TREND")
        if structure.get("bullish_bos"):
            buy_score += 10
        if structure.get("bearish_bos"):
            sell_score += 10
        if structure.get("bullish_choch"):
            buy_score += 8
        if structure.get("bearish_choch"):
            sell_score += 8
        if support.get("institutional_zone"):
            buy_score += 10
        if resistance.get("institutional_zone"):
            sell_score += 10
        if option_chain.get("put_writing"):
            buy_score += 10
        if option_chain.get("call_writing"):
            sell_score += 10
        if option_chain.get("call_unwinding"):
            buy_score += 8
        if option_chain.get("put_unwinding"):
            sell_score += 8
        if gamma.get("gamma_squeeze_up"):
            buy_score += 12
        if gamma.get("gamma_squeeze_down"):
            sell_score += 12
        if dealer.get("buy_hedging"):
            buy_score += 8
        if dealer.get("sell_hedging"):
            sell_score += 8
        if greeks.get("delta", 0) > 0.70:
            buy_score += 6
        if greeks.get("delta", 0) < -0.70:
            sell_score += 6
        if orderflow.get("aggressive_buy"):
            buy_score += 10
        if orderflow.get("aggressive_sell"):
            sell_score += 10
        if footprint.get("buy_absorption"):
            buy_score += 8
        if footprint.get("sell_absorption"):
            sell_score += 8
        if liquidity.get("sell_side_taken"):
            buy_score += 8
        if liquidity.get("buy_side_taken"):
            sell_score += 8
        if volume.get("relative_volume", 1) >= 2:
            buy_score += 6
            sell_score += 6
        if news.get("bullish"):
            buy_score += 4
        if news.get("bearish"):
            sell_score += 4

        entry = market["price"]
        buy_sl = max(support["price"], entry - market["atr"] * 0.80)
        sell_sl = min(resistance["price"], entry + market["atr"] * 0.80)
        buy_risk = max(entry - buy_sl, 0.01)
        sell_risk = max(sell_sl - entry, 0.01)

        buy_targets = {
            "T1": entry + buy_risk * 1.0,
            "T2": entry + buy_risk * 1.5,
            "T3": entry + buy_risk * 2.0,
            "FINAL": resistance["price"],
        }
        sell_targets = {
            "T1": entry - sell_risk * 1.0,
            "T2": entry - sell_risk * 1.5,
            "T3": entry - sell_risk * 2.0,
            "FINAL": support["price"],
        }

        trailing = {
            "enable": True,
            "breakeven_after": 1.0,
            "trail_ema9": True,
            "trail_vwap": True,
            "trail_last_swing": True,
            "lock_profit_every": 0.50,
        }

        signal = "NO_TRADE"
        if allow_buy and buy_score >= self.entry_score:
            signal = "BUY_CE"
        elif allow_sell and sell_score >= self.entry_score:
            signal = "BUY_PE"

        return {
            "signal": signal,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "allow_buy": allow_buy,
            "allow_sell": allow_sell,
            "buy_reasons": buy_reasons,
            "sell_reasons": sell_reasons,
            "entry_price": entry,
            "buy_stop_loss": buy_sl,
            "sell_stop_loss": sell_sl,
            "buy_targets": buy_targets,
            "sell_targets": sell_targets,
            "trailing": trailing,
            "paper_trade_ready": max(buy_score, sell_score) >= 95,
            "high_lot_allowed": max(buy_score, sell_score) >= 99,
        }


scalper_decision_engine = ScalperDecisionEngine()


# ---------------------------------------------------------------------------
# T2. Capital Protection Engine (MERGED) - combines EmergencyMarketRegimeSwitchEngine
#     + SmartCapitalProtectionEngine + NewsReversalProtectionEngine.
#     All 3 handle "something bad is happening, protect the open position" -
#     kept each duplicate score check once. Unique pieces folded in:
#       - FORCE_EXIT as a SEPARATE action from reversal, triggered purely by
#         danger score (no direction needed) <- SmartCapitalProtectionEngine
#       - reversal requires confirmed_reversal + opposite trend alignment +
#         N confirmation candles before flipping direction, not just a score
#         threshold - this is the safer design and is now the ONLY path to
#         a reverse trade <- NewsReversalProtectionEngine
# ---------------------------------------------------------------------------
class CapitalProtectionEngine:
    def __init__(self):
        self.force_exit_score = 90
        self.reverse_score = 110
        self.confirmation_candles = 2
        self.max_execution_ms = 250

    async def evaluate(
        self,
        market: Dict[str, Any],
        news: Dict[str, Any],
        trend: Dict[str, Any],
        structure: Dict[str, Any],
        support: Dict[str, Any],
        resistance: Dict[str, Any],
        option_chain: Dict[str, Any],
        gamma: Dict[str, Any],
        greeks: Dict[str, Any],
        dealer: Dict[str, Any],
        liquidity: Dict[str, Any],
        footprint: Dict[str, Any],
        orderflow: Dict[str, Any],
        cumulative_delta: Dict[str, Any],
        tape: Dict[str, Any],
        volume: Dict[str, Any],
        volatility: Dict[str, Any],
        position: Dict[str, Any],
    ):
        danger_score = 0
        bullish = 0
        bearish = 0
        actions = []

        # --- danger score - triggers FORCE_EXIT regardless of direction ---
        if news.get("breaking") or news.get("high_impact"):
            danger_score += 20
        if news.get("circuit_risk"):
            danger_score += 25
        if volatility.get("atr_spike") or volatility.get("iv_blast") or volatility.get("iv_spike"):
            danger_score += 15
        if volume.get("panic_volume"):
            danger_score += 15
        if tape.get("speed_spike"):
            danger_score += 12

        # --- directional scoring (for the reversal path) ---
        if gamma.get("gamma_flip") or gamma.get("flip_up"):
            bullish += 15
        if gamma.get("flip_down"):
            bearish += 15
        if option_chain.get("fresh_put_writing"):
            bullish += 14
        if option_chain.get("fresh_call_writing"):
            bearish += 14
        if option_chain.get("call_unwinding"):
            bullish += 8
        if option_chain.get("put_unwinding"):
            bearish += 8
        if dealer.get("buy_hedging"):
            bullish += 10
        if dealer.get("sell_hedging"):
            bearish += 10
        if greeks.get("delta_flip_up"):
            bullish += 10
        if greeks.get("delta_flip_down"):
            bearish += 10
        if orderflow.get("institutional_buy"):
            bullish += 14
        if orderflow.get("institutional_sell"):
            bearish += 14
        if footprint.get("buy_absorption"):
            bullish += 10
        if footprint.get("sell_absorption"):
            bearish += 10
        if liquidity.get("sell_side_swept") or liquidity.get("sell_liquidity_taken"):
            bullish += 10
        if liquidity.get("buy_side_swept") or liquidity.get("buy_liquidity_taken"):
            bearish += 10
        if cumulative_delta.get("positive") or cumulative_delta.get("bullish"):
            bullish += 10
        if cumulative_delta.get("negative") or cumulative_delta.get("bearish"):
            bearish += 10
        if trend.get("bullish_alignment"):
            bullish += 10
        if trend.get("bearish_alignment"):
            bearish += 10
        if structure.get("bullish_bos"):
            bullish += 10
        if structure.get("bearish_bos"):
            bearish += 10
        if structure.get("bullish_choch"):
            bullish += 8
        if structure.get("bearish_choch"):
            bearish += 8

        direction = position.get("direction")

        # FORCE_EXIT: pure danger, no direction/confirmation required - fastest path
        if danger_score >= self.force_exit_score and position.get("is_open"):
            actions.append("FORCE_EXIT")

        # Reversal: only after danger + opposite score threshold + explicit
        # confirmation (structure confirmed + opposite trend + N candles).
        # This is deliberately stricter than a bare score check.
        confirmed = (
            structure.get("confirmed_reversal")
            and market.get("confirmation_bars", 0) >= self.confirmation_candles
        )

        if direction == "LONG" and bearish >= self.reverse_score and confirmed and trend.get("opposite_alignment"):
            actions.extend(["CANCEL_ALL_PENDING", "MARKET_EXIT", "BUY_PE", "ENABLE_AUTO_TRAILING"])

        elif direction == "SHORT" and bullish >= self.reverse_score and confirmed and trend.get("opposite_alignment"):
            actions.extend(["CANCEL_ALL_PENDING", "MARKET_EXIT", "BUY_CE", "ENABLE_AUTO_TRAILING"])

        return {
            "actions": actions,
            "bullish_score": bullish,
            "bearish_score": bearish,
            "danger_score": danger_score,
            "reverse_trade": "BUY_CE" in actions or "BUY_PE" in actions,
            "force_exit": "FORCE_EXIT" in actions,
            "new_stop_loss": support["price"] if "BUY_CE" in actions else resistance["price"],
            "partial_profit": [25, 25, 25, 25],
            "dynamic_trailing": {
                "ema9": True, "ema20": True, "vwap": True,
                "supertrend": True, "atr": True, "structure": True, "break_even": True,
            },
            "execution_timeout_ms": self.max_execution_ms,
        }


capital_protection_engine = CapitalProtectionEngine()


# ---------------------------------------------------------------------------
# T3. Market State Engine - regime classifier (kept SEPARATE, not merged).
#     This is a genuinely distinct concept: it doesn't produce a trade
#     action, it classifies the CURRENT market regime (institutional
#     bull/bear, trap, exhaustion, range) so other engines can gate on it
#     via `trade_allowed`.
# ---------------------------------------------------------------------------
class MarketStateEngine:
    def __init__(self):
        self.bull_threshold = 120
        self.bear_threshold = 120
        self.trap_threshold = 35
        self.exhaustion_threshold = 20

    async def detect(
        self,
        trend: Dict[str, Any],
        structure: Dict[str, Any],
        option_chain: Dict[str, Any],
        gamma: Dict[str, Any],
        dealer: Dict[str, Any],
        orderflow: Dict[str, Any],
        footprint: Dict[str, Any],
        cumulative_delta: Dict[str, Any],
        liquidity: Dict[str, Any],
        tape: Dict[str, Any],
        volume: Dict[str, Any],
        volatility: Dict[str, Any],
        market_profile: Dict[str, Any],
        volume_profile: Dict[str, Any],
        breadth: Dict[str, Any],
        sentiment: Dict[str, Any],
        news: Dict[str, Any],
        vix: Dict[str, Any],
    ):
        bull = 0
        bear = 0
        trap = 0
        exhaustion = 0

        if trend.get("ema_alignment_bull"):
            bull += 10
        if trend.get("ema_alignment_bear"):
            bear += 10
        if trend.get("vwap_above"):
            bull += 8
        if trend.get("vwap_below"):
            bear += 8
        if structure.get("bullish_bos"):
            bull += 10
        if structure.get("bearish_bos"):
            bear += 10
        if structure.get("bullish_choch"):
            bull += 8
        if structure.get("bearish_choch"):
            bear += 8
        if option_chain.get("fresh_put_writing"):
            bull += 14
        if option_chain.get("fresh_call_writing"):
            bear += 14
        if option_chain.get("put_short_covering"):
            bull += 8
        if option_chain.get("call_short_covering"):
            bear += 8
        if gamma.get("positive_gamma"):
            bull += 12
        if gamma.get("negative_gamma"):
            bear += 12
        if dealer.get("buy_hedging"):
            bull += 10
        if dealer.get("sell_hedging"):
            bear += 10
        if orderflow.get("institutional_buy"):
            bull += 15
        if orderflow.get("institutional_sell"):
            bear += 15
        if footprint.get("buy_absorption"):
            bull += 10
        if footprint.get("sell_absorption"):
            bear += 10
        if cumulative_delta.get("bullish"):
            bull += 10
        if cumulative_delta.get("bearish"):
            bear += 10
        if liquidity.get("sell_side_taken"):
            bull += 10
        if liquidity.get("buy_side_taken"):
            bear += 10
        if volume.get("climax"):
            exhaustion += 15
        if volatility.get("atr_expansion"):
            bull += 6
            bear += 6
        if tape.get("fake_breakout"):
            trap += 15
        if tape.get("fake_breakdown"):
            trap += 15
        if market_profile.get("trend_day"):
            bull += 6
            bear += 6
        if volume_profile.get("low_volume_node"):
            trap += 6
        if breadth.get("strong_positive"):
            bull += 8
        if breadth.get("strong_negative"):
            bear += 8
        if sentiment.get("risk_on"):
            bull += 5
        if sentiment.get("risk_off"):
            bear += 5
        if news.get("high_impact"):
            trap += 10
        if vix.get("panic"):
            bear += 12

        if bull >= self.bull_threshold:
            state = "INSTITUTIONAL_BULL"
        elif bear >= self.bear_threshold:
            state = "INSTITUTIONAL_BEAR"
        elif trap >= self.trap_threshold:
            state = "TRAP_MARKET"
        elif exhaustion >= self.exhaustion_threshold:
            state = "EXHAUSTION"
        else:
            state = "RANGE"

        return {
            "market_state": state,
            "bull_score": bull,
            "bear_score": bear,
            "trap_score": trap,
            "exhaustion_score": exhaustion,
            "trade_allowed": state not in ("TRAP_MARKET", "EXHAUSTION"),
            "recommended_mode": {
                "scalping": state in ("INSTITUTIONAL_BULL", "INSTITUTIONAL_BEAR"),
                "expiry": state in ("INSTITUTIONAL_BULL", "INSTITUTIONAL_BEAR"),
                "intraday": True,
                "swing": False,
            },
        }


market_state_engine = MarketStateEngine()


# ---------------------------------------------------------------------------
# T4. Expiry Adaptive Position Engine (Expiry category) - kept mostly as-is.
#     Unique piece not seen anywhere else: CapitalMode-based LOT SIZING -
#     confidence tiers map to 1/3/10 lots (MICRO/MEDIUM/AGGRESSIVE). Also
#     folds in the "validate_market" pre-trade gate from the newer
#     DynamicInstitutionalZoneEngine (exchange halt / circuit / major news
#     check) since no engine so far checked those specific conditions.
#
#     Discarded as <60%: InstitutionalGammaBlastEngine (gamma scoring is a
#     duplicate of ExpiryZoneEngine/CapitalProtectionEngine; its
#     blast_probability metric wasn't distinct enough to justify keeping).
# ---------------------------------------------------------------------------
class CapitalMode(Enum):
    MICRO = "MICRO"
    MEDIUM = "MEDIUM"
    AGGRESSIVE = "AGGRESSIVE"


class Signal(Enum):
    WAIT = "WAIT"
    BUY_CE = "BUY_CE"
    BUY_PE = "BUY_PE"


class ExpiryAdaptivePositionEngine:
    def __init__(self):
        self.entry_threshold = 118
        self.strong_threshold = 130
        self.ultra_threshold = 145
        self.max_spread = 0.08
        self.max_slippage = 0.05

    async def validate_market(self, market: Dict[str, Any]):
        """Pre-trade gate - checked BEFORE any scoring runs."""
        if market.get("exchange_halted"):
            return False, "EXCHANGE_HALTED"
        if market.get("major_news"):
            return False, "MAJOR_NEWS"
        if market.get("circuit"):
            return False, "CIRCUIT"
        if market.get("spread", 999) > self.max_spread:
            return False, "HIGH_SPREAD"
        if market.get("slippage", 999) > self.max_slippage:
            return False, "HIGH_SLIPPAGE"
        return True, "OK"

    async def evaluate(
        self,
        instrument: Dict[str, Any],
        market: Dict[str, Any],
        trend: Dict[str, Any],
        structure: Dict[str, Any],
        option_chain: Dict[str, Any],
        gamma: Dict[str, Any],
        greeks: Dict[str, Any],
        orderflow: Dict[str, Any],
        footprint: Dict[str, Any],
        liquidity: Dict[str, Any],
        volume: Dict[str, Any],
        support: Dict[str, Any],
        resistance: Dict[str, Any],
    ):
        valid, reason = await self.validate_market(market)
        if not valid:
            return {"status": "REJECTED", "reason": reason}

        long_score = 0
        short_score = 0

        if instrument.get("expiry_today"):
            long_score += 15
            short_score += 15
        if trend.get("bullish_alignment"):
            long_score += 12
        if trend.get("bearish_alignment"):
            short_score += 12
        if structure.get("bullish_bos"):
            long_score += 12
        if structure.get("bearish_bos"):
            short_score += 12
        if option_chain.get("put_writing"):
            long_score += 15
        if option_chain.get("call_writing"):
            short_score += 15
        if option_chain.get("call_unwinding"):
            long_score += 10
        if option_chain.get("put_unwinding"):
            short_score += 10
        if gamma.get("positive_gamma"):
            long_score += 12
        if gamma.get("negative_gamma"):
            short_score += 12
        if greeks.get("delta", 0) > 0.75:
            long_score += 8
        if greeks.get("delta", 0) < -0.75:
            short_score += 8
        if orderflow.get("institutional_buy"):
            long_score += 12
        if orderflow.get("institutional_sell"):
            short_score += 12
        if footprint.get("buy_absorption"):
            long_score += 10
        if footprint.get("sell_absorption"):
            short_score += 10
        if liquidity.get("sell_side_taken"):
            long_score += 10
        if liquidity.get("buy_side_taken"):
            short_score += 10
        if volume.get("explosive"):
            long_score += 8
            short_score += 8

        signal = Signal.WAIT
        confidence = max(long_score, short_score)

        if long_score >= self.entry_threshold:
            signal = Signal.BUY_CE
        elif short_score >= self.entry_threshold:
            signal = Signal.BUY_PE

        if confidence >= self.ultra_threshold:
            mode = CapitalMode.AGGRESSIVE
            lots = 10
        elif confidence >= self.strong_threshold:
            mode = CapitalMode.MEDIUM
            lots = 3
        else:
            mode = CapitalMode.MICRO
            lots = 1

        entry = market["price"]
        stop_loss = support["price"] if signal == Signal.BUY_CE else resistance["price"]
        risk = abs(entry - stop_loss)

        targets = [
            entry + risk * 1.2 if signal == Signal.BUY_CE else entry - risk * 1.2,
            entry + risk * 2.0 if signal == Signal.BUY_CE else entry - risk * 2.0,
            entry + risk * 3.2 if signal == Signal.BUY_CE else entry - risk * 3.2,
            entry + risk * 5.0 if signal == Signal.BUY_CE else entry - risk * 5.0,
        ]

        trailing = {
            "breakeven": True,
            "trail_ema9": True,
            "trail_supertrend": True,
            "trail_vwap": True,
            "trail_atr": True,
            "lock_profit_rr": 0.50,
        }

        return {
            "status": "SUCCESS",
            "instrument": instrument["name"],
            "signal": signal.value,
            "confidence": confidence,
            "capital_mode": mode.value,
            "recommended_lots": lots,
            "entry": entry,
            "stop_loss": stop_loss,
            "targets": targets,
            "trailing": trailing,
        }


expiry_adaptive_position_engine = ExpiryAdaptivePositionEngine()


# ---------------------------------------------------------------------------
# T5. BTST Scanner Engine (NEW, distinct domain) - Buy Today Sell Tomorrow.
#     Overnight/swing scanning across a stock universe using sector + index
#     confirmation, delivery %, and closing-strength checks. Not covered by
#     any intraday/scalping/expiry engine above - kept as its own category.
# ---------------------------------------------------------------------------
class BTSTScannerEngine:
    def __init__(self):
        self.minimum_score = 145
        self.minimum_rvol = 2.20
        self.minimum_delivery = 45
        self.minimum_sector_strength = 75
        self.minimum_market_strength = 70

    async def calculate_score(self, stock: Dict[str, Any], sector: Dict[str, Any], index: Dict[str, Any]):
        buy = 0
        reasons = []

        if stock["trend"] == "BULLISH":
            buy += 12
            reasons.append("STOCK_TREND")
        if sector["trend"] == "BULLISH":
            buy += 12
            reasons.append("SECTOR_TREND")
        if index["trend"] == "BULLISH":
            buy += 12
            reasons.append("INDEX_TREND")
        if stock.get("mtf_alignment"):
            buy += 12
            reasons.append("MTF")
        if stock.get("bullish_bos"):
            buy += 12
            reasons.append("BOS")
        if stock.get("bullish_choch"):
            buy += 10
            reasons.append("CHOCH")
        if stock.get("order_block"):
            buy += 15
            reasons.append("ORDER_BLOCK")
        if stock.get("breaker_block"):
            buy += 8
        if stock.get("mitigation_block"):
            buy += 8
        if stock.get("fair_value_gap"):
            buy += 12
            reasons.append("FVG")
        if stock.get("liquidity_sweep"):
            buy += 12
            reasons.append("LIQUIDITY")
        if stock.get("relative_volume", 0) >= self.minimum_rvol:
            buy += 12
            reasons.append("RVOL")
        if stock.get("delivery_percent", 0) >= self.minimum_delivery:
            buy += 10
            reasons.append("DELIVERY")
        if stock.get("volume_breakout"):
            buy += 10
        if stock.get("put_writing"):
            buy += 10
        if stock.get("call_unwinding"):
            buy += 8
        if stock.get("positive_gamma"):
            buy += 12
        if stock.get("gamma_flip"):
            buy += 10
        if stock.get("dealer_buy_hedging"):
            buy += 10
        if stock.get("institutional_buying"):
            buy += 15
            reasons.append("INSTITUTION")
        if stock.get("positive_delta"):
            buy += 8
        if stock.get("positive_cumulative_delta"):
            buy += 8
        if stock.get("buy_absorption"):
            buy += 8
        if sector.get("strength", 0) >= self.minimum_sector_strength:
            buy += 10
        if index.get("strength", 0) >= self.minimum_market_strength:
            buy += 10
        if stock.get("closing_near_high"):
            buy += 8
        if stock.get("closing_above_vwap"):
            buy += 8
        if stock.get("above_all_ema"):
            buy += 10

        return buy, reasons

    async def scan(self, stock_list: List[str], data_provider):
        """data_provider needs get_stock_analysis/get_sector_analysis/get_index_analysis."""
        signals = []

        for symbol in stock_list:
            stock = await data_provider.get_stock_analysis(symbol)
            sector = await data_provider.get_sector_analysis(stock["sector"])
            index = await data_provider.get_index_analysis(stock["index"])

            score, reasons = await self.calculate_score(stock, sector, index)
            if score < self.minimum_score:
                continue

            entry = stock["close"]
            stop = min(stock["swing_low"], stock["order_block_low"])
            risk = entry - stop

            signals.append({
                "symbol": symbol,
                "signal": "BTST_BUY",
                "confidence": min(score, 170),
                "institutional_score": score,
                "entry": entry,
                "stop_loss": stop,
                "target_1": entry + risk * 1.5,
                "target_2": entry + risk * 2.5,
                "target_3": entry + risk * 4.0,
                "sector": stock["sector"],
                "reasons": reasons,
            })

        signals.sort(key=lambda x: (x["confidence"], x["institutional_score"]), reverse=True)

        return {
            "scan_status": "SUCCESS",
            "total_candidates": len(signals),
            "signals": signals,
        }


btst_scanner_engine = BTSTScannerEngine()


# ---------------------------------------------------------------------------
# T-SPEC: Expiry 10-Level Entry Pipeline (design spec received, not yet built)
# ---------------------------------------------------------------------------
# The user provided a full 10-level filter spec for Expiry entries:
#   1. Market Filter (expiry day, IV, spread, slippage, session, halt, circuit)
#   2. Trend Filter (EMA9>20>50>200, VWAP, SuperTrend, ADX>25)
#   3. Market Structure (BOS, CHOCH, retest, liquidity sweep)
#   4. Smart Money (order block, breaker, mitigation, FVG, premium/discount)
#   5. Option Chain (fresh put writing, call unwinding, PCR, OI, ATM premium)
#   6. Gamma Engine (positive gamma, blast, flip, dealer long gamma/hedging)
#   7. Orderflow (bid aggression, imbalance, delta, footprint, iceberg)
#   8. Volume (RVOL>2, delivery spike, opening drive, HVN/LVN)
#   9. Candle (strong bullish close, engulfing, three white soldiers, marubozu)
#  10. Execution (entry=break high, stop=swing low/ATR, T1/T2/T3=1R/2R/3R,
#      runner=EMA9 trail) + position sizing tiers + exit/reverse-entry rules
#
# This maps closely onto engines already built above (MarketStateEngine,
# PriceActionEngine, SupportResistanceEngine, ExpiryZoneEngine,
# TradeManagementEngine, ExpiryAdaptivePositionEngine's lot sizing). Once the
# remaining Trishul pages (Index/Stock) are reviewed, this will be built as
# a single ExpiryEntryPipeline class that runs all 10 levels in order and
# short-circuits with a REJECTED status + reason at whichever level fails.
# ---------------------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: UNIFIED SCANNER ENGINE — bridges Part 1 (indicators) into Part 2
# (decision/fusion layer) with a single async entry point.
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedScannerEngine:
    """
    Single entry point for the merged scanner.

    Usage:
        engine = UnifiedScannerEngine()
        result = await engine.run_full_scan(df, symbol="RELIANCE", live_data={...})

    `df` must be an OHLCV pandas DataFrame with columns:
        open, high, low, close, volume

    `live_data` (all optional, sane defaults used if missing) can contain:
        option_chain: dict   e.g. {"put_writing": True, "call_unwinding": False,
                                    "bullish_bias": True}
        greeks: dict         e.g. {"delta": 0.65}
        market_depth: dict   e.g. {"total_bid_qty": 120000, "total_ask_qty": 80000}
        tick: dict           e.g. {"ltp": 2450.5, "volume": 1500000, "oi": 200000}
        market: dict         e.g. {"oi_buildup": True, "delivery_spike": True,
                                    "above_vwap": True, "ema_alignment": True}
        breadth: dict        e.g. {"advance_decline_ratio": 1.8}
        news: dict           e.g. {"positive": True}
        macro: dict          e.g. {"market_trend": "BULLISH"}
        orderbook: dict      e.g. {"spread_percent": 0.15}
        broker: dict         e.g. {"connected": True, "expected_slippage": 0.1}
        position_size: float

    Returns a combined dict:
        {
            "symbol": ...,
            "technical": <ShaktiScanner ScannerResult.to_dict()>,
            "flow": <InstitutionalFlowEngine output>,
            "signal": <Gemini2026TradingEngine output>,
            "price_action": <PriceActionEngine output>,
            "validator": <AutoTradeValidator output>,
            "execution": <AIExecutionGuardian output>,
            "final": <HighAccuracyFusionEngine output>   <- the decision that matters
        }

    NOTE on multi-timeframe data: this bridge currently reuses Shakti's own
    (hardcoded / placeholder) multi_timeframe_alignment() output for the
    `mtf` dict passed into Core's engines, since we only have one timeframe
    of candles here. Replace `_build_mtf_dict()` with a real multi-timeframe
    feed (via multi_timeframe_engine.analyze()) once your data layer can
    supply ema20/50/200 + price + vwap + rsi per timeframe.
    """

    def __init__(self):
        self.shakti = ShaktiScanner()

    async def run_full_scan(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        live_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        live_data = live_data or {}

        option_chain = live_data.get("option_chain", {})
        greeks = live_data.get("greeks", {})
        market_depth = live_data.get("market_depth", {})
        tick = live_data.get("tick", {"ltp": float(df["close"].iloc[-1]),
                                       "volume": int(df["volume"].iloc[-1])})
        market = live_data.get("market", {})
        breadth = live_data.get("breadth", {})
        news = live_data.get("news", {})
        macro = live_data.get("macro", {})
        orderbook = live_data.get("orderbook", {"spread_percent": 0.1})
        broker = live_data.get("broker", {"connected": True, "expected_slippage": 0.05})
        position_size = live_data.get("position_size", 1.0)

        # --- Step 1: technical scan via Shakti (Part 1) ---
        technical: ScannerResult = self.shakti.scan(df, symbol=symbol)
        row = self.shakti.df.iloc[-1]

        # --- Step 2: build dict shapes Core (Part 2) engines expect ---
        chart = self._build_chart_dict(technical, row)
        candles, structure, liquidity, smart_money, order_blocks, fvg, volume, orderflow = \
            self._build_price_action_inputs(row)
        mtf = self._build_mtf_dict(technical)

        # --- Step 3: run Core engines ---
        flow = await institutional_flow_engine.analyze(
            symbol=symbol, tick=tick, option_chain=option_chain,
            greeks=greeks, market_depth=market_depth,
        )

        signal = await gemini_2026_engine.generate_signal(
            market=market, chart=chart, option_chain=option_chain,
            greeks=greeks, flow=flow, breadth=breadth,
        )

        price_action = await price_action_engine.analyze(
            symbol=symbol, candles=candles, structure=structure, liquidity=liquidity,
            smart_money=smart_money, order_blocks=order_blocks, fair_value_gap=fvg,
            volume=volume, orderflow=orderflow, option_chain=option_chain,
            news=news, macro=macro,
        )

        validator_report = await trade_validator.validate(
            signal=signal, chart=chart, flow=flow, option_chain=option_chain,
            greeks=greeks, risk={}, broker_state=broker,
        )

        execution_report = await execution_guardian.validate_execution(
            signal=signal, mtf=mtf, chart=chart, flow=flow,
            option_chain=option_chain, orderbook=orderbook, broker=broker,
            position_size=position_size,
        )

        final = await fusion_engine.decide(
            symbol=symbol, signal=signal, mtf=mtf, flow=flow,
            price_action=price_action, validator_report=validator_report,
            execution_report=execution_report,
        )

        return {
            "symbol": symbol,
            "technical": technical.to_dict(),
            "flow": flow,
            "signal": signal,
            "price_action": price_action,
            "validator": validator_report,
            "execution": execution_report,
            "final": final,
        }

    # -----------------------------------------------------------------
    def _build_chart_dict(self, technical: ScannerResult, row) -> Dict[str, Any]:
        return {
            "trend": "BULLISH" if technical.trend_score >= 60 else
                     "BEARISH" if technical.trend_score <= 40 else "NEUTRAL",
            "volume_spike": bool(row.get("MTF_Volume", 0) and row.get("MTF_Volume", 0) >= 0.5),
            "breakout": bool(row.get("BOS_Bullish", False)),
            "breakdown": bool(row.get("BOS_Bearish", False)),
        }

    def _build_price_action_inputs(self, row):
        candles = {
            "gap_up": False,
            "gap_down": False,
            "fake_breakout": bool(row.get("Bull_Trap", False) or row.get("Bear_Trap", False)),
            "bull_trap": bool(row.get("Bull_Trap", False)),
            "bear_trap": bool(row.get("Bear_Trap", False)),
        }
        structure = {
            "bos": bool(row.get("BOS_Bullish", False)),
            "choch": bool(row.get("CHoCH", False)),
        }
        liquidity = {
            "equal_high_sweep": bool(row.get("Sweep_High", False)),
            "equal_low_sweep": bool(row.get("Sweep_Low", False)),
        }
        smart_money = {
            "institutional_entry": bool(row.get("Displacement", False)),
        }
        order_blocks = {
            "bullish_ob": bool(row.get("Bullish_OB", False)),
            "mitigation_complete": False,
        }
        fvg = {
            "bullish_fvg": bool(row.get("Bullish_FVG", False)),
        }
        volume = {
            "institutional_volume": bool(row.get("MTF_Volume", 0) and row.get("MTF_Volume", 0) >= 0.5),
            "climax_volume": False,
        }
        orderflow = {
            "buy_pressure": bool(row.get("OB_Imbalance", 0) and row.get("OB_Imbalance", 0) > 0),
        }
        return candles, structure, liquidity, smart_money, order_blocks, fvg, volume, orderflow

    def _build_mtf_dict(self, technical: ScannerResult) -> Dict[str, Any]:
        # See class docstring: this reuses Shakti's placeholder MTF data.
        aligned = sum(
            1 for d in technical.mtf_alignment.values() if d["trend"] != "NEUTRAL"
        )
        entry_quality = "A+" if aligned >= 6 else "A" if aligned >= 4 else "B" if aligned >= 2 else "C"
        return {
            "confidence": min(abs(technical.mtf_confluence), 100),
            "entry_quality": entry_quality,
            "overall_trend": "BULLISH" if technical.mtf_confluence > 0 else
                              "BEARISH" if technical.mtf_confluence < 0 else "NEUTRAL",
        }


unified_scanner = UnifiedScannerEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    print("UNIFIED STOCK SCANNER - Loading...")

    np.random.seed(42)
    n = 500
    price = 100.0
    data = []
    for i in range(n):
        change = np.random.normal(0, 0.02)
        o = price * (1 + np.random.normal(0, 0.005))
        c = price * (1 + change)
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.01)))
        l = min(o, c) * (1 - abs(np.random.normal(0, 0.01)))
        v = max(int(np.random.normal(1000000, 300000)), 1000)
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
        price = c

    demo_df = pd.DataFrame(data)

    async def _demo():
        result = await unified_scanner.run_full_scan(
            demo_df,
            symbol="RELIANCE",
            live_data={
                "option_chain": {"put_writing": True, "call_unwinding": True, "bullish_bias": True},
                "greeks": {"delta": 0.65},
                "market_depth": {"total_bid_qty": 150000, "total_ask_qty": 90000},
                "market": {"oi_buildup": True, "delivery_spike": True,
                           "above_vwap": True, "ema_alignment": True},
                "breadth": {"advance_decline_ratio": 1.8},
                "broker": {"connected": True, "expected_slippage": 0.05},
            },
        )

        print("\n" + "=" * 70)
        print(f"TECHNICAL: {result['technical']['signal']} | "
              f"Composite {result['technical']['composite_score']}/100")
        print(f"FLOW: smart_money={result['flow']['smart_money']} "
              f"confidence={result['flow']['confidence']}")
        print(f"AI SIGNAL: {result['signal']['signal']} "
              f"confidence={result['signal']['confidence']}")
        print(f"PRICE ACTION: {result['price_action']['action']} "
              f"confidence={result['price_action']['confidence']}")
        print(f"VALIDATOR APPROVED: {result['validator']['approved']}")
        print(f"EXECUTION ALLOWED: {result['execution']['allow_trade']}")
        print("-" * 70)
        print(f"FINAL DECISION: {result['final']['final_decision']} "
              f"(blended confidence {result['final']['blended_confidence']})")
        if result['final']['diagnostics']:
            print("Diagnostics:")
            for d in result['final']['diagnostics']:
                print(f"  - {d}")
        print("=" * 70)

    asyncio.run(_demo())
