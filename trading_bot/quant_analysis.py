"""Quantitative analysis module for trading signals."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

class QuantAnalysis:
    
    def __init__(self, client=None):
        self.client = client
        self.lookback_zscore = 20
        self.lookback_volatility = 14
        self.zscore_threshold = 1.5
        self.momentum_periods = [5, 10, 20]
        self.htf_cache = {}
        self.htf_cache_time = None
        
    def analyze(self, df: pd.DataFrame, current_price: float, 
                entry_price: float = None, position: str = None,
                balance: float = 100, symbol: str = 'BTCUSDT') -> Dict:
        """Perform quantitative analysis and generate trading signals."""
        result = {
            'signal': 'HOLD',
            'signal_score': 0.0,
            'confidence': 0.5,
            'z_score': 0.0,
            'volatility_regime': 'NORMAL',
            'volatility_percentile': 50,
            'kelly_fraction': 0.0,
            'optimal_position_size': 0.0,
            'liquidation_distance': 100.0,
            'htf_trend': 'NEUTRAL',
            'htf_bias': 0.0,
            'reasoning': '',
            'swing_low': False,
            'swing_high': False,
            'swing_score': 0.0,
            'recent_swing_low': None,
            'recent_swing_high': None,
            'distance_from_swing_low': 0.0,
            'distance_from_swing_high': 0.0,
            'projected_entry': None,
            'entry_type': 'MARKET',
            'limit_offset_pct': 0.0
        }
        
        if len(df) < self.lookback_zscore:
            result['reasoning'] = 'Insufficient data for quant analysis'
            return result
        
        z_score = self._calculate_zscore(df)
        result['z_score'] = z_score
        
        vol_regime, vol_percentile = self._calculate_volatility_regime(df)
        result['volatility_regime'] = vol_regime
        result['volatility_percentile'] = vol_percentile
        
        momentum = self._calculate_momentum(df)
        result['momentum'] = momentum
        
        kelly = self._calculate_kelly(df)
        result['kelly_fraction'] = kelly
        
        optimal_size = self._calculate_optimal_position(kelly, vol_regime, balance)
        result['optimal_position_size'] = optimal_size
        
        if position and entry_price:
            liq_distance = self._calculate_liquidation_distance(
                current_price, entry_price, position
            )
            result['liquidation_distance'] = liq_distance
        
        htf_trend, htf_bias = self._analyze_higher_timeframes(symbol)
        result['htf_trend'] = htf_trend
        result['htf_bias'] = htf_bias
        
        swing_data = self._detect_swings(df, current_price)
        result['swing_low'] = swing_data['is_swing_low']
        result['swing_high'] = swing_data['is_swing_high']
        result['swing_score'] = swing_data['swing_score']
        result['recent_swing_low'] = swing_data['recent_low']
        result['recent_swing_high'] = swing_data['recent_high']
        result['distance_from_swing_low'] = swing_data['dist_from_low']
        result['distance_from_swing_high'] = swing_data['dist_from_high']
        
        signal, score, confidence, reasoning = self._generate_signal(
            z_score, vol_regime, vol_percentile, momentum, htf_trend, htf_bias, swing_data
        )
        
        result['signal'] = signal
        result['signal_score'] = score
        result['confidence'] = confidence
        result['reasoning'] = reasoning
        
        return result
    
    def _calculate_zscore(self, df: pd.DataFrame) -> float:
        closes = df['close'].tail(self.lookback_zscore).values
        mean = np.mean(closes)
        std = np.std(closes)
        
        if std == 0:
            return 0.0
        
        current = closes[-1]
        z_score = (current - mean) / std
        return round(z_score, 3)
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict:
        closes = df['close'].values
        
        momentum = {}
        for period in self.momentum_periods:
            if len(closes) > period:
                pct_change = (closes[-1] - closes[-period]) / closes[-period]
                momentum[f'{period}p'] = round(pct_change * 100, 3)
            else:
                momentum[f'{period}p'] = 0.0
        
        avg_momentum = np.mean(list(momentum.values()))
        momentum['avg'] = round(avg_momentum, 3)
        
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in momentum.values() if isinstance(v, (int, float))]
        if len(set(signs)) == 1 and signs[0] != 0:
            momentum['aligned'] = True
        else:
            momentum['aligned'] = False
        
        return momentum
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> Tuple[str, float]:
        high = df['high'].tail(self.lookback_volatility * 2)
        low = df['low'].tail(self.lookback_volatility * 2)
        close = df['close'].tail(self.lookback_volatility * 2)
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(self.lookback_volatility).mean()
        atr_pct = (atr / close) * 100
        
        current_atr_pct = atr_pct.iloc[-1]
        
        historical_atr = atr_pct.dropna().values
        if len(historical_atr) > 0:
            percentile = (np.sum(historical_atr < current_atr_pct) / len(historical_atr)) * 100
        else:
            percentile = 50
        
        if percentile < 25:
            regime = 'LOW'
        elif percentile < 75:
            regime = 'NORMAL'
        elif percentile < 90:
            regime = 'HIGH'
        else:
            regime = 'EXTREME'
        
        return regime, round(percentile, 1)
    
    def _calculate_kelly(self, df: pd.DataFrame) -> float:
        """Calculate Kelly criterion for position sizing."""
        returns = df['close'].pct_change().tail(50).dropna()
        
        if len(returns) < 10:
            return 0.1
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.1
        
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return 0.1
        
        win_loss_ratio = avg_win / avg_loss
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        kelly = max(0, min(kelly * 0.5, 0.25))
        
        return round(kelly, 3)
    
    def _calculate_optimal_position(self, kelly: float, vol_regime: str, 
                                    balance: float) -> float:
        vol_multiplier = {
            'LOW': 1.2,
            'NORMAL': 1.0,
            'HIGH': 0.6,
            'EXTREME': 0.3
        }
        
        adjusted_kelly = kelly * vol_multiplier.get(vol_regime, 1.0)
        optimal_size = balance * adjusted_kelly
        
        return round(optimal_size, 2)
    
    def _detect_swings(self, df: pd.DataFrame, current_price: float, 
                       lookback: int = 30, swing_window: int = 3) -> Dict:
        """Detect swing highs and lows in price action."""
        result = {
            'is_swing_low': False,
            'is_swing_high': False,
            'swing_score': 0.0,
            'recent_low': None,
            'recent_high': None,
            'dist_from_low': 0.0,
            'dist_from_high': 0.0
        }
        
        if len(df) < lookback + swing_window:
            return result
        
        recent = df.tail(lookback).copy()
        lows = recent['low'].values
        highs = recent['high'].values
        closes = recent['close'].values
        
        period_low = min(lows)
        period_high = max(highs)
        period_low_idx = np.argmin(lows)
        period_high_idx = np.argmax(highs)
        
        swing_lows = []
        for i in range(swing_window, len(lows) - swing_window):
            is_low = True
            for j in range(1, swing_window + 1):
                if lows[i] > lows[i-j] or lows[i] > lows[i+j]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append((i, lows[i]))
        
        if not swing_lows and period_low_idx > 2 and period_low_idx < len(lows) - 2:
            swing_lows.append((period_low_idx, period_low))
        
        swing_highs = []
        for i in range(swing_window, len(highs) - swing_window):
            is_high = True
            for j in range(1, swing_window + 1):
                if highs[i] < highs[i-j] or highs[i] < highs[i+j]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append((i, highs[i]))
        
        if not swing_highs and period_high_idx > 2 and period_high_idx < len(highs) - 2:
            swing_highs.append((period_high_idx, period_high))
        
        if swing_lows:
            result['recent_low'] = swing_lows[-1][1]
            result['dist_from_low'] = ((current_price - result['recent_low']) / result['recent_low']) * 100
        
        if swing_highs:
            result['recent_high'] = swing_highs[-1][1]
            result['dist_from_high'] = ((current_price - result['recent_high']) / result['recent_high']) * 100
        
        last_5_lows = lows[-5:]
        last_5_highs = highs[-5:]
        current_low = lows[-1]
        current_high = highs[-1]
        current_close = closes[-1]
        
        near_low = current_close <= min(last_5_lows) * 1.003
        bouncing = current_close > current_low
        
        if near_low and bouncing:
            result['is_swing_low'] = True
        
        near_high = current_close >= max(last_5_highs) * 0.997
        rejecting = current_close < current_high
        
        if near_high and rejecting:
            result['is_swing_high'] = True
        
        if result['recent_low']:
            dist_from_low_pct = (current_price - result['recent_low']) / result['recent_low']
            if dist_from_low_pct < 0.005:
                result['is_swing_low'] = True
        
        if result['recent_high']:
            dist_from_high_pct = (result['recent_high'] - current_price) / result['recent_high']
            if dist_from_high_pct < 0.005:
                result['is_swing_high'] = True
        
        if result['recent_low'] and result['recent_high']:
            swing_range = result['recent_high'] - result['recent_low']
            if swing_range > 0:
                position_in_range = (current_price - result['recent_low']) / swing_range
                result['swing_score'] = 1 - (2 * position_in_range)
                result['swing_score'] = max(-1, min(1, result['swing_score']))
        
        result['projected_entry'], result['entry_type'], result['limit_offset_pct'] = \
            self._calculate_projected_entry(current_price, result, closes)
        
        return result
    
    def _calculate_projected_entry(self, current_price: float, swing_data: Dict, 
                                   closes: np.ndarray) -> Tuple[float, str, float]:
        """Calculate optimal entry price based on swing levels."""
        recent_low = swing_data.get('recent_low')
        recent_high = swing_data.get('recent_high')
        swing_score = swing_data.get('swing_score', 0)
        is_swing_low = swing_data.get('is_swing_low', False)
        is_swing_high = swing_data.get('is_swing_high', False)
        
        if not recent_low or not recent_high:
            return current_price, 'MARKET', 0.0
        
        swing_range = recent_high - recent_low
        atr_estimate = swing_range * 0.1
        
        if is_swing_low and swing_score > 0.3:
            buffer_pct = 0.001 + (0.002 * (1 - swing_score))
            projected = recent_low * (1 + buffer_pct)
            
            if current_price <= projected * 1.003:
                return current_price, 'MARKET', 0.0
            
            offset = ((current_price - projected) / current_price) * 100
            return round(projected, 2), 'LIMIT', round(offset, 3)
        
        elif is_swing_high and swing_score < -0.3:
            buffer_pct = 0.001 + (0.002 * (1 + swing_score))
            projected = recent_high * (1 - buffer_pct)
            
            if current_price >= projected * 0.997:
                return current_price, 'MARKET', 0.0
            
            offset = ((projected - current_price) / current_price) * 100
            return round(projected, 2), 'LIMIT', round(offset, 3)
        
        elif abs(swing_score) > 0.5:
            if swing_score > 0.5:
                offset_pct = 0.002 * swing_score
                projected = current_price * (1 - offset_pct)
                return round(projected, 2), 'LIMIT', round(offset_pct * 100, 3)
            else:
                offset_pct = 0.002 * abs(swing_score)
                projected = current_price * (1 + offset_pct)
                return round(projected, 2), 'LIMIT', round(offset_pct * 100, 3)
        
        return current_price, 'MARKET', 0.0
    
    def _calculate_liquidation_distance(self, current_price: float, 
                                        entry_price: float, position: str,
                                        leverage: int = 10) -> float:
        liq_threshold = 1 / leverage
        
        if position == 'LONG':
            distance = (current_price - entry_price) / entry_price + liq_threshold
        else:
            distance = (entry_price - current_price) / entry_price + liq_threshold
        
        return round(distance * 100, 2)
    
    def _analyze_higher_timeframes(self, symbol: str = 'BTCUSDT') -> Tuple[str, float]:
        """Analyze higher timeframe trends for confirmation."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        if self.htf_cache_time and (now - self.htf_cache_time).seconds < 900:
            return self.htf_cache.get('trend', 'NEUTRAL'), self.htf_cache.get('bias', 0.0)
        
        if not self.client:
            return 'NEUTRAL', 0.0
        
        try:
            klines_4h = self.client.get_klines(symbol=symbol, interval='4h', limit=50)
            klines_1d = self.client.get_klines(symbol=symbol, interval='1d', limit=30)
            
            df_4h = pd.DataFrame(klines_4h, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df_4h['close'] = pd.to_numeric(df_4h['close'])
            
            df_1d = pd.DataFrame(klines_1d, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df_1d['close'] = pd.to_numeric(df_1d['close'])
            
            ema20_4h = df_4h['close'].ewm(span=20).mean().iloc[-1]
            ema50_4h = df_4h['close'].ewm(span=50).mean().iloc[-1]
            price_4h = df_4h['close'].iloc[-1]
            
            ema20_1d = df_1d['close'].ewm(span=20).mean().iloc[-1]
            ema50_1d = df_1d['close'].ewm(span=50).mean().iloc[-1]
            price_1d = df_1d['close'].iloc[-1]
            
            score_4h = 0
            score_1d = 0
            
            if price_4h > ema20_4h > ema50_4h:
                score_4h = 1.0
            elif price_4h > ema20_4h:
                score_4h = 0.5
            elif price_4h < ema20_4h < ema50_4h:
                score_4h = -1.0
            elif price_4h < ema20_4h:
                score_4h = -0.5
            
            if price_1d > ema20_1d > ema50_1d:
                score_1d = 1.0
            elif price_1d > ema20_1d:
                score_1d = 0.5
            elif price_1d < ema20_1d < ema50_1d:
                score_1d = -1.0
            elif price_1d < ema20_1d:
                score_1d = -0.5
            
            bias = (score_1d * 0.6) + (score_4h * 0.4)
            
            if bias > 0.3:
                trend = 'BULLISH'
            elif bias < -0.3:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
            
            self.htf_cache = {'trend': trend, 'bias': bias}
            self.htf_cache_time = now
            
            return trend, round(bias, 2)
            
        except Exception as e:
            print(f"  MTF analysis error: {e}")
            return 'NEUTRAL', 0.0
    
    def _generate_signal(self, z_score: float, vol_regime: str, 
                        vol_percentile: float, momentum: Dict = None,
                        htf_trend: str = 'NEUTRAL', htf_bias: float = 0.0,
                        swing_data: Dict = None) -> Tuple[str, float, float, str]:
        signal = 'HOLD'
        score = 0.0
        confidence = 0.5
        reasons = []
        
        momentum = momentum or {'avg': 0, 'aligned': False}
        swing_data = swing_data or {'is_swing_low': False, 'is_swing_high': False, 'swing_score': 0}
        avg_mom = momentum.get('avg', 0)
        mom_aligned = momentum.get('aligned', False)
        
        swing_override_htf = False
        if swing_data.get('is_swing_low') and swing_data.get('swing_score', 0) > 0.3:
            signal = 'LONG'
            score = min(swing_data['swing_score'], 0.9)
            confidence = 0.75
            reasons.append(f"🎯 SWING LOW detected (score={swing_data['swing_score']:.2f})")
            if swing_data['swing_score'] > 0.7:
                swing_override_htf = True
                reasons.append("💪 Strong swing - HTF override enabled")
        elif swing_data.get('is_swing_high') and swing_data.get('swing_score', 0) < -0.3:
            signal = 'SHORT'
            score = max(swing_data['swing_score'], -0.9)
            confidence = 0.75
            reasons.append(f"🎯 SWING HIGH detected (score={swing_data['swing_score']:.2f})")
            if swing_data['swing_score'] < -0.7:
                swing_override_htf = True
                reasons.append("💪 Strong swing - HTF override enabled")
        
        elif z_score > self.zscore_threshold:
            signal = 'SHORT'
            score = -min(z_score / 2, 1.0)
            reasons.append(f"Overbought Z={z_score:.2f}")
        elif z_score < -self.zscore_threshold:
            signal = 'LONG'
            score = min(abs(z_score) / 2, 1.0)
            reasons.append(f"Oversold Z={z_score:.2f}")
        
        elif abs(z_score) < 1.0:
            if avg_mom > 0.1:
                signal = 'LONG'
                score = min(avg_mom / 0.5, 0.7)
                reasons.append(f"Trend UP (mom={avg_mom:+.2f}%)")
            elif avg_mom < -0.1:
                signal = 'SHORT'
                score = max(avg_mom / 0.5, -0.7)
                reasons.append(f"Trend DOWN (mom={avg_mom:+.2f}%)")
            else:
                reasons.append(f"Neutral (Z={z_score:.2f}, mom={avg_mom:+.2f}%)")
        else:
            reasons.append(f"Weak signal zone (Z={z_score:.2f})")
        
        if mom_aligned and signal != 'HOLD':
            confidence *= 1.2
            reasons.append("✓ Momentum aligned")
        
        if vol_regime == 'EXTREME':
            confidence *= 0.5
            reasons.append("⚠️ EXTREME vol")
        elif vol_regime == 'HIGH':
            confidence *= 0.7
            reasons.append("High vol")
        elif vol_regime == 'LOW':
            confidence *= 1.1
            reasons.append("Low vol ✓")
        
        if signal != 'HOLD' and htf_trend != 'NEUTRAL':
            if (signal == 'LONG' and htf_trend == 'BULLISH') or \
               (signal == 'SHORT' and htf_trend == 'BEARISH'):
                confidence *= 1.3
                score *= 1.2
                reasons.append(f"✓ HTF {htf_trend} confirms")
            elif (signal == 'LONG' and htf_trend == 'BEARISH') or \
                 (signal == 'SHORT' and htf_trend == 'BULLISH'):
                if swing_override_htf:
                    confidence *= 0.8
                    score *= 0.85
                    reasons.append(f"🔥 HTF {htf_trend} overridden by strong swing")
                elif abs(htf_bias) > 0.5:
                    signal = 'HOLD'
                    score = 0
                    confidence = 0.3
                    reasons.append(f"⚠️ HTF {htf_trend} AGAINST - skipping")
                else:
                    confidence *= 0.6
                    reasons.append(f"⚠️ HTF {htf_trend} conflict")
        elif htf_trend != 'NEUTRAL':
            reasons.append(f"HTF: {htf_trend} ({htf_bias:+.2f})")
        
        confidence = min(confidence, 1.0)
        
        reasoning = " | ".join(reasons)
        
        return signal, round(score, 3), round(confidence, 3), reasoning

def test_quant():
    import sys
    sys.path.insert(0, '.')
    from dotenv import load_dotenv
    load_dotenv('../.env')
    from binance.client import Client
    import os
    
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    klines = client.get_klines(symbol='BTCUSDC', interval='15m', limit=100)
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    quant = QuantAnalysis()
    result = quant.analyze(df, df['close'].iloc[-1], balance=56.40)
    
    print("=" * 60)
    print("QUANT ANALYSIS TEST")
    print("=" * 60)
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['signal_score']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Z-Score: {result['z_score']}")
    print(f"Volatility: {result['volatility_regime']} ({result['volatility_percentile']}th percentile)")
    print(f"Kelly Fraction: {result['kelly_fraction']:.1%}")
    print(f"Optimal Position: ${result['optimal_position_size']:.2f}")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == '__main__':
    test_quant()
