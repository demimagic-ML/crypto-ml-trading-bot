"""Trading wisdom module with legendary trader personalities."""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Tuple

TRADER_PERSONALITIES = {
    "Jesse Livermore": {
        "specialty": "Momentum & Trend Following",
        "triggers": ["strong_trend", "breakout", "momentum_aligned"],
        "style": "aggressive",
        "famous_for": "Made fortunes riding big trends, lost them fighting them",
        "key_principles": [
            "The trend is your friend until it ends",
            "Cut losses quickly, let winners run",
            "Wait for the right moment, then strike decisively"
        ],
        "personality_prompt": """You are Jesse Livermore, the legendary 'Boy Plunger' of Wall Street.
You made and lost several fortunes by reading the tape and riding momentum.
Your style: AGGRESSIVE when trend is clear, PATIENT when waiting for setup.
You believe: 'There is only one side of the market, and it's not the bull side or bear side, but the RIGHT side.'
Focus on: Price action, momentum, volume confirmation, and trend strength.
You HATE: Fighting the trend, averaging down on losers, and premature entries."""
    },
    
    "George Soros": {
        "specialty": "Macro & Reflexivity",
        "triggers": ["news_driven", "sentiment_extreme", "funding_extreme"],
        "style": "contrarian",
        "famous_for": "Broke the Bank of England, master of macro plays",
        "key_principles": [
            "Markets are always biased in one direction",
            "Find the flaw in the prevailing thesis",
            "When you're right, be aggressive"
        ],
        "personality_prompt": """You are George Soros, the master of macro trading and reflexivity theory.
You broke the Bank of England and made billions betting against flawed systems.
Your style: CONTRARIAN at extremes, follow your conviction with SIZE when right.
You believe: 'Markets influence the events they anticipate' - reflexivity means sentiment creates reality.
Focus on: Funding rates, sentiment extremes, crowd positioning, macro factors.
You LOVE: Fading extreme sentiment, identifying reflexive feedback loops.
You HATE: Following the crowd, small positions when conviction is high."""
    },
    
    "Paul Tudor Jones": {
        "specialty": "Technical Analysis & Risk Management",
        "triggers": ["technical_setup", "support_resistance", "risk_reward_favorable"],
        "style": "technical",
        "famous_for": "Predicted 1987 crash, master of risk management",
        "key_principles": [
            "The most important rule is to play great defense",
            "Don't be a hero, don't have an ego",
            "Losers average losers"
        ],
        "personality_prompt": """You are Paul Tudor Jones, legendary macro trader and risk management master.
You predicted the 1987 crash and built a billion-dollar fund with exceptional risk control.
Your style: TECHNICAL precision with STRICT risk management. Never risk more than you can afford.
You believe: '5:1 risk/reward. I'm risking one dollar to make five.'
Focus on: Support/resistance levels, chart patterns, risk/reward ratios, stop placement.
You DEMAND: Clear technical setups, defined risk, favorable reward ratios.
You REFUSE: Trades without clear stops, poor risk/reward, fighting key levels."""
    },
    
    "Warren Buffett": {
        "specialty": "Value & Patience",
        "triggers": ["oversold_extreme", "fear_in_market", "accumulation_zone"],
        "style": "patient",
        "famous_for": "Greatest long-term investor, buys when others are fearful",
        "key_principles": [
            "Be fearful when others are greedy, greedy when others are fearful",
            "Price is what you pay, value is what you get",
            "Our favorite holding period is forever"
        ],
        "personality_prompt": """You are Warren Buffett, the Oracle of Omaha.
While you're known for stocks, your principles of value and patience apply universally.
Your style: PATIENT accumulation during fear, HOLD through volatility.
You believe: 'Be fearful when others are greedy, and greedy when others are fearful.'
Focus on: Extreme oversold conditions, panic selling, crowd fear indicators.
You LOVE: Blood in the streets, extreme RSI readings, capitulation volume.
You HATE: Chasing momentum, FOMO entries, trading without conviction."""
    },
    
    "Ray Dalio": {
        "specialty": "Economic Cycles & Regime Changes",
        "triggers": ["volatility_regime_change", "cycle_shift", "correlation_breakdown"],
        "style": "systematic",
        "famous_for": "Bridgewater's All Weather, understanding economic machines",
        "key_principles": [
            "Understand the economic machine",
            "Diversify well to reduce risk without reducing returns",
            "Study history because everything has happened before"
        ],
        "personality_prompt": """You are Ray Dalio, founder of the world's largest hedge fund.
You view markets as machines with predictable patterns across cycles.
Your style: SYSTEMATIC analysis, REGIME-AWARE positioning.
You believe: 'He who lives by the crystal ball will eat shattered glass.'
Focus on: Volatility regimes, cycle positioning, correlation shifts.
You EXCEL at: Identifying regime changes, adapting to new market conditions.
You WARN: When volatility is extreme, when correlations break, when cycles shift."""
    }
}

class TradingWisdom:
    
    def __init__(self):
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.api_url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.last_analysis = None
        self.last_thinking = None
        self.analysis_cache = {}
        self.selected_trader = None
        self.trader_reason = None
        
    def get_wisdom_signal(self, market_data: Dict) -> Tuple[float, float, str]:
        """Get trading signal from selected trader personality."""
        try:
            self.selected_trader, self.trader_reason = self._select_trader(market_data)
            print(f"  [Wisdom] 🎭 {self.selected_trader} selected: {self.trader_reason}")
            
            prompt = self._build_wisdom_prompt(market_data)
            answer, thinking = self._query_oracle(prompt)
            self.last_thinking = thinking
            return self._parse_wisdom(answer)
        except Exception as e:
            print(f"  [Wisdom] Error: {e}")
            return 0.0, 0.0, "Oracle unavailable"
    
    def _select_trader(self, data: Dict) -> Tuple[str, str]:
        scores = {}
        reasons = {}
        
        rsi = data.get('rsi', 50)
        funding = data.get('funding_rate', 0)
        ls_ratio = data.get('long_short_ratio', 1)
        volume_ratio = data.get('volume_ratio', 1)
        momentum_1h = data.get('momentum_1h', 0)
        momentum_4h = data.get('momentum_4h', 0)
        bb_position = data.get('bb_position', 0.5)
        news_score = abs(data.get('news_score', 0))
        
        livermore_score = 0
        if momentum_1h > 0.003 and momentum_4h > 0.005:
            livermore_score += 4
            reasons["Jesse Livermore"] = f"Bullish momentum ({momentum_1h:.2%}, {momentum_4h:.2%})"
        elif momentum_1h < -0.003 and momentum_4h < -0.005:
            livermore_score += 3
            reasons["Jesse Livermore"] = f"Bearish trend momentum ({momentum_1h:.2%}, {momentum_4h:.2%})"
        if volume_ratio > 1.5 and livermore_score > 0:
            livermore_score += 2
            reasons["Jesse Livermore"] = reasons.get("Jesse Livermore", "") + " + high volume"
        scores["Jesse Livermore"] = livermore_score
        
        soros_score = 0
        if abs(funding) > 0.0005:
            soros_score += 3
            reasons["George Soros"] = f"Extreme funding rate ({funding:.4%})"
        if ls_ratio > 2.0 or ls_ratio < 0.7:
            soros_score += 3
            reasons["George Soros"] = f"Extreme L/S ratio ({ls_ratio:.2f})"
        if news_score > 0.3:
            soros_score += 2
            reasons["George Soros"] = "News-driven market conditions"
        scores["George Soros"] = soros_score
        
        ptj_score = 0
        if 0.2 < bb_position < 0.8:
            ptj_score += 2
            reasons["Paul Tudor Jones"] = f"Technical setup zone (BB: {bb_position:.2f})"
        if 40 < rsi < 60:
            ptj_score += 2
            reasons["Paul Tudor Jones"] = "Neutral RSI - technical focus"
        scores["Paul Tudor Jones"] = ptj_score
        
        buffett_score = 0
        if rsi < 20:
            buffett_score += 6
            reasons["Warren Buffett"] = f"EXTREME fear (RSI: {rsi:.1f}) - 'Be greedy when others are fearful'"
        elif rsi < 30:
            buffett_score += 4
            reasons["Warren Buffett"] = f"Strong fear (RSI: {rsi:.1f})"
        elif rsi < 40:
            buffett_score += 2
            reasons["Warren Buffett"] = f"Oversold conditions (RSI: {rsi:.1f})"
        if bb_position < 0.1:
            buffett_score += 3
            reasons["Warren Buffett"] = reasons.get("Warren Buffett", "Value zone") + f" + BB at {bb_position:.2f}"
        scores["Warren Buffett"] = buffett_score
        
        dalio_score = 0
        if volume_ratio > 2.0 or volume_ratio < 0.3:
            dalio_score += 3
            reasons["Ray Dalio"] = "Regime change indicators (unusual volume)"
        dalio_score += 1
        if "Ray Dalio" not in reasons:
            reasons["Ray Dalio"] = "Systematic analysis needed"
        scores["Ray Dalio"] = dalio_score
        
        selected = max(scores, key=scores.get)
        reason = reasons.get(selected, "General market analysis")
        
        self.trader_scores = scores
        
        return selected, reason
    
    def _build_wisdom_prompt(self, data: Dict) -> str:
        
        trader = TRADER_PERSONALITIES.get(self.selected_trader, TRADER_PERSONALITIES["Ray Dalio"])
        personality_prompt = trader["personality_prompt"]
        specialty = trader["specialty"]
        principles = "\n".join(f"- {p}" for p in trader["key_principles"])
        
        system_prompt = f"""{personality_prompt}

YOUR SPECIALTY: {specialty}
YOUR PRINCIPLES:
{principles}

You are making a BITCOIN TRADING decision. Be true to your trading philosophy.

TECHNICAL INDICATORS:
1. RSI < 30 = Oversold (LONG), RSI > 70 = Overbought (SHORT)
2. MACD crossing above signal = LONG, below = SHORT
3. BB Position < 0.2 = near lower band (LONG), > 0.8 = near upper (SHORT)
4. Momentum: all timeframes green = LONG, all red = SHORT
5. Volume ratio > 1.5 = strong conviction
6. EMA cross positive = bullish, negative = bearish

FUTURES/SENTIMENT DATA (CRITICAL):
1. Funding Rate:
   - > 0.05% = overleveraged longs → contrarian SHORT
   - < -0.05% = overleveraged shorts → contrarian LONG
   - Near 0 = neutral
2. Long/Short Ratio:
   - > 2.0 = crowd very long → potential SHORT (contrarian)
   - < 0.8 = crowd very short → potential LONG (contrarian)
   - 1.0-1.5 = balanced
3. Taker Buy Ratio:
   - > 0.6 = aggressive buying → bullish
   - < 0.4 = aggressive selling → bearish
4. Open Interest Change:
   - OI rising + price rising = strong trend (follow it)
   - OI rising + price falling = accumulating shorts (bearish)
   - OI falling + price rising = short squeeze (temporary)
   - OI falling + price falling = long liquidations (temporary)
5. Top Trader Ratio:
   - Smart money positioning - if different from crowd, follow smart money

CONFLUENCE SCORING:
- 5+ signals agree = STRONG (confidence 0.85+)
- 4 signals agree = MODERATE (confidence 0.7)
- 3 or less = WEAK/NO TRADE (confidence < 0.5)

CONTRARIAN RULES:
- When funding > 0.1% AND L/S ratio > 2.5 → HIGH probability SHORT
- When funding < -0.05% AND L/S ratio < 0.7 → HIGH probability LONG
- Crowd is usually wrong at extremes

BE CONCRETE: Give specific levels and reasoning based on the data."""

        user_prompt = f"""Analyze this Bitcoin market situation:

TECHNICAL INDICATORS:
- Price: ${data.get('price', 0):,.2f}
- 24h Change: {data.get('change_24h', 0):+.2f}%
- RSI (14): {data.get('rsi', 50):.1f}
- MACD: {data.get('macd', 0):.2f} | Signal: {data.get('macd_signal', 0):.2f} | Hist: {data.get('macd_hist', 0):.2f}
- Bollinger Position: {data.get('bb_position', 0.5):.2f} (0=lower, 1=upper)
- ATR %: {data.get('atr_pct', 0):.2f}%
- Volume Ratio: {data.get('volume_ratio', 1):.2f}x average
- EMA Cross: {data.get('ema_cross', 0):+.4f}
- Stochastic K/D: {data.get('stoch_k', 50):.1f} / {data.get('stoch_d', 50):.1f}
- Momentum: 1h={data.get('momentum_1h', 0):+.2%} | 4h={data.get('momentum_4h', 0):+.2%} | 1d={data.get('momentum_1d', 0):+.2%}

FUTURES/SENTIMENT DATA:
- Funding Rate: {data.get('funding_rate', 0):.4%}
- Long/Short Ratio: {data.get('long_short_ratio', 1):.2f} ({data.get('long_account', 50):.0%} long)
- Top Trader Ratio: {data.get('top_trader_ratio', 1):.2f}
- Taker Buy Ratio: {data.get('taker_buy_ratio', 0.5):.2f}
- Open Interest Change: {data.get('oi_change', 0):+.2%}

ML MODEL: ${data.get('ml_prediction', 0):,.2f} ({data.get('ml_change', 0):+.2f}%)
NEWS: {data.get('news_sentiment', 'NEUTRAL')} ({data.get('news_score', 0):+.2f})

POSITION: {data.get('current_position', 'NONE')}

{f'''HISTORICAL PATTERN INSIGHTS (from past trades):
{data.get('historical_patterns', '')}''' if data.get('historical_patterns') else ''}

As {self.selected_trader}, give your trading verdict:
1. SIGNAL: LONG, SHORT, or HOLD
2. SCORE: -1.0 to +1.0 (negative=bearish, positive=bullish)
3. CONFIDENCE: 0.0 to 1.0
4. REASONING: Brief explanation in your character's voice
5. KEY_LEVELS: Entry, stop loss, take profit suggestions
6. TRADE_QUALITY: A+, A, B, C, D, or F grade

Format your response as JSON:
{{"signal": "LONG/SHORT/HOLD", "score": 0.5, "confidence": 0.8, "reasoning": "...", "key_levels": {{"entry": 0, "stop": 0, "target": 0}}, "trade_quality": "B", "risk_reward_ratio": 2.0}}"""

        return system_prompt, user_prompt
    
    def _query_oracle(self, prompts: Tuple[str, str]) -> Tuple[str, str]:
        """Query the LLM oracle for trading wisdom."""
        system_prompt, user_prompt = prompts
        
        if not self.api_key:
            return '{"signal": "HOLD", "score": 0, "confidence": 0.3, "reasoning": "API key not configured"}', ""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen-plus",
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            "parameters": {
                "temperature": 0.3,
                "max_tokens": 1000
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            answer = result.get('output', {}).get('text', '')
            thinking = result.get('output', {}).get('thinking', '')
            
            return answer, thinking
        except Exception as e:
            print(f"  [Wisdom] Oracle query failed: {e}")
            return '{"signal": "HOLD", "score": 0, "confidence": 0.3, "reasoning": "Oracle unavailable"}', ""
    
    def _parse_wisdom(self, answer: str) -> Tuple[float, float, str]:
        """Parse the oracle's response into trading signals."""
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', answer, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(answer)
            
            signal = data.get('signal', 'HOLD').upper()
            score = float(data.get('score', 0))
            confidence = float(data.get('confidence', 0.5))
            reasoning = data.get('reasoning', 'No reasoning provided')
            
            self.last_analysis = {
                'signal': signal,
                'score': score,
                'confidence': confidence,
                'reasoning': reasoning,
                'which_master_speaks': self.selected_trader,
                'trade_quality': data.get('trade_quality', 'C'),
                'key_levels': data.get('key_levels', {}),
                'risk_reward_ratio': data.get('risk_reward_ratio', 1.5),
                'warnings': data.get('warnings', []),
                'analysis': reasoning,
                'trader_specialty': TRADER_PERSONALITIES.get(self.selected_trader, {}).get('specialty', ''),
                'trader_style': TRADER_PERSONALITIES.get(self.selected_trader, {}).get('style', ''),
                'selection_reason': self.trader_reason
            }
            
            return score, confidence, reasoning
            
        except Exception as e:
            print(f"  [Wisdom] Parse error: {e}")
            self.last_analysis = {'signal': 'HOLD', 'score': 0, 'confidence': 0.3}
            return 0.0, 0.3, f"Parse error: {e}"


if __name__ == '__main__':
    wisdom = TradingWisdom()
    test_data = {
        'price': 100000,
        'rsi': 45,
        'macd': 0.5,
        'bb_position': 0.5,
        'momentum_1h': 0.01,
        'momentum_4h': 0.02,
        'funding_rate': 0.0001,
        'long_short_ratio': 1.2
    }
    score, conf, reason = wisdom.get_wisdom_signal(test_data)
    print(f"Score: {score}, Confidence: {conf}, Reason: {reason}")