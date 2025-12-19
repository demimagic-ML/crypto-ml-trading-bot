"""
LLM Overlord - Final Decision Maker using DeepSeek R1
======================================================
Takes all signals, analysis, and context into account to make the final trading decision.
Uses DeepSeek R1 for superior math/reasoning at low cost.
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

class LLMOverlord:
    """DeepSeek R1-based final decision maker for trading."""
    
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.model = "deepseek-reasoner"
        self.last_decision = None
        self.last_reasoning = None
        self.call_count = 0
        self.enabled = True
        
        self.min_interval_seconds = 120
        self.last_call_time = None
        
        if not self.api_key:
            print("  [Overlord] Warning: DEEPSEEK_API_KEY not set - Overlord disabled")
            self.enabled = False
        elif not OPENAI_SDK_AVAILABLE:
            print("  [Overlord] Warning: OpenAI SDK not installed - Overlord disabled")
            self.enabled = False
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
    
    def make_decision(self, context: Dict) -> Dict:
        """
        Make final trading decision based on all available signals.
        
        Args:
            context: Dict containing all signals and market data
            
        Returns:
            Dict with decision, confidence, reasoning, and entry suggestion
        """
        if not self.enabled:
            return self._passthrough_decision(context)
        
        now = datetime.now()
        if self.last_call_time:
            elapsed = (now - self.last_call_time).total_seconds()
            if elapsed < self.min_interval_seconds:
                if self.last_decision:
                    cached = self.last_decision.copy()
                    cached['cached'] = True
                    cached['cache_age'] = int(elapsed)
                    return cached
                return self._passthrough_decision(context)
        
        try:
            prompt = self._build_prompt(context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000
            )
            
            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            
            if reasoning:
                print(f"  [Overlord] Reasoning: {reasoning[:100]}...")
            
            if not content:
                print(f"  [Overlord] Empty response")
                return self._passthrough_decision(context)
            
            decision = self._parse_response(content, context)
            
            self.last_decision = decision
            self.last_call_time = now
            self.call_count += 1
            
            return decision
            
        except Exception as e:
            print(f"  [Overlord] Error: {e}")
            return self._passthrough_decision(context)
    
    def _get_system_prompt(self) -> str:
        """System prompt for the DeepSeek R1 Overlord."""
        return """You are the OVERLORD - an elite institutional-grade trading AI and the FINAL decision maker for RocketRat, a BTC futures bot with 10x leverage.

You think like a senior prop firm trader who has passed multiple funded challenges and manages millions. Your edge comes from combining quantitative signals with smart money concepts and flawless risk management.

=== TRADING PHILOSOPHY ===
"Risk management is more important than strategy. No strategy is 100% accurate, but a predefined risk plan can be."

You execute every 5 MINUTES. Think in terms of short-term price action while respecting the macro trend.

=== SMART MONEY CONCEPTS (SMC) ===
You understand how institutions move markets:
1. LIQUIDITY GRABS: Institutions hunt stop-losses before reversing. Swing highs/lows often mark these zones.
2. ORDER BLOCKS: Large institutional orders create strong S/R. Price often returns to these zones.
3. FAIR VALUE GAPS: Price imbalances that institutions revisit before continuing trends.
4. CHANGE OF CHARACTER (ChoCH): A shift in market structure signaling potential reversal.
5. ACCUMULATION/DISTRIBUTION: Wyckoff phases where smart money quietly enters/exits.

=== PROP FIRM RISK RULES ===
With 10x leverage, discipline is everything:
- MAX RISK PER TRADE: Never exceed 1-2% of account
- EMOTIONAL DISCIPLINE: After a loss, reduce size. Never revenge trade.
- RISK:REWARD MINIMUM: Only take trades with 1:2 R:R or better
- DRAWDOWN PROTECTION: If losing, reduce exposure. Preserve capital.
- POSITION SIZING: Consistent sizing prevents account blowups

=== SIGNAL ANALYSIS ===
You receive:
- ML predictions (XGBoost + LightGBM ensemble)
- News sentiment (hourly cache)
- Trading wisdom (Buffett, Soros, Livermore patterns)
- Quant analysis (Z-score, momentum, volatility, Kelly)
- HTF trend (4H + Daily bias)
- Swing detection (local extremes = institutional interest zones)
- Whale activity (on-chain flow)
- CURRENT POSITION & PnL

=== DECISION RULES ===
1. CONFLUENCE IS KING: 3+ signals aligned = HIGH confidence entry
2. CONFLICTING SIGNALS: Default to HOLD. Patience pays.
3. SWING EXTREMES: High priority entries - institutions accumulate here
4. Z-SCORE EXTREMES: >1.5 or <-1.5 = mean reversion opportunity
5. HTF ALIGNMENT: Trade WITH the 4H/Daily trend unless at swing extreme
6. WHALE DIVERGENCE: If whales accumulate while price drops = bullish setup

=== POSITION MANAGEMENT ===
NO POSITION:
- Wait for high-conviction setup (swing + 3+ signal confluence)
- Don't force trades. The best trade is often no trade.

IN PROFIT (>0.5%):
- Trail stop to breakeven minimum
- Consider partial profit if signals weakening
- Let winners run if trend intact

IN LOSS:
- If signals reversed: CUT immediately. Small losses are acceptable.
- If signals still aligned: HOLD. Give the trade room.
- Never add to losers.

REVERSAL LOGIC:
- LONG position + bearish signals = Consider SHORT (close and reverse)
- SHORT position + bullish signals = Consider LONG (close and reverse)
- Only reverse with strong conviction (60%+ confidence)

=== SECRET EDGE ===
1. "The crowd is usually wrong at extremes" - Fade extreme sentiment
2. "Institutions need liquidity" - They push price to stops before reversing
3. "Time your entries at session opens" - London/NY opens bring volatility
4. "Divergences precede reversals" - Price vs momentum divergence is powerful
5. "The trend is your friend until the bend" - Respect HTF until ChoCH confirmed

IMPORTANT: Respond in ENGLISH only.

Output ONLY valid JSON:
{
    "decision": "LONG" | "SHORT" | "HOLD",
    "confidence": 0-100,
    "entry_type": "MARKET" | "LIMIT",
    "limit_price": null or price,
    "reasoning": "brief explanation",
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "key_factors": ["factor1", "factor2", "factor3"]
}"""

    def _build_prompt(self, ctx: Dict) -> str:
        """Build the analysis prompt from context."""
        
        price = ctx.get('current_price', 0) or 0
        position = ctx.get('position', None)
        
        ml = ctx.get('ml', {}) or {}
        news = ctx.get('news', {}) or {}
        wisdom = ctx.get('wisdom', {}) or {}
        quant = ctx.get('quant', {}) or {}
        whale = ctx.get('whale', {}) or {}
        
        ensemble = ctx.get('ensemble', {}) or {}
        
        def safe_float(val, default=0):
            try:
                return float(val) if val is not None else default
            except:
                return default
        
        def safe_str(val, default='N/A'):
            return str(val) if val is not None else default
        
        ml_pred = safe_float(ml.get('predicted_price'), price)
        ml_conf = safe_float(ml.get('confidence'), 0)
        news_score = safe_float(news.get('score'), 0)
        news_conf = safe_float(news.get('confidence'), 0)
        wisdom_score = safe_float(wisdom.get('score'), 0)
        quant_zscore = safe_float(quant.get('z_score'), 0)
        quant_mom5 = safe_float(quant.get('momentum_5p'), 0)
        quant_mom10 = safe_float(quant.get('momentum_10p'), 0)
        quant_mom20 = safe_float(quant.get('momentum_20p'), 0)
        quant_vol_pctl = safe_float(quant.get('volatility_pctl'), 50)
        quant_htf_bias = safe_float(quant.get('htf_bias'), 0)
        quant_swing_score = safe_float(quant.get('swing_score'), 0)
        quant_proj_entry = safe_float(quant.get('projected_entry'), price)
        whale_net_flow = safe_float(whale.get('net_flow'), 0)
        ensemble_strength = safe_float(ensemble.get('strength'), 0)
        ensemble_threshold = safe_float(ensemble.get('threshold'), 0.08)
        
        swing_status = 'None'
        if quant.get('swing_low'):
            swing_status = '🎯 SWING LOW'
        elif quant.get('swing_high'):
            swing_status = '🎯 SWING HIGH'
        
        position_info = ctx.get('position_info', {}) or {}
        entry_price = safe_float(position_info.get('entry_price'), 0)
        pnl_pct = safe_float(position_info.get('pnl_pct'), 0)
        pnl_leveraged = safe_float(position_info.get('pnl_leveraged'), 0)
        quantity = safe_float(position_info.get('quantity'), 0)
        
        position_str = 'NONE (flat)'
        if position:
            position_str = f"{position} @ ${entry_price:,.2f} | PnL: {pnl_pct:+.2f}% ({pnl_leveraged:+.1f}% with 10x) | Qty: {quantity} BTC"
        
        prompt = f"""CURRENT MARKET STATE:
- BTC Price: ${price:,.2f}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Trading Frequency: Every 5 minutes

===== CURRENT POSITION =====
{position_str}

===== SIGNAL ANALYSIS =====

1. ML PREDICTION:
   - Direction: {safe_str(ml.get('direction'), 'HOLD')}
   - Predicted Price: ${ml_pred:,.2f}
   - Confidence: {ml_conf:.0%}

2. NEWS SENTIMENT:
   - Signal: {safe_str(news.get('direction'), 'HOLD')}
   - Score: {news_score:+.2f}
   - Confidence: {news_conf:.0%}
   - Summary: {safe_str(news.get('summary'), 'N/A')[:100]}

3. TRADING WISDOM:
   - Signal: {safe_str(wisdom.get('direction'), 'HOLD')}
   - Score: {wisdom_score:+.2f}
   - Master: {safe_str(wisdom.get('master'), 'N/A')}
   - Insight: {safe_str(wisdom.get('insight'), 'N/A')[:100]}

4. QUANT ANALYSIS:
   - Signal: {safe_str(quant.get('signal'), 'HOLD')}
   - Z-Score: {quant_zscore:+.2f}
   - Momentum (5p/10p/20p): {quant_mom5:+.2f}% / {quant_mom10:+.2f}% / {quant_mom20:+.2f}%
   - Volatility: {safe_str(quant.get('volatility_regime'), 'NORMAL')} ({quant_vol_pctl:.0f}th pctl)
   - HTF Trend: {safe_str(quant.get('htf_trend'), 'NEUTRAL')} (bias: {quant_htf_bias:+.2f})
   - Swing: {swing_status} (score: {quant_swing_score:+.2f})
   - Projected Entry: ${quant_proj_entry:,.2f} ({safe_str(quant.get('entry_type'), 'MARKET')})

5. WHALE ACTIVITY:
   - Sentiment: {safe_str(whale.get('sentiment'), 'NEUTRAL')}
   - Net Flow: {whale_net_flow:+.1f} BTC
   - Analysis: {safe_str(whale.get('reasoning'), 'N/A')[:80]}

===== ENSEMBLE VOTE (WEIGHTED) =====
- Direction: {safe_str(ensemble.get('direction'), 'HOLD')}
- Raw Vote: {ensemble_strength:+.4f}
- Threshold: {ensemble_threshold:.2f}
- Weights: ML=25%, News=15%, Wisdom=25%, Quant=25%, HTF=10%

NOTE: The ensemble vote is calculated using the weights above. 
Your job is to make the FINAL decision - you can agree or override the ensemble.

===== YOUR TASK =====
Analyze all signals above and provide your final trading decision.
Consider risk/reward, signal alignment, and current market conditions.
If swing low/high detected with supporting signals, this is a HIGH PRIORITY entry opportunity.

Respond with JSON only."""

        return prompt
    
    def _parse_response(self, content: str, context: Dict) -> Dict:
        """Parse LLM response into decision dict."""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            decision = json.loads(content.strip())
            
            if 'decision' not in decision:
                decision['decision'] = 'HOLD'
            if 'confidence' not in decision:
                decision['confidence'] = 50
            if 'reasoning' not in decision:
                decision['reasoning'] = 'No reasoning provided'
            if 'entry_type' not in decision:
                decision['entry_type'] = 'MARKET'
            if 'risk_level' not in decision:
                decision['risk_level'] = 'MEDIUM'
            if 'key_factors' not in decision:
                decision['key_factors'] = []
            
            decision['decision'] = decision['decision'].upper()
            if decision['decision'] not in ['LONG', 'SHORT', 'HOLD']:
                decision['decision'] = 'HOLD'
            
            decision['confidence'] = max(0, min(100, int(decision['confidence'])))
            decision['cached'] = False
            decision['model'] = self.model
            
            return decision
            
        except json.JSONDecodeError as e:
            print(f"  [Overlord] JSON parse error: {e}")
            print(f"  [Overlord] Raw response: {content[:200]}...")
            return self._passthrough_decision(context)
    
    def _passthrough_decision(self, context: Dict) -> Dict:
        """Pass through the ensemble decision when Overlord can't decide."""
        ensemble = context.get('ensemble', {})
        return {
            'decision': ensemble.get('direction', 'HOLD'),
            'confidence': int(ensemble.get('strength', 0) * 100),
            'reasoning': 'Passthrough from ensemble (Overlord unavailable)',
            'entry_type': context.get('quant', {}).get('entry_type', 'MARKET'),
            'limit_price': context.get('quant', {}).get('projected_entry'),
            'risk_level': 'MEDIUM',
            'key_factors': ['ensemble_passthrough'],
            'cached': False,
            'passthrough': True,
            'model': 'ensemble'
        }
    
    def get_status(self) -> Dict:
        """Get Overlord status for dashboard."""
        return {
            'enabled': self.enabled,
            'model': self.model,
            'call_count': self.call_count,
            'last_decision': self.last_decision.get('decision') if self.last_decision else None,
            'last_confidence': self.last_decision.get('confidence') if self.last_decision else None,
            'last_reasoning': self.last_decision.get('reasoning') if self.last_decision else None
        }


def test_overlord():
    """Test the LLM Overlord."""
    from dotenv import load_dotenv
    load_dotenv()
    
    overlord = LLMOverlord()
    
    if not overlord.enabled:
        print("Overlord not enabled - check DEEPSEEK_API_KEY")
        return
    
    context = {
        'current_price': 86500,
        'position': None,
        'ml': {
            'direction': 'HOLD',
            'predicted_price': 86450,
            'confidence': 0.40
        },
        'news': {
            'direction': 'LONG',
            'score': 0.32,
            'confidence': 0.65,
            'summary': 'Bitcoin showing strength amid market uncertainty'
        },
        'wisdom': {
            'direction': 'LONG',
            'score': 0.41,
            'master': 'Warren Buffett',
            'insight': 'Market is deeply oversold, blood in the streets moment'
        },
        'quant': {
            'signal': 'LONG',
            'z_score': 1.97,
            'momentum_5p': -0.20,
            'momentum_10p': -0.78,
            'momentum_20p': -1.16,
            'volatility_regime': 'EXTREME',
            'volatility_pctl': 93,
            'htf_trend': 'BEARISH',
            'htf_bias': -1.0,
            'swing_low': True,
            'swing_high': False,
            'swing_score': 1.0,
            'projected_entry': 86200,
            'entry_type': 'LIMIT'
        },
        'whale': {
            'sentiment': 'NEUTRAL',
            'net_flow': 0,
            'reasoning': 'No significant whale activity'
        },
        'ensemble': {
            'direction': 'HOLD',
            'strength': 0.0254,
            'threshold': 0.07
        }
    }
    
    print("Testing LLM Overlord with DeepSeek R1...")
    print("=" * 50)
    
    decision = overlord.make_decision(context)
    
    print(f"\n🤖 OVERLORD DECISION:")
    print(f"   Decision: {decision['decision']}")
    print(f"   Confidence: {decision['confidence']}%")
    print(f"   Entry Type: {decision.get('entry_type', 'MARKET')}")
    print(f"   Risk Level: {decision.get('risk_level', 'MEDIUM')}")
    print(f"   Reasoning: {decision['reasoning']}")
    print(f"   Key Factors: {', '.join(decision.get('key_factors', []))}")


if __name__ == "__main__":
    test_overlord()
