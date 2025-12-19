"""Learning engine for trade analysis and model optimization."""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import pickle

class TradeMemory:
    
    def __init__(self, storage_path: str = 'learning_data'):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.trades_file = os.path.join(storage_path, 'trade_history.json')
        self.trades: List[Dict] = self._load_trades()
        
    def _load_trades(self) -> List[Dict]:
        if os.path.exists(self.trades_file):
            try:
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_trades(self):
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)
    
    def record_trade_entry(self, trade_id: str, context: Dict) -> None:
        trade = {
            'id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'open',
            'entry': {
                'price': context.get('entry_price'),
                'side': context.get('side'),
                'signals': context.get('signals', {
                    'ml_score': context.get('ml_score'),
                    'ml_confidence': context.get('ml_confidence'),
                    'news_score': context.get('news_score'),
                    'wisdom_score': context.get('wisdom_score'),
                    'wisdom_master': context.get('wisdom_master'),
                    'quant_score': context.get('quant_score'),
                    'htf_bias': context.get('htf_bias'),
                    'final_vote': context.get('final_vote')
                }),
                'market_state': context.get('market_state', {
                    'rsi': context.get('rsi'),
                    'macd': context.get('macd'),
                    'bb_position': context.get('bb_position'),
                    'volume_ratio': context.get('volume_ratio'),
                    'funding_rate': context.get('funding_rate'),
                    'volatility_percentile': context.get('volatility_percentile')
                })
            },
            'exit': None,
            'outcome': None
        }
        self.trades.append(trade)
        self._save_trades()
        print(f"  [Learning] 📝 Trade {trade_id} recorded for learning")
    
    def record_trade_exit(self, trade_id: str, exit_price: float, pnl_percent: float) -> None:
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['status'] = 'closed'
                trade['exit'] = {
                    'price': exit_price,
                    'timestamp': datetime.now().isoformat()
                }
                trade['outcome'] = {
                    'pnl_percent': pnl_percent,
                    'won': pnl_percent > 0,
                    'grade': self._grade_trade(pnl_percent)
                }
                self._save_trades()
                print(f"  [Learning] ✅ Trade {trade_id} closed: {pnl_percent:+.2f}% ({trade['outcome']['grade']})")
                return
    
    def _grade_trade(self, pnl_percent: float) -> str:
        if pnl_percent >= 3.0:
            return 'A+'
        elif pnl_percent >= 1.5:
            return 'A'
        elif pnl_percent >= 0.5:
            return 'B'
        elif pnl_percent >= 0:
            return 'C'
        elif pnl_percent >= -1.0:
            return 'D'
        else:
            return 'F'
    
    def get_closed_trades(self, days: int = 30) -> List[Dict]:
        cutoff = datetime.now() - timedelta(days=days)
        return [
            t for t in self.trades 
            if t['status'] == 'closed' and 
            datetime.fromisoformat(t['timestamp']) > cutoff
        ]
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        trades = self.get_closed_trades(days)
        if not trades:
            return {
                'total': 0, 'win_rate': 0, 'avg_pnl': 0,
                'kelly_fraction': 0.5, 'expectancy': 0, 'max_drawdown': 0,
                'sharpe_ratio': 0, 'profit_factor': 0
            }
        
        wins = sum(1 for t in trades if t['outcome']['won'])
        losses = len(trades) - wins
        pnls = [t['outcome']['pnl_percent'] for t in trades]
        
        win_pnls = [p for p in pnls if p > 0]
        loss_pnls = [abs(p) for p in pnls if p < 0]
        
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        
        win_rate = wins / len(trades) if trades else 0
        loss_rate = 1 - win_rate
        
        if avg_loss > 0 and win_rate > 0:
            b = avg_win / avg_loss
            kelly = (b * win_rate - loss_rate) / b
            kelly_fraction = max(0.1, min(0.8, kelly))
        else:
            kelly_fraction = 0.5
        
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        cumulative = 0
        peak = 0
        max_drawdown = 0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        if np.std(pnls) > 0:
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252 * 3)
        else:
            sharpe = 0
        
        total_wins = sum(win_pnls)
        total_losses = sum(loss_pnls)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate * 100,
            'avg_pnl': np.mean(pnls),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': sum(pnls),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'kelly_fraction': round(kelly_fraction, 3),
            'expectancy': round(expectancy, 4),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999
        }
    
    def get_kelly_position_size(self, base_size: float = 0.6, min_trades: int = 10) -> float:
        """Calculate position size using Kelly criterion."""
        stats = self.get_performance_stats(days=30)
        
        if stats['total'] < min_trades:
            return base_size
        
        kelly = stats['kelly_fraction']
        
        if stats['expectancy'] < 0:
            return 0.2
        
        if stats['max_drawdown'] > 5:
            kelly *= 0.5
        
        adjusted = (kelly * 0.5) + (base_size * 0.5)
        
        return round(max(0.1, min(0.8, adjusted)), 2)

class WeightOptimizer:
    
    def __init__(self, memory: TradeMemory, storage_path: str = 'learning_data'):
        self.memory = memory
        self.storage_path = storage_path
        self.weights_file = os.path.join(storage_path, 'optimized_weights.json')
        
        self.default_weights = {
            'ml': 0.25,
            'news': 0.15,
            'wisdom': 0.25,
            'quant': 0.25,
            'htf': 0.10
        }
        self.current_weights = self._load_weights()
        
    def _load_weights(self) -> Dict[str, float]:
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    return json.load(f)
            except:
                return self.default_weights.copy()
        return self.default_weights.copy()
    
    def _save_weights(self):
        with open(self.weights_file, 'w') as f:
            json.dump(self.current_weights, f, indent=2)
    
    def analyze_pillar_performance(self, days: int = 30) -> Dict[str, Dict]:
        trades = self.memory.get_closed_trades(days)
        if len(trades) < 10:
            return {}
        
        pillar_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for trade in trades:
            signals = trade['entry']['signals']
            side = trade['entry']['side']
            won = trade['outcome']['won']
            
            ml_agreed = (signals.get('ml_score', 0) > 0 and side == 'LONG') or \
                       (signals.get('ml_score', 0) < 0 and side == 'SHORT')
            if ml_agreed:
                pillar_stats['ml']['total'] += 1
                if won:
                    pillar_stats['ml']['correct'] += 1
            
            news_agreed = (signals.get('news_score', 0) > 0.2 and side == 'LONG') or \
                         (signals.get('news_score', 0) < -0.2 and side == 'SHORT')
            if news_agreed:
                pillar_stats['news']['total'] += 1
                if won:
                    pillar_stats['news']['correct'] += 1
            
            wisdom_agreed = (signals.get('wisdom_score', 0) > 0.3 and side == 'LONG') or \
                           (signals.get('wisdom_score', 0) < -0.3 and side == 'SHORT')
            if wisdom_agreed:
                pillar_stats['wisdom']['total'] += 1
                if won:
                    pillar_stats['wisdom']['correct'] += 1
            
            quant_agreed = (signals.get('quant_score', 0) > 0 and side == 'LONG') or \
                          (signals.get('quant_score', 0) < 0 and side == 'SHORT')
            if quant_agreed:
                pillar_stats['quant']['total'] += 1
                if won:
                    pillar_stats['quant']['correct'] += 1
            
            htf_agreed = (signals.get('htf_bias', 0) > 0 and side == 'LONG') or \
                        (signals.get('htf_bias', 0) < 0 and side == 'SHORT')
            if htf_agreed:
                pillar_stats['htf']['total'] += 1
                if won:
                    pillar_stats['htf']['correct'] += 1
        
        result = {}
        for pillar, stats in pillar_stats.items():
            if stats['total'] >= 5:
                result[pillar] = {
                    'accuracy': stats['correct'] / stats['total'] * 100,
                    'samples': stats['total'],
                    'correct': stats['correct']
                }
        
        return result
    
    def optimize_weights(self, min_trades: int = 20) -> Dict[str, float]:
        trades = self.memory.get_closed_trades(days=60)
        if len(trades) < min_trades:
            print(f"  [Learning] Need {min_trades} trades to optimize, have {len(trades)}")
            return self.current_weights
        
        pillar_performance = self.analyze_pillar_performance(days=60)
        if not pillar_performance:
            return self.current_weights
        
        total_accuracy = sum(p['accuracy'] for p in pillar_performance.values())
        
        if total_accuracy > 0:
            new_weights = {}
            for pillar in self.default_weights.keys():
                if pillar in pillar_performance:
                    new_weights[pillar] = pillar_performance[pillar]['accuracy'] / total_accuracy
                else:
                    new_weights[pillar] = self.default_weights[pillar]
            
            total = sum(new_weights.values())
            new_weights = {k: v/total for k, v in new_weights.items()}
            
            for pillar in new_weights:
                self.current_weights[pillar] = (
                    0.7 * self.current_weights[pillar] + 
                    0.3 * new_weights[pillar]
                )
            
            self._save_weights()
            print(f"  [Learning] 🎯 Weights optimized based on {len(trades)} trades")
            
        return self.current_weights

class PromptEnhancer:
    
    def __init__(self, memory: TradeMemory, storage_path: str = 'learning_data'):
        self.memory = memory
        self.storage_path = storage_path
        self.patterns_file = os.path.join(storage_path, 'winning_patterns.json')
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> List[Dict]:
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_patterns(self):
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def extract_patterns(self, min_trades: int = 10) -> None:
        trades = self.memory.get_closed_trades(days=90)
        
        condition_outcomes = defaultdict(list)
        
        for trade in trades:
            market = trade['entry']['market_state']
            signals = trade['entry']['signals']
            
            rsi = market.get('rsi', 50)
            rsi_zone = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            htf = 'bullish' if signals.get('htf_bias', 0) > 0.5 else 'bearish' if signals.get('htf_bias', 0) < -0.5 else 'neutral'
            vol = 'high' if market.get('volatility_percentile', 50) > 70 else 'low' if market.get('volatility_percentile', 50) < 30 else 'normal'
            
            condition_key = f"{rsi_zone}_{htf}_{vol}"
            
            condition_outcomes[condition_key].append({
                'side': trade['entry']['side'],
                'master': signals.get('wisdom_master', 'Unknown'),
                'pnl': trade['outcome']['pnl_percent'],
                'won': trade['outcome']['won']
            })
        
        self.patterns = []
        for condition, outcomes in condition_outcomes.items():
            if len(outcomes) >= 5:
                wins = [o for o in outcomes if o['won']]
                win_rate = len(wins) / len(outcomes) * 100
                avg_pnl = np.mean([o['pnl'] for o in outcomes])
                
                if win_rate >= 60:
                    master_wins = defaultdict(int)
                    for o in wins:
                        master_wins[o['master']] += 1
                    best_master = max(master_wins.items(), key=lambda x: x[1])[0] if master_wins else 'Unknown'
                    
                    self.patterns.append({
                        'condition': condition,
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'samples': len(outcomes),
                        'best_master': best_master,
                        'preferred_side': 'LONG' if sum(1 for o in wins if o['side'] == 'LONG') > len(wins)/2 else 'SHORT'
                    })
        
        self._save_patterns()
        print(f"  [Learning] 📊 Extracted {len(self.patterns)} winning patterns")
    
    def get_context_for_prompt(self, current_market: Dict) -> str:
        if not self.patterns:
            return ""
        
        rsi = current_market.get('rsi', 50)
        rsi_zone = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
        htf_bias = current_market.get('htf_bias', 0)
        htf = 'bullish' if htf_bias > 0.5 else 'bearish' if htf_bias < -0.5 else 'neutral'
        vol_pctl = current_market.get('volatility_percentile', 50)
        vol = 'high' if vol_pctl > 70 else 'low' if vol_pctl < 30 else 'normal'
        
        current_condition = f"{rsi_zone}_{htf}_{vol}"
        
        relevant_patterns = []
        for pattern in self.patterns:
            if pattern['condition'] == current_condition:
                relevant_patterns.append(pattern)
            elif rsi_zone in pattern['condition'] and pattern['win_rate'] >= 65:
                relevant_patterns.append(pattern)
        
        if not relevant_patterns:
            return ""
        
        context_parts = ["[HISTORICAL PATTERN INSIGHTS]"]
        for p in relevant_patterns[:3]:
            context_parts.append(
                f"- In {p['condition'].replace('_', ' ')} conditions: "
                f"{p['win_rate']:.0f}% win rate over {p['samples']} trades, "
                f"avg PnL {p['avg_pnl']:+.1f}%. "
                f"Best performer: {p['best_master']}. Preferred: {p['preferred_side']}."
            )
        
        return "\n".join(context_parts)

class ParameterTuner:
    
    def __init__(self, memory: TradeMemory, storage_path: str = 'learning_data'):
        self.memory = memory
        self.storage_path = storage_path
        self.params_file = os.path.join(storage_path, 'tuned_parameters.json')
        
        self.default_params = {
            'zscore_threshold': 2.0,
            'htf_strong_threshold': 0.7,
            'htf_moderate_threshold': 0.5,
            'htf_penalty_strong': 0.5,
            'htf_penalty_moderate': 0.75,
            'consensus_threshold': 0.12,
            'single_pillar_threshold': 0.25,
            'extreme_rsi_low': 15,
            'extreme_rsi_high': 85
        }
        self.current_params = self._load_params()
        
    def _load_params(self) -> Dict:
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f:
                    return json.load(f)
            except:
                return self.default_params.copy()
        return self.default_params.copy()
    
    def _save_params(self):
        with open(self.params_file, 'w') as f:
            json.dump(self.current_params, f, indent=2)
    
    def analyze_rsi_performance(self) -> Dict:
        trades = self.memory.get_closed_trades(days=60)
        
        rsi_bins = {
            'extreme_oversold': {'range': (0, 15), 'trades': []},
            'oversold': {'range': (15, 30), 'trades': []},
            'neutral': {'range': (30, 70), 'trades': []},
            'overbought': {'range': (70, 85), 'trades': []},
            'extreme_overbought': {'range': (85, 100), 'trades': []}
        }
        
        for trade in trades:
            rsi = trade['entry']['market_state'].get('rsi', 50)
            for zone, data in rsi_bins.items():
                if data['range'][0] <= rsi < data['range'][1]:
                    data['trades'].append(trade)
                    break
        
        results = {}
        for zone, data in rsi_bins.items():
            if len(data['trades']) >= 3:
                wins = sum(1 for t in data['trades'] if t['outcome']['won'])
                results[zone] = {
                    'win_rate': wins / len(data['trades']) * 100,
                    'samples': len(data['trades']),
                    'avg_pnl': np.mean([t['outcome']['pnl_percent'] for t in data['trades']])
                }
        
        return results
    
    def tune_parameters(self, min_trades: int = 30) -> Dict:
        trades = self.memory.get_closed_trades(days=90)
        if len(trades) < min_trades:
            return self.current_params
        
        rsi_perf = self.analyze_rsi_performance()
        
        if 'extreme_oversold' in rsi_perf and rsi_perf['extreme_oversold']['win_rate'] > 60:
            self.current_params['extreme_rsi_low'] = max(10, self.current_params['extreme_rsi_low'] - 2)
        elif 'extreme_oversold' in rsi_perf and rsi_perf['extreme_oversold']['win_rate'] < 40:
            self.current_params['extreme_rsi_low'] = min(20, self.current_params['extreme_rsi_low'] + 2)
        
        if 'extreme_overbought' in rsi_perf and rsi_perf['extreme_overbought']['win_rate'] > 60:
            self.current_params['extreme_rsi_high'] = min(90, self.current_params['extreme_rsi_high'] + 2)
        elif 'extreme_overbought' in rsi_perf and rsi_perf['extreme_overbought']['win_rate'] < 40:
            self.current_params['extreme_rsi_high'] = max(80, self.current_params['extreme_rsi_high'] - 2)
        
        htf_aligned_trades = [t for t in trades if 
                             (t['entry']['side'] == 'LONG' and t['entry']['signals'].get('htf_bias', 0) > 0) or
                             (t['entry']['side'] == 'SHORT' and t['entry']['signals'].get('htf_bias', 0) < 0)]
        htf_against_trades = [t for t in trades if 
                             (t['entry']['side'] == 'LONG' and t['entry']['signals'].get('htf_bias', 0) < -0.5) or
                             (t['entry']['side'] == 'SHORT' and t['entry']['signals'].get('htf_bias', 0) > 0.5)]
        
        if len(htf_aligned_trades) >= 5 and len(htf_against_trades) >= 5:
            aligned_wr = sum(1 for t in htf_aligned_trades if t['outcome']['won']) / len(htf_aligned_trades)
            against_wr = sum(1 for t in htf_against_trades if t['outcome']['won']) / len(htf_against_trades)
            
            if against_wr > 0.55:
                self.current_params['htf_penalty_strong'] = min(0.7, self.current_params['htf_penalty_strong'] + 0.05)
            elif against_wr < 0.35:
                self.current_params['htf_penalty_strong'] = max(0.3, self.current_params['htf_penalty_strong'] - 0.05)
        
        self._save_params()
        print(f"  [Learning] ⚙️ Parameters tuned based on {len(trades)} trades")
        
        return self.current_params

class MLRetrainer:
    
    def __init__(self, memory: TradeMemory, storage_path: str = 'learning_data'):
        self.memory = memory
        self.storage_path = storage_path
        self.training_data_file = os.path.join(storage_path, 'ml_training_data.json')
        self.model_path = os.path.join(os.path.dirname(storage_path), 'models')
        self.training_samples: List[Dict] = self._load_training_data()
        self.min_samples_for_retrain = 50
        self.last_retrain = None
        
    def _load_training_data(self) -> List[Dict]:
        if os.path.exists(self.training_data_file):
            try:
                with open(self.training_data_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_training_data(self):
        with open(self.training_data_file, 'w') as f:
            json.dump(self.training_samples, f, indent=2)
    
    def add_training_sample(self, features: Dict, label: int, weight: float = 1.0) -> None:
        """Add a training sample for ML retraining."""
        sample = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'label': label,
            'weight': weight
        }
        self.training_samples.append(sample)
        self._save_training_data()
    
    def prepare_training_data_from_trades(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        trades = self.memory.get_closed_trades(days=180)
        
        if len(trades) < self.min_samples_for_retrain:
            print(f"  [ML Retrain] Need {self.min_samples_for_retrain} trades, have {len(trades)}")
            return None, None, None
        
        X = []
        y = []
        weights = []
        
        for trade in trades:
            market = trade['entry'].get('market_state', {})
            signals = trade['entry'].get('signals', {})
            outcome = trade.get('outcome', {})
            
            if not market or not outcome:
                continue
            
            features = [
                market.get('rsi', 50) / 100,
                market.get('bb_position', 0.5),
                market.get('volume_ratio', 1) / 3,
                signals.get('htf_bias', 0),
                market.get('funding_rate', 0) * 100,
                market.get('volatility_percentile', 50) / 100,
                signals.get('ml_score', 0),
                signals.get('news_score', 0),
                signals.get('wisdom_score', 0),
                signals.get('quant_score', 0)
            ]
            
            pnl = outcome.get('pnl_percent', 0)
            side = trade['entry'].get('side', 'LONG')
            
            if outcome.get('won', False):
                label = 1 if side == 'LONG' else -1
            else:
                label = -1 if side == 'LONG' else 1
            
            weight = min(abs(pnl) / 2, 2.0)
            
            X.append(features)
            y.append(label)
            weights.append(weight)
        
        if len(X) < self.min_samples_for_retrain:
            return None, None, None
        
        return np.array(X), np.array(y), np.array(weights)
    
    def retrain_xgboost(self) -> bool:
        try:
            import xgboost as xgb
        except ImportError:
            print("  [ML Retrain] XGBoost not available")
            return False
        
        X, y, weights = self.prepare_training_data_from_trades()
        if X is None:
            return False
        
        print(f"  [ML Retrain] Training XGBoost on {len(X)} samples...")
        
        y_class = y + 1
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        model.fit(X, y_class, sample_weight=weights)
        
        retrained_path = os.path.join(self.model_path, 'xgb_retrained.pkl')
        os.makedirs(self.model_path, exist_ok=True)
        with open(retrained_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  [ML Retrain] ✅ XGBoost retrained and saved to {retrained_path}")
        self.last_retrain = datetime.now()
        return True
    
    def retrain_lightgbm(self) -> bool:
        try:
            import lightgbm as lgb
        except ImportError:
            print("  [ML Retrain] LightGBM not available")
            return False
        
        X, y, weights = self.prepare_training_data_from_trades()
        if X is None:
            return False
        
        print(f"  [ML Retrain] Training LightGBM on {len(X)} samples...")
        
        y_class = y + 1
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='multiclass',
            num_class=3,
            verbose=-1
        )
        
        model.fit(X, y_class, sample_weight=weights)
        
        retrained_path = os.path.join(self.model_path, 'lgbm_retrained.pkl')
        os.makedirs(self.model_path, exist_ok=True)
        with open(retrained_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  [ML Retrain] ✅ LightGBM retrained and saved to {retrained_path}")
        self.last_retrain = datetime.now()
        return True
    
    def retrain_all(self) -> Dict[str, bool]:
        results = {
            'xgboost': self.retrain_xgboost(),
            'lightgbm': self.retrain_lightgbm()
        }
        return results
    
    def get_retrain_status(self) -> Dict:
        closed_trades = self.memory.get_closed_trades(days=180)
        all_trades = len(self.memory.trades)
        return {
            'total_trades': len(closed_trades),
            'all_trades': all_trades,
            'min_required': self.min_samples_for_retrain,
            'ready_to_retrain': len(closed_trades) >= self.min_samples_for_retrain,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'training_samples': len(self.training_samples)
        }

class LearningEngine:
    
    def __init__(self, storage_path: str = 'learning_data'):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.memory = TradeMemory(storage_path)
        self.weight_optimizer = WeightOptimizer(self.memory, storage_path)
        self.prompt_enhancer = PromptEnhancer(self.memory, storage_path)
        self.param_tuner = ParameterTuner(self.memory, storage_path)
        self.ml_retrainer = MLRetrainer(self.memory, storage_path)
        
        self.last_optimization = None
        self.optimization_interval = timedelta(hours=6)
        self.retrain_interval = timedelta(days=7)
        
    def record_entry(self, trade_id: str, context: Dict) -> None:
        self.memory.record_trade_entry(trade_id, context)
    
    def record_exit(self, trade_id: str, exit_price: float, pnl_percent: float) -> None:
        self.memory.record_trade_exit(trade_id, exit_price, pnl_percent)
        
        if self.last_optimization is None or \
           datetime.now() - self.last_optimization > self.optimization_interval:
            self.run_optimization()
    
    def run_optimization(self) -> Dict:
        print("\n" + "="*60)
        print("🧠 LEARNING ENGINE - OPTIMIZATION CYCLE")
        print("="*60)
        
        stats = self.memory.get_performance_stats(days=30)
        print(f"\n📊 Last 30 days: {stats['total']} trades, {stats.get('win_rate', 0):.1f}% win rate")
        
        weights = self.weight_optimizer.optimize_weights()
        print(f"\n📐 Current weights: ML={weights['ml']:.2f}, News={weights['news']:.2f}, "
              f"Wisdom={weights['wisdom']:.2f}, Quant={weights['quant']:.2f}, HTF={weights['htf']:.2f}")
        
        self.prompt_enhancer.extract_patterns()
        
        params = self.param_tuner.tune_parameters()
        print(f"\n⚙️ Tuned params: RSI extreme=[{params['extreme_rsi_low']}, {params['extreme_rsi_high']}], "
              f"HTF penalty={params['htf_penalty_strong']:.2f}")
        
        self.last_optimization = datetime.now()
        
        print("="*60 + "\n")
        
        return {
            'stats': stats,
            'weights': weights,
            'params': params,
            'patterns_count': len(self.prompt_enhancer.patterns)
        }
    
    def get_optimized_weights(self) -> Dict[str, float]:
        return self.weight_optimizer.current_weights
    
    def get_tuned_parameters(self) -> Dict:
        return self.param_tuner.current_params
    
    def get_prompt_context(self, market_state: Dict) -> str:
        return self.prompt_enhancer.get_context_for_prompt(market_state)
    
    def get_stats(self) -> Dict:
        return self.memory.get_performance_stats()
    
    def check_and_retrain_ml(self) -> Dict[str, bool]:
        status = self.ml_retrainer.get_retrain_status()
        
        if not status['ready_to_retrain']:
            print(f"  [ML Retrain] Not ready: {status['total_trades']}/{status['min_required']} trades")
            return {'retrained': False, 'reason': 'insufficient_data'}
        
        if self.ml_retrainer.last_retrain:
            days_since = (datetime.now() - self.ml_retrainer.last_retrain).days
            if days_since < 7:
                return {'retrained': False, 'reason': f'retrained_{days_since}_days_ago'}
        
        print("\n" + "="*60)
        print("🤖 ML MODEL RETRAINING")
        print("="*60)
        
        results = self.ml_retrainer.retrain_all()
        
        print("="*60 + "\n")
        
        return {'retrained': True, 'results': results}
    
    def force_retrain_ml(self) -> Dict[str, bool]:
        print("\n" + "="*60)
        print("🤖 FORCED ML MODEL RETRAINING")
        print("="*60)
        
        results = self.ml_retrainer.retrain_all()
        
        print("="*60 + "\n")
        
        return results
    
    def get_ml_retrain_status(self) -> Dict:
        return self.ml_retrainer.get_retrain_status()

_learning_engine: Optional[LearningEngine] = None

def get_learning_engine(storage_path: str = 'learning_data') -> LearningEngine:
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine(storage_path)
    return _learning_engine

if __name__ == '__main__':
    engine = get_learning_engine()
    
    for i in range(5):
        trade_id = f"test_{i}"
        engine.record_entry(trade_id, {
            'entry_price': 85000 + i * 100,
            'side': 'LONG' if i % 2 == 0 else 'SHORT',
            'ml_score': 0.3 if i % 2 == 0 else -0.3,
            'ml_confidence': 0.75,
            'news_score': 0.2,
            'wisdom_score': 0.5,
            'wisdom_master': 'Warren Buffett',
            'quant_score': 0.1,
            'htf_bias': 0.5,
            'final_vote': 0.15,
            'rsi': 25 + i * 10,
            'volatility_percentile': 50
        })
        
        pnl = 2.0 if i % 3 != 0 else -1.5
        engine.record_exit(trade_id, 85000 + i * 100 + (pnl * 100), pnl)
    
    results = engine.run_optimization()
    print(f"\nOptimization results: {results}")
