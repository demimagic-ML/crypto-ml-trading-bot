"""Backtesting module for ML trading models."""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from binance.client import Client
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

class ModelBacktester:
    def __init__(self, model_dir='models_v2'):
        self.model_dir = model_dir
        self.time_steps = 60
        
        print(f"\n📦 Loading models from {model_dir}/...")
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(f'{model_dir}/xgb_model.json')
        
        self.lgbm_model = lgb.Booster(model_file=f'{model_dir}/lgbm_model.txt')
        
        with open(f'{model_dir}/feature_scaler_advanced.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        with open(f'{model_dir}/target_scaler_advanced.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
        
        with open(f'{model_dir}/feature_cols.pkl', 'rb') as f:
            self.feature_cols = pickle.load(f)
        
        print(f"  ✅ Loaded {len(self.feature_cols)} features")
        
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
    
    def fetch_data(self, symbol='BTCUSDT', interval='15m', days=30):
        print(f"\n📊 Fetching {days} days of {interval} data...")
        
        klines = self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=1000
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
            df[col] = df[col].astype(float)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"  ✅ Fetched {len(df)} candles")
        return df
    
    def add_indicators(self, df):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        df['momentum_1h'] = df['close'].pct_change(4)
        df['momentum_4h'] = df['close'].pct_change(16)
        df['momentum_1d'] = df['close'].pct_change(96)
        
        df['momentum_5p'] = df['close'].pct_change(5)
        df['momentum_10p'] = df['close'].pct_change(10)
        df['momentum_20p'] = df['close'].pct_change(20)
        
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - rolling_mean) / rolling_std
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_percentile'] = df['volatility'].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
        ).fillna(50)
        
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['close']
        
        df['ema_4h_fast'] = df['close'].ewm(span=16, adjust=False).mean()
        df['ema_4h_slow'] = df['close'].ewm(span=48, adjust=False).mean()
        df['htf_4h_bias'] = (df['ema_4h_fast'] - df['ema_4h_slow']) / df['close']
        
        df['ema_1d_fast'] = df['close'].ewm(span=96, adjust=False).mean()
        df['ema_1d_slow'] = df['close'].ewm(span=288, adjust=False).mean()
        df['htf_1d_bias'] = (df['ema_1d_fast'] - df['ema_1d_slow']) / df['close']
        
        df['htf_combined'] = np.where(
            (df['htf_4h_bias'] > 0) & (df['htf_1d_bias'] > 0), 1,
            np.where((df['htf_4h_bias'] < 0) & (df['htf_1d_bias'] < 0), -1, 0)
        )
        
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        df['rsi_slope'] = df['rsi'].diff(5)
        df['price_slope'] = df['close'].pct_change(5)
        df['rsi_divergence'] = np.where(
            (df['rsi_slope'] > 0) & (df['price_slope'] < 0), -1,
            np.where((df['rsi_slope'] < 0) & (df['price_slope'] > 0), 1, 0)
        )
        
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        if 'taker_buy_base' in df.columns:
            df['taker_buy_ratio'] = df['taker_buy_base'].astype(float) / df['volume']
        else:
            df['taker_buy_ratio'] = 0.5
        
        df['fundingRate'] = 0
        df['longShortRatio'] = 1
        df['longAccount'] = 0.5
        df['topTraderRatio'] = 1
        df['oi_change'] = 0
        
        return df.dropna()
    
    def predict(self, df):
        X = df[self.feature_cols].values
        X_scaled = self.feature_scaler.transform(X)
        
        sequences = []
        for i in range(self.time_steps, len(X_scaled)):
            sequences.append(X_scaled[i-self.time_steps:i])
        
        if len(sequences) == 0:
            return None, None
        
        X_seq = np.array(sequences)
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        
        pred_xgb = self.xgb_model.predict(X_flat)
        pred_lgbm = self.lgbm_model.predict(X_flat)
        
        pred_ensemble = (pred_xgb + pred_lgbm) / 2
        
        pred_inv = self.target_scaler.inverse_transform(pred_ensemble.reshape(-1, 1)).flatten()
        
        return pred_inv, df['close'].values[self.time_steps:]
    
    def backtest(self, initial_balance=10000, tp_pct=1.0, sl_pct=0.5, threshold=0.001):
        """Run backtest simulation with given parameters."""
        if not hasattr(self, '_data_cached'):
            df = self.fetch_data()
            df = self.add_indicators(df)
            self._data_cached = df
        else:
            df = self._data_cached
        
        predictions, actuals = self.predict(df)
        if predictions is None:
            print("❌ Not enough data for predictions")
            return
        
        prices = df['close'].values[self.time_steps:]
        
        balance = initial_balance
        position = None
        entry_price = 0
        wins = 0
        losses = 0
        trades = []
        
        for i in range(len(predictions) - 1):
            current_price = prices[i]
            predicted_price = predictions[i]
            next_price = prices[i + 1]
            
            predicted_change = (predicted_price - current_price) / current_price
            
            if position == 'LONG':
                pnl_pct = (next_price - entry_price) / entry_price * 100
                if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                    pnl = balance * (pnl_pct / 100)
                    balance += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    trades.append({'type': 'LONG', 'pnl': pnl_pct})
                    position = None
                    
            elif position == 'SHORT':
                pnl_pct = (entry_price - next_price) / entry_price * 100
                if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                    pnl = balance * (pnl_pct / 100)
                    balance += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    trades.append({'type': 'SHORT', 'pnl': pnl_pct})
                    position = None
            
            if position is None:
                if predicted_change > threshold:
                    position = 'LONG'
                    entry_price = current_price
                elif predicted_change < -threshold:
                    position = 'SHORT'
                    entry_price = current_price
        
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        profit_pct = ((balance - initial_balance) / initial_balance) * 100
        
        correct_dir = 0
        for i in range(len(predictions) - 1):
            pred_dir = 1 if predictions[i] > prices[i] else -1
            actual_dir = 1 if prices[i+1] > prices[i] else -1
            if pred_dir == actual_dir:
                correct_dir += 1
        
        dir_acc = correct_dir / (len(predictions) - 1) * 100
        
        return {
            'final_balance': balance,
            'profit_pct': profit_pct,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'direction_accuracy': dir_acc
        }

if __name__ == '__main__':
    backtester = ModelBacktester(model_dir='models_v2')
    
    configs = [
        {'threshold': 0.001, 'tp': 1.0, 'sl': 0.5, 'name': 'Default'},
        {'threshold': 0.002, 'tp': 1.5, 'sl': 0.5, 'name': 'Higher TP'},
        {'threshold': 0.003, 'tp': 1.0, 'sl': 0.3, 'name': 'Tight SL'},
        {'threshold': 0.002, 'tp': 2.0, 'sl': 0.8, 'name': 'Wide Range'},
        {'threshold': 0.005, 'tp': 1.5, 'sl': 0.5, 'name': 'High Thresh'},
        {'threshold': 0.001, 'tp': 0.8, 'sl': 0.3, 'name': 'Scalper'},
    ]
    
    print("\n" + "="*70)
    print("QUICK PARAMETER TUNING")
    print("="*70)
    
    results = []
    for cfg in configs:
        print(f"\n>> Testing: {cfg['name']}")
        r = backtester.backtest(
            initial_balance=10000,
            tp_pct=cfg['tp'],
            sl_pct=cfg['sl'],
            threshold=cfg['threshold']
        )
        if r:
            results.append({**cfg, **r})
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Config':<15} {'Thresh':>7} {'TP':>5} {'SL':>5} {'Profit':>10} {'WinRate':>8} {'Trades':>7}")
    print("-"*70)
    
    results.sort(key=lambda x: x['profit_pct'], reverse=True)
    for r in results:
        print(f"{r['name']:<15} {r['threshold']*100:>6.2f}% {r['tp']:>4.1f}% {r['sl']:>4.1f}% {r['profit_pct']:>+9.2f}% {r['win_rate']:>7.1f}% {r['total_trades']:>7}")
    
    print("\n✅ Best: " + results[0]['name'])
    print("="*70)
