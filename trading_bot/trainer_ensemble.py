"""
4-Model Ensemble Trainer: Transformer + XGBoost + LSTM + LightGBM
Higher confidence through model agreement.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from binance.client import Client

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Concatenate, Conv1D, Bidirectional, GRU
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not available - install with: pip install lightgbm")


class EnsembleTrainer:
    """
    4-Model Ensemble: Transformer + XGBoost + LSTM + LightGBM
    """
    
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.client = Client(api_key, api_secret)
        
        self.symbol = 'BTCUSDC'
        self.interval = Client.KLINE_INTERVAL_15MINUTE
        self.time_steps = 60
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        self.transformer_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.lgbm_model = None
        
    def fetch_data(self, years=5):
        """Fetch historical data."""
        print(f"Fetching {years} years of {self.symbol} data...")
        
        start_time = f"{years} year ago UTC"
        klines = self.client.get_historical_klines(
            self.symbol, 
            self.interval, 
            start_time
        )
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, np.nan)
        df['taker_sell_ratio'] = 1 - df['taker_buy_ratio']
        
        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        df = self._add_futures_data(df)
        
        return df
    
    def _add_futures_data(self, df):
        """Add futures market data: funding rate, open interest, long/short ratio."""
        print("Fetching futures data (funding rate, OI, L/S ratio)...")
        
        try:
            funding_data = self.client.futures_funding_rate(symbol='BTCUSDT', limit=1000)
            if funding_data:
                funding_df = pd.DataFrame(funding_data)
                funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
                funding_df['fundingRate'] = pd.to_numeric(funding_df['fundingRate'])
                funding_df.set_index('fundingTime', inplace=True)
                funding_df = funding_df[['fundingRate']]
                
                df = df.join(funding_df, how='left')
                df['fundingRate'] = df['fundingRate'].ffill().fillna(0)
                print(f"  Added funding rate data")
        except Exception as e:
            print(f"  Funding rate error: {e}")
            df['fundingRate'] = 0
        
        try:
            oi_data = self.client.futures_open_interest_hist(symbol='BTCUSDT', period='15m', limit=500)
            if oi_data:
                oi_df = pd.DataFrame(oi_data)
                oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                oi_df['sumOpenInterest'] = pd.to_numeric(oi_df['sumOpenInterest'])
                oi_df.set_index('timestamp', inplace=True)
                oi_df = oi_df[['sumOpenInterest']]
                oi_df.columns = ['openInterest']
                
                df = df.join(oi_df, how='left')
                df['openInterest'] = df['openInterest'].ffill().fillna(method='bfill')
                df['oi_change'] = df['openInterest'].pct_change()
                print(f"  Added open interest data")
        except Exception as e:
            print(f"  Open interest error: {e}")
            df['openInterest'] = 0
            df['oi_change'] = 0
        
        try:
            ls_data = self.client.futures_global_longshort_ratio(symbol='BTCUSDT', period='15m', limit=500)
            if ls_data:
                ls_df = pd.DataFrame(ls_data)
                ls_df['timestamp'] = pd.to_datetime(ls_df['timestamp'], unit='ms')
                ls_df['longShortRatio'] = pd.to_numeric(ls_df['longShortRatio'])
                ls_df['longAccount'] = pd.to_numeric(ls_df['longAccount'])
                ls_df.set_index('timestamp', inplace=True)
                ls_df = ls_df[['longShortRatio', 'longAccount']]
                
                df = df.join(ls_df, how='left')
                df['longShortRatio'] = df['longShortRatio'].ffill().fillna(1)
                df['longAccount'] = df['longAccount'].ffill().fillna(0.5)
                print(f"  Added long/short ratio data")
        except Exception as e:
            print(f"  Long/short ratio error: {e}")
            df['longShortRatio'] = 1
            df['longAccount'] = 0.5
        
        try:
            top_data = self.client.futures_top_longshort_position_ratio(symbol='BTCUSDT', period='15m', limit=500)
            if top_data:
                top_df = pd.DataFrame(top_data)
                top_df['timestamp'] = pd.to_datetime(top_df['timestamp'], unit='ms')
                top_df['longShortRatio'] = pd.to_numeric(top_df['longShortRatio'])
                top_df.set_index('timestamp', inplace=True)
                top_df = top_df[['longShortRatio']]
                top_df.columns = ['topTraderRatio']
                
                df = df.join(top_df, how='left')
                df['topTraderRatio'] = df['topTraderRatio'].ffill().fillna(1)
                print(f"  Added top trader ratio data")
        except Exception as e:
            print(f"  Top trader ratio error: {e}")
            df['topTraderRatio'] = 1
        
        return df
    
    def add_technical_indicators(self, df):
        """Add all technical indicators."""
        print("Adding technical indicators...")
        
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
        df['volume_delta'] = df['volume'].diff()
        
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
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
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
        
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        df['target'] = df['close'].shift(-1)
        
        df = df.dropna()
        
        print(f"Data shape after indicators: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """Prepare features for training."""
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position',
            'atr', 'atr_pct',
            'volume_ratio',
            'momentum_1h', 'momentum_4h', 'momentum_1d',
            'ema_cross',
            'stoch_k', 'stoch_d',
            'body_size', 'upper_wick', 'lower_wick',
            'taker_buy_ratio',
            'fundingRate',
            'longShortRatio',
            'longAccount',
            'topTraderRatio',
            'oi_change',
            'momentum_5p',
            'momentum_10p',
            'momentum_20p',
            'z_score',
            'volatility_percentile',
            'htf_4h_bias',
            'htf_1d_bias',
            'htf_combined',
            'rsi_divergence',
        ]
        
        X = df[feature_cols].values
        y = df['target'].values.reshape(-1, 1)
        
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        return X_scaled, y_scaled, feature_cols
    
    def create_sequences(self, X, y):
        """Create time series sequences."""
        X_seq, y_seq = [], []
        for i in range(self.time_steps, len(X)):
            X_seq.append(X[i-self.time_steps:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def build_transformer_model(self, input_shape):
        """Build Transformer-based model with CNN + Attention."""
        inputs = Input(shape=input_shape)
        
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = LayerNormalization()(x + attention_output)
        
        attention_output2 = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = LayerNormalization()(x + attention_output2)
        
        pooled = GlobalAveragePooling1D()(x)
        
        x = Dense(64, activation='relu')(pooled)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        
        return model
    
    def build_lstm_model(self, input_shape):
        """Build dedicated LSTM model."""
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with GridSearch."""
        if not XGB_AVAILABLE:
            return None
        
        print("\n[1/2] Training XGBoost with GridSearch...")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        X_combined = np.vstack([X_train_flat, X_val_flat])
        y_combined = np.concatenate([y_train.ravel(), y_val.ravel()])
        
        param_grid = {
            'n_estimators': [300],
            'max_depth': [6],
            'learning_rate': [0.03],
        }
        
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=3, scoring='neg_mean_squared_error',
            n_jobs=4, verbose=1
        )
        
        grid_search.fit(X_combined, y_combined)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best RMSE: ${np.sqrt(-grid_search.best_score_):.2f}")
        
        return grid_search.best_estimator_
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model with GridSearch."""
        if not LGBM_AVAILABLE:
            return None
        
        print("\n[2/2] Training LightGBM with GridSearch...")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        X_combined = np.vstack([X_train_flat, X_val_flat])
        y_combined = np.concatenate([y_train.ravel(), y_val.ravel()])
        
        param_grid = {
            'n_estimators': [300],
            'max_depth': [6],
            'learning_rate': [0.03],
        }
        
        base_model = lgb.LGBMRegressor(
            objective='regression',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=3, scoring='neg_mean_squared_error',
            n_jobs=4, verbose=1
        )
        
        grid_search.fit(X_combined, y_combined)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best RMSE: ${np.sqrt(-grid_search.best_score_):.2f}")
        
        return grid_search.best_estimator_
    
    def train(self):
        """Train the 2-model ensemble (gradient boosting only)."""
        print("=" * 60)
        print("2-MODEL ENSEMBLE TRAINER")
        print("XGBoost + LightGBM (best performers)")
        print("=" * 60)
        
        df = self.fetch_data(years=5)
        
        df = self.add_technical_indicators(df)
        
        X, y, feature_cols = self.prepare_features(df)
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        
        X_seq, y_seq = self.create_sequences(X, y)
        print(f"Sequences: {X_seq.shape}")
        
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        self.xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        self.lgbm_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        
        y_test_inv = self.target_scaler.inverse_transform(y_test)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        predictions = {}
        
        if self.xgb_model:
            pred_xgb = self.xgb_model.predict(X_test_flat).reshape(-1, 1)
            pred_xgb_inv = self.target_scaler.inverse_transform(pred_xgb)
            predictions['XGBoost'] = pred_xgb_inv
        
        if self.lgbm_model:
            pred_lgbm = self.lgbm_model.predict(X_test_flat).reshape(-1, 1)
            pred_lgbm_inv = self.target_scaler.inverse_transform(pred_lgbm)
            predictions['LightGBM'] = pred_lgbm_inv
        
        for name, pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_test_inv, pred))
            mae = mean_absolute_error(y_test_inv, pred)
            
            actual_dir = np.sign(np.diff(y_test_inv.flatten()))
            pred_dir = np.sign(pred[1:].flatten() - y_test_inv[:-1].flatten())
            dir_acc = np.mean(actual_dir == pred_dir) * 100
            
            print(f"{name:12} | RMSE: ${rmse:7.2f} | MAE: ${mae:7.2f} | Dir: {dir_acc:.1f}%")
        
        all_preds = np.stack([p for p in predictions.values()], axis=0)
        pred_ensemble = np.mean(all_preds, axis=0)
        
        rmse_ens = np.sqrt(mean_squared_error(y_test_inv, pred_ensemble))
        mae_ens = mean_absolute_error(y_test_inv, pred_ensemble)
        
        actual_dir = np.sign(np.diff(y_test_inv.flatten()))
        ens_dir = np.sign(pred_ensemble[1:].flatten() - y_test_inv[:-1].flatten())
        dir_acc_ens = np.mean(actual_dir == ens_dir) * 100
        
        print("-" * 60)
        print(f"{'ENSEMBLE':12} | RMSE: ${rmse_ens:7.2f} | MAE: ${mae_ens:7.2f} | Dir: {dir_acc_ens:.1f}%")
        
        print("\n" + "=" * 60)
        print("MODEL AGREEMENT ANALYSIS")
        print("=" * 60)
        
        directions = {}
        for name, pred in predictions.items():
            directions[name] = np.sign(pred[1:].flatten() - y_test_inv[:-1].flatten())
        
        agreement_2 = 0
        disagree = 0
        
        first_model = list(directions.keys())[0]
        for i in range(len(directions[first_model])):
            dirs = [directions[name][i] for name in directions.keys()]
            if len(set(dirs)) == 1:
                agreement_2 += 1
            else:
                disagree += 1
        
        total = len(directions[first_model])
        print(f"2/2 models agree: {agreement_2:5} ({agreement_2/total*100:.1f}%)")
        print(f"Models disagree:  {disagree:5} ({disagree/total*100:.1f}%)")
        
        print("\nSaving models to models_v2/...")
        os.makedirs('models_v2', exist_ok=True)
        
        if self.xgb_model:
            self.xgb_model.save_model('models_v2/xgb_model.json')
        
        if self.lgbm_model:
            self.lgbm_model.booster_.save_model('models_v2/lgbm_model.txt')
        
        with open('models_v2/feature_scaler_advanced.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        with open('models_v2/target_scaler_advanced.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)
        
        with open('models_v2/feature_cols.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        print("All models saved to models_v2/!")
        
        self._plot_results(y_test_inv, predictions, pred_ensemble)
        
        return {
            'ensemble_rmse': rmse_ens,
            'ensemble_mae': mae_ens,
            'directional_accuracy': dir_acc_ens,
            'agreement_2': agreement_2 / total * 100,
            'disagree': disagree / total * 100
        }
    
    def _plot_results(self, y_true, predictions, ensemble_pred):
        """Plot training results."""
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(y_true[-200:], label='Actual', linewidth=2, color='black')
        colors = ['blue', 'green', 'orange', 'red']
        for (name, pred), color in zip(predictions.items(), colors):
            axes[0, 0].plot(pred[-200:], label=name, alpha=0.6, color=color)
        axes[0, 0].set_title('All Models vs Actual (Last 200)')
        axes[0, 0].legend()
        
        axes[0, 1].plot(y_true[-200:], label='Actual', linewidth=2, color='black')
        axes[0, 1].plot(ensemble_pred[-200:], label='Ensemble', linewidth=2, color='purple')
        axes[0, 1].set_title('Ensemble vs Actual (Last 200)')
        axes[0, 1].legend()
        
        for (name, pred), color in zip(predictions.items(), colors):
            errors = pred.flatten() - y_true.flatten()
            axes[1, 0].hist(errors, bins=50, alpha=0.5, label=name, color=color)
        axes[1, 0].set_title('Error Distribution by Model')
        axes[1, 0].legend()
        axes[1, 0].axvline(x=0, color='black', linestyle='--')
        
        axes[1, 1].scatter(y_true, ensemble_pred, alpha=0.3, s=1)
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Ensemble Predicted')
        axes[1, 1].set_title('Ensemble: Actual vs Predicted')
        
        plt.tight_layout()
        plt.savefig('results/ensemble_training.png', dpi=150)
        print("Plot saved to results/ensemble_training.png")


if __name__ == '__main__':
    trainer = EnsembleTrainer()
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Ensemble RMSE: ${results['ensemble_rmse']:.2f}")
    print(f"Ensemble MAE: ${results['ensemble_mae']:.2f}")
    print(f"Directional Accuracy: {results['directional_accuracy']:.1f}%")
    print(f"2/2 Agreement: {results['agreement_2']:.1f}%")
