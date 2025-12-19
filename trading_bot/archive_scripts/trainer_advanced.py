"""Advanced trainer for ML models."""
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Concatenate, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available, using LSTM only")

class AdvancedTrainer:
    """Advanced trainer for transformer and XGBoost models."""
    
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
        
    def fetch_data(self, years=2):
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
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
    
    def add_technical_indicators(self, df):
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
        
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['close']
        
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
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
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position',
            'atr', 'atr_pct',
            'volume_ratio',
            'momentum_1h', 'momentum_4h', 'momentum_1d',
            'ema_cross',
            'stoch_k', 'stoch_d',
            'body_size', 'upper_wick', 'lower_wick'
        ]
        
        X = df[feature_cols].values
        y = df['target'].values.reshape(-1, 1)
        
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        return X_scaled, y_scaled, feature_cols
    
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(self.time_steps, len(X)):
            X_seq.append(X[i-self.time_steps:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def build_transformer_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=32
        )(x, x)
        x = LayerNormalization()(x + attention_output)
        
        attention_output2 = MultiHeadAttention(
            num_heads=4, key_dim=32
        )(x, x)
        x = LayerNormalization()(x + attention_output2)
        
        lstm_out = LSTM(64, return_sequences=True)(x)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(32)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        
        pooled = GlobalAveragePooling1D()(x)
        
        combined = Concatenate()([lstm_out, pooled])
        
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        if not XGB_AVAILABLE:
            return None
        
        print("\nTraining XGBoost model...")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train_flat, y_train.ravel(),
            eval_set=[(X_val_flat, y_val.ravel())],
            verbose=False
        )
        
        return model
    
    def train(self):
        print("="*60)
        print("ADVANCED TRAINER - Transformer + XGBoost Ensemble")
        print("="*60)
        
        df = self.fetch_data(years=2)
        
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
        
        print("\nBuilding Transformer model...")
        self.transformer_model = self.build_transformer_model(
            (X_train.shape[1], X_train.shape[2])
        )
        self.transformer_model.summary()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint('models/transformer_model.h5', save_best_only=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ]
        
        print("\nTraining Transformer model...")
        history = self.transformer_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        self.xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        pred_transformer = self.transformer_model.predict(X_test, verbose=0)
        pred_transformer_inv = self.target_scaler.inverse_transform(pred_transformer)
        y_test_inv = self.target_scaler.inverse_transform(y_test)
        
        rmse_transformer = np.sqrt(mean_squared_error(y_test_inv, pred_transformer_inv))
        mae_transformer = mean_absolute_error(y_test_inv, pred_transformer_inv)
        
        print(f"Transformer RMSE: ${rmse_transformer:.2f}")
        print(f"Transformer MAE: ${mae_transformer:.2f}")
        
        if self.xgb_model:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            pred_xgb = self.xgb_model.predict(X_test_flat).reshape(-1, 1)
            pred_xgb_inv = self.target_scaler.inverse_transform(pred_xgb)
            
            rmse_xgb = np.sqrt(mean_squared_error(y_test_inv, pred_xgb_inv))
            mae_xgb = mean_absolute_error(y_test_inv, pred_xgb_inv)
            
            print(f"XGBoost RMSE: ${rmse_xgb:.2f}")
            print(f"XGBoost MAE: ${mae_xgb:.2f}")
            
            pred_ensemble = (pred_transformer_inv + pred_xgb_inv) / 2
            rmse_ensemble = np.sqrt(mean_squared_error(y_test_inv, pred_ensemble))
            mae_ensemble = mean_absolute_error(y_test_inv, pred_ensemble)
            
            print(f"Ensemble RMSE: ${rmse_ensemble:.2f}")
            print(f"Ensemble MAE: ${mae_ensemble:.2f}")
        
        actual_direction = np.sign(np.diff(y_test_inv.flatten()))
        pred_direction = np.sign(pred_transformer_inv[1:].flatten() - y_test_inv[:-1].flatten())
        dir_accuracy = np.mean(actual_direction == pred_direction) * 100
        print(f"Directional Accuracy: {dir_accuracy:.1f}%")
        
        print("\nSaving models...")
        os.makedirs('models', exist_ok=True)
        
        with open('models/feature_scaler_advanced.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        with open('models/target_scaler_advanced.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)
        
        if self.xgb_model:
            self.xgb_model.save_model('models/xgb_model.json')
        
        with open('models/feature_cols.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        print("Models saved!")
        
        self._plot_results(y_test_inv, pred_transformer_inv, history)
        
        return {
            'rmse': rmse_transformer,
            'mae': mae_transformer,
            'directional_accuracy': dir_accuracy
        }
    
    def _plot_results(self, y_true, y_pred, history):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(history.history['loss'], label='Train')
        axes[0, 0].plot(history.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        
        axes[0, 1].plot(y_true[-200:], label='Actual', alpha=0.7)
        axes[0, 1].plot(y_pred[-200:], label='Predicted', alpha=0.7)
        axes[0, 1].set_title('Predictions vs Actual (Last 200)')
        axes[0, 1].legend()
        
        axes[1, 0].scatter(y_true, y_pred, alpha=0.3, s=1)
        axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title('Actual vs Predicted')
        
        errors = y_pred.flatten() - y_true.flatten()
        axes[1, 1].hist(errors, bins=50, edgecolor='black')
        axes[1, 1].set_title(f'Error Distribution (Mean: ${errors.mean():.2f})')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.savefig('results/advanced_training.png', dpi=150)
        print("Plot saved to results/advanced_training.png")

if __name__ == '__main__':
    trainer = AdvancedTrainer()
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"RMSE: ${results['rmse']:.2f}")
    print(f"MAE: ${results['mae']:.2f}")
    print(f"Directional Accuracy: {results['directional_accuracy']:.1f}%")
