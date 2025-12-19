"""Advanced trading bot with ML ensemble and multi-pillar signals."""
import os
import sys
import pickle
import time
import signal
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import schedule
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from binance.client import Client
from news_fetcher import NewsFetcher
from trading_wisdom import TradingWisdom
from quant_analysis import QuantAnalysis
from whale_tracker import WhaleTracker
from learning_engine import get_learning_engine

try:
    from llm_overlord import LLMOverlord
    OVERLORD_AVAILABLE = True
except ImportError:
    OVERLORD_AVAILABLE = False

try:
    import dashboard_server
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

class AdvancedBot:
    """Advanced trading bot with ML ensemble and multi-pillar signals."""
    
    def __init__(self):
        self.symbol = 'BTCUSDC'
        self.interval = Client.KLINE_INTERVAL_15MINUTE
        self.time_steps = 60
        
        self.paper_trading = os.getenv('TRADING_MODE', 'paper').lower() == 'paper'
        
        self.position = None
        self.entry_price = None
        self.quantity = 0
        
        self.stop_loss_pct = 0.008
        self.take_profit_min = 0.015
        self.take_profit_max = 0.030
        self.take_profit_pct = 0.020
        self.leverage = 10
        self.position_size_pct = 0.60
        
        self.trailing_stop_enabled = True
        self.trailing_stop_activation = 0.01
        self.trailing_stop_distance = 0.005
        self.current_trailing_stop = None
        
        self.signal_based_exit = True
        self.reversal_threshold = 0.25
        self.profit_take_threshold = 0.008
        
        self.use_limit_orders = True
        self.scaled_entry_enabled = True
        self.entry_mode = 'auto'
        self.aggressive_signal_threshold = 0.35
        self.num_entry_tranches = 4
        self.tranche_offsets = [0.0005, 0.0015, 0.0025, 0.004]
        self.tranche_weights = [0.30, 0.30, 0.25, 0.15]
        self.limit_timeout_base = 60
        self.limit_timeout_max = 180
        self.min_spread_bps = 5
        self.max_spread_bps = 30
        self.adverse_move_cancel = 0.003
        self.aggressive_fill_threshold = 0.5
        self._recent_candles = None
        self._active_orders = []
        
        self.ml_weight = 0.30
        self.news_weight = 0.20
        self.wisdom_weight = 0.30
        self.quant_weight = 0.20
        
        self.news_trigger_threshold = 0.7
        self.news_check_interval = 2
        
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data', 'market')
        os.makedirs(self.data_dir, exist_ok=True)
        
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.client = Client(api_key, api_secret, requests_params={"timeout": 30})
        
        self._sync_time()
        
        self._load_models()
        
        self.news_fetcher = NewsFetcher()
        self.last_news = None
        self.last_news_fetch_time = None
        self.news_cache_duration = 3600
        
        self.wisdom = TradingWisdom()
        self.last_wisdom = None
        
        self.quant = QuantAnalysis(client=self.client)
        self.last_quant = None
        
        self.whale_tracker = WhaleTracker()
        self.last_whale = None
        
        if OVERLORD_AVAILABLE:
            self.overlord = LLMOverlord()
            self.last_overlord = None
        else:
            self.overlord = None
            self.last_overlord = None
        
        self.learning_engine = get_learning_engine(os.path.join(os.path.dirname(__file__), 'learning_data'))
        self.trade_counter = 0
        self.current_trade_id = None
        
        self._setup_futures_leverage()
        
        self.usdt_balance = self._get_futures_balance()
        
        self.state_file = 'state/advanced_bot_state.json'
        self._load_state()
        self._check_and_fix_open_positions()
        
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        mode_str = 'PAPER' if self.paper_trading else '🔴 LIVE'
        print(f"\n{'='*60}")
        print(f"FUTURES BOT - {self.leverage}x Leverage + ML Ensemble")
        print(f"{'='*60}")
        print(f"Mode: {mode_str}")
        print(f"Symbol: {self.symbol}")
        print(f"USDC Balance: ${self.usdt_balance:.2f}")
        print(f"ML: {self.ml_weight*100:.0f}% | News: {self.news_weight*100:.0f}% | Wisdom: {self.wisdom_weight*100:.0f}%")
        print(f"{'='*60}\n")
    
    def _signal_handler(self, signum, frame):
        print("\nShutdown signal received...")
        self.running = False
    
    def _sync_time(self):
        try:
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            self.client.timestamp_offset = server_time['serverTime'] - local_time
        except Exception as e:
            print(f"Time sync failed: {e}")
    
    def _setup_futures_leverage(self):
        if self.paper_trading:
            return
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            print(f"Futures leverage set to {self.leverage}x")
        except Exception as e:
            print(f"Leverage setup warning: {e}")
    
    def _get_futures_balance(self):
        if self.paper_trading:
            return 100.0
        try:
            account = self.client.futures_account()
            return float(account['availableBalance'])
        except Exception as e:
            print(f"Error getting futures balance: {e}")
            return 0.0
    
    def _get_futures_position(self):
        if self.paper_trading:
            return None, 0, 0
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    qty = float(pos['positionAmt'])
                    entry = float(pos['entryPrice'])
                    if qty > 0:
                        return 'LONG', abs(qty), entry
                    elif qty < 0:
                        return 'SHORT', abs(qty), entry
            return None, 0, 0
        except Exception as e:
            print(f"Error getting position: {e}")
            return None, 0, 0
    
    def _get_trading_costs(self, hours=24):
        if self.paper_trading:
            return {'commission': 0, 'funding': 0, 'total': 0, 'trades': []}
        
        try:
            start_time = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
            income = self.client.futures_income_history(
                symbol=self.symbol,
                startTime=start_time,
                limit=100
            )
            
            commission_total = 0
            funding_total = 0
            trades = []
            
            for item in income:
                income_type = item.get('incomeType', '')
                amount = float(item.get('income', 0))
                asset = item.get('asset', 'USDC')
                time_ms = item.get('time', 0)
                
                if income_type == 'COMMISSION':
                    commission_total += abs(amount)
                    trades.append({
                        'type': 'commission',
                        'amount': amount,
                        'asset': asset,
                        'time': datetime.fromtimestamp(time_ms/1000).strftime('%H:%M:%S')
                    })
                elif income_type == 'FUNDING_FEE':
                    funding_total += amount
                    trades.append({
                        'type': 'funding',
                        'amount': amount,
                        'asset': asset,
                        'time': datetime.fromtimestamp(time_ms/1000).strftime('%H:%M:%S')
                    })
            
            return {
                'commission': commission_total,
                'funding': funding_total,
                'total': commission_total + abs(funding_total),
                'trades': trades
            }
        except Exception as e:
            print(f"Error getting costs: {e}")
            return {'commission': 0, 'funding': 0, 'total': 0, 'trades': []}
    
    def _show_position_costs(self):
        if self.paper_trading or not self.position:
            return
        
        costs = self._get_trading_costs(hours=24)
        if costs['total'] > 0:
            print(f"  💰 COSTS (24h): Commission: ${costs['commission']:.4f} | Funding: ${costs['funding']:+.4f} | Total: ${costs['total']:.4f}")
    
    def _load_models(self):
        self.models_loaded = []
        
        try:
            model_dir = 'models_v2' if os.path.exists('models_v2/xgb_model.json') else 'models'
            print(f"  📦 Loading models from {model_dir}/")
            
            if os.path.exists(f'{model_dir}/xgb_model.json'):
                with open(f'{model_dir}/feature_scaler_advanced.pkl', 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                with open(f'{model_dir}/target_scaler_advanced.pkl', 'rb') as f:
                    self.target_scaler = pickle.load(f)
                with open(f'{model_dir}/feature_cols.pkl', 'rb') as f:
                    self.feature_cols = pickle.load(f)
                
                if XGB_AVAILABLE:
                    if os.path.exists(f'{model_dir}/xgb_model.json'):
                        self.xgb_model = xgb.XGBRegressor()
                        self.xgb_model.load_model(f'{model_dir}/xgb_model.json')
                        self.models_loaded.append('XGBoost')
                    else:
                        self.xgb_model = None
                else:
                    self.xgb_model = None
                
                if LGBM_AVAILABLE:
                    if os.path.exists(f'{model_dir}/lgbm_model.txt'):
                        self.lgbm_model = lgb.Booster(model_file=f'{model_dir}/lgbm_model.txt')
                        self.models_loaded.append('LightGBM')
                    else:
                        self.lgbm_model = None
                else:
                    self.lgbm_model = None
                
                self.transformer_model = None
                
                self.model_type = 'ensemble'
                print(f"Loaded {len(self.models_loaded)}-model ensemble: {', '.join(self.models_loaded)}")
            else:
                self.transformer_model = load_model('models/model_15m.h5')
                with open('models/feature_scaler_15m.pkl', 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                with open('models/close_scaler_15m.pkl', 'rb') as f:
                    self.target_scaler = pickle.load(f)
                self.feature_cols = ['open', 'high', 'low', 'volume', 'rsi']
                self.xgb_model = None
                self.lstm_model = None
                self.lgbm_model = None
                self.model_type = 'simple'
                self.models_loaded = ['Simple LSTM']
                print("Loaded simple LSTM model (advanced not found)")
                
        except Exception as e:
            print(f"Failed to load models: {e}")
            sys.exit(1)
    
    def _load_state(self):
        os.makedirs('state', exist_ok=True)
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.position = state.get('position')
            self.entry_price = state.get('entry_price')
            self.quantity = state.get('quantity', 0)
            print(f"Loaded state: {self.position}, Qty={self.quantity}")
        except:
            print("No saved state, starting fresh")
    
    def _save_state(self):
        state = {
            'position': self.position,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'last_save': datetime.utcnow().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _check_and_fix_open_positions(self):
        if self.paper_trading:
            return
        try:
            actual_pos, actual_qty, actual_entry = self._get_futures_position()
            
            if actual_pos != self.position or abs(actual_qty - self.quantity) > 0.0001 or abs((actual_entry or 0) - (self.entry_price or 0)) > 0.01:
                print(f"  ⚡ SYNCING with Binance: {actual_pos} {actual_qty:.4f} @ ${actual_entry:.2f}")
                
                if self.position is not None and actual_pos is None:
                    print(f"  🎯 Position closed by TP/SL order!")
                    
                    try:
                        self._cancel_open_orders()
                        print(f"  🧹 Cancelled remaining TP/SL orders")
                    except Exception as e:
                        print(f"  Error cancelling orders: {e}")
                    
                    try:
                        current_price = float(self.client.futures_symbol_ticker(symbol=self.symbol)['price'])
                        if self.position == 'LONG':
                            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                        else:
                            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                        
                        if hasattr(self, 'current_trade_id') and self.current_trade_id:
                            self.learning_engine.record_exit(self.current_trade_id, current_price, pnl_pct)
                            self.current_trade_id = None
                            leveraged_pnl = pnl_pct * self.leverage
                            self._add_dashboard_log(f"CLOSED {self.position}: TP/SL Hit | PnL: {leveraged_pnl:+.1f}%")
                    except Exception as e:
                        print(f"  Error recording TP/SL exit: {e}")
                
                self.position = actual_pos
                self.quantity = actual_qty
                self.entry_price = actual_entry
                self._save_state()
            
            if self.position:
                print(f"  📍 Current Position: {self.position} {self.quantity:.4f} BTC @ ${self.entry_price:.2f}")
                self._show_position_costs()
                
                current_price = float(self.client.futures_symbol_ticker(symbol=self.symbol)['price'])
                if self.position == 'LONG':
                    pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                else:
                    pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                leveraged_pnl = pnl_pct * self.leverage
                
                self._update_dashboard(
                    currentPrice=current_price,
                    position={
                        'side': self.position,
                        'quantity': self.quantity,
                        'entryPrice': self.entry_price,
                        'pnl': leveraged_pnl,
                        'pnlPercent': pnl_pct
                    }
                )
        except Exception as e:
            print(f"  Position check error: {e}")
    
    def get_market_data(self, limit=200):
        klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    
    def log_market_snapshot(self, df, news_data=None):
        try:
            df_with_ind = self.add_indicators(df.copy())
            if len(df_with_ind) == 0:
                return
            
            latest = df_with_ind.iloc[-1].to_dict()
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'open_time': str(latest.get('open_time', '')),
                'price': latest.get('close', 0),
                'open': latest.get('open', 0),
                'high': latest.get('high', 0),
                'low': latest.get('low', 0),
                'volume': latest.get('volume', 0),
                'rsi': latest.get('rsi', 0),
                'macd': latest.get('macd', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'macd_hist': latest.get('macd_hist', 0),
                'bb_width': latest.get('bb_width', 0),
                'bb_position': latest.get('bb_position', 0),
                'atr': latest.get('atr', 0),
                'atr_pct': latest.get('atr_pct', 0),
                'volume_ratio': latest.get('volume_ratio', 0),
                'momentum_1h': latest.get('momentum_1h', 0),
                'momentum_4h': latest.get('momentum_4h', 0),
                'stoch_k': latest.get('stoch_k', 0),
                'stoch_d': latest.get('stoch_d', 0),
                'ema_cross': latest.get('ema_cross', 0),
            }
            
            if news_data:
                snapshot['news_sentiment'] = news_data.get('sentiment', 'NEUTRAL')
                snapshot['news_score'] = news_data.get('sentiment_score', 0)
                snapshot['news_confidence'] = news_data.get('confidence', 0)
                snapshot['news_suggestion'] = news_data.get('trading_suggestion', 'HOLD')
            
            today = datetime.now().strftime('%Y-%m-%d')
            csv_path = os.path.join(self.data_dir, f'{today}_market_data.csv')
            
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a') as f:
                if not file_exists:
                    f.write(','.join(snapshot.keys()) + '\n')
                f.write(','.join(str(v) for v in snapshot.values()) + '\n')
            
        except Exception as e:
            print(f"  Log error: {e}")
    
    def add_indicators(self, df):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        if self.model_type in ['advanced', 'ensemble']:
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
            
            futures_data = self._get_futures_data()
            df['fundingRate'] = futures_data.get('funding_rate', 0)
            df['longShortRatio'] = futures_data.get('long_short_ratio', 1)
            df['longAccount'] = futures_data.get('long_account', 0.5)
            df['topTraderRatio'] = futures_data.get('top_trader_ratio', 1)
            df['oi_change'] = futures_data.get('oi_change', 0)
        
        return df.dropna()
    
    def predict(self, df):
        current_price = float(df['close'].iloc[-1])
        
        df = self.add_indicators(df.copy())
        if len(df) < self.time_steps:
            return current_price, None, 0, {}
        
        X = df[self.feature_cols].tail(self.time_steps).values
        X_scaled = self.feature_scaler.transform(X)
        X_seq = X_scaled.reshape(1, self.time_steps, len(self.feature_cols))
        X_flat = X_scaled.reshape(1, -1)
        
        predictions = {}
        
        if self.xgb_model:
            pred_xgb = self.xgb_model.predict(X_flat).reshape(-1, 1)
            predictions['XGBoost'] = self.target_scaler.inverse_transform(pred_xgb)[0][0]
        
        if self.lgbm_model:
            pred_lgbm = self.lgbm_model.predict(X_flat).reshape(-1, 1)
            predictions['LightGBM'] = self.target_scaler.inverse_transform(pred_lgbm)[0][0]
        
        prediction = np.mean(list(predictions.values()))
        
        directions = [1 if p > current_price else -1 for p in predictions.values()]
        agree = len(set(directions)) == 1
        
        if agree:
            ml_confidence = 0.85
        else:
            ml_confidence = 0.4
        
        return current_price, float(prediction), ml_confidence, predictions
    
    def get_news_signal(self, force_refresh=False):
        now = datetime.now()
        
        should_fetch = force_refresh or \
                       self.last_news is None or \
                       self.last_news_fetch_time is None or \
                       (now - self.last_news_fetch_time).total_seconds() >= self.news_cache_duration
        
        if should_fetch:
            try:
                print("  📰 Fetching fresh news (hourly update)...")
                self.last_news = self.news_fetcher.fetch_all_news()
                self.last_news_fetch_time = now
                sentiment = self.last_news.get('sentiment', 'NEUTRAL')
                score = self.last_news.get('sentiment_score', 0)
                confidence = self.last_news.get('confidence', 0)
                print(f"  News: {sentiment} ({score:+.2f}, conf: {confidence:.0%})")
                return score, confidence
            except Exception as e:
                print(f"  News fetch error: {e}")
                return 0.0, 0.0
        else:
            cache_age = int((now - self.last_news_fetch_time).total_seconds() / 60)
            sentiment = self.last_news.get('sentiment', 'NEUTRAL')
            score = self.last_news.get('sentiment_score', 0)
            confidence = self.last_news.get('confidence', 0)
            print(f"  📰 Using cached news ({cache_age}m old): {sentiment} ({score:+.2f})")
            return score, confidence
    
    def check_breaking_news(self):
        """Check for breaking news that might trigger trades."""
        try:
            self.news_fetcher.cache_duration = 0
            news = self.news_fetcher.fetch_all_news()
            self.news_fetcher.cache_duration = 300
            
            df = self.get_market_data()
            self.log_market_snapshot(df, news)
            
            score = news.get('sentiment_score', 0)
            confidence = news.get('confidence', 0)
            events = news.get('key_events', [])
            suggestion = news.get('trading_suggestion', 'HOLD')
            
            if abs(score) >= self.news_trigger_threshold and confidence >= 0.8:
                print(f"\n{'!'*50}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] BREAKING NEWS DETECTED!")
                print(f"{'!'*50}")
                print(f"  Sentiment: {score:+.2f} (confidence: {confidence:.0%})")
                print(f"  Events: {events}")
                print(f"  Suggestion: {suggestion}")
                
                if self.position is None:
                    self._trigger_news_trade(score, suggestion)
                elif (self.position == 'LONG' and score < -0.5) or \
                     (self.position == 'SHORT' and score > 0.5):
                    print(f"  >> News contradicts {self.position} position!")
                    df = self.get_market_data()
                    current_price = float(df['close'].iloc[-1])
                    if self.position == 'LONG':
                        self._close_long(current_price, "NEWS SIGNAL")
                    else:
                        self._close_short(current_price, "NEWS SIGNAL")
                
                return True
            
            return False
            
        except Exception as e:
            print(f"  Breaking news check error: {e}")
            return False
    
    def _trigger_news_trade(self, score, suggestion):
        df = self.get_market_data()
        current_price = float(df['close'].iloc[-1])
        
        self.usdc_balance = self._get_margin_balance('USDC')
        
        if suggestion == 'BUY' or score > self.news_trigger_threshold:
            print(f"  >> NEWS TRIGGER: Opening LONG @ ${current_price:.2f}")
            self._open_long(current_price)
        elif suggestion == 'SELL' or score < -self.news_trigger_threshold:
            print(f"  >> NEWS TRIGGER: Opening SHORT @ ${current_price:.2f}")
            self._open_short(current_price)
        
        self._save_state()
    
    def _get_futures_data(self):
        futures_data = {
            'funding_rate': 0,
            'long_short_ratio': 1,
            'long_account': 0.5,
            'top_trader_ratio': 1,
            'taker_buy_ratio': 0.5,
            'oi_change': 0
        }
        
        try:
            funding = self.client.futures_funding_rate(symbol='BTCUSDC', limit=1)
            if funding:
                futures_data['funding_rate'] = float(funding[0]['fundingRate'])
            
            ls = self.client.futures_global_longshort_ratio(symbol='BTCUSDC', period='1h', limit=2)
            if ls:
                futures_data['long_short_ratio'] = float(ls[0]['longShortRatio'])
                futures_data['long_account'] = float(ls[0]['longAccount'])
            
            top = self.client.futures_top_longshort_position_ratio(symbol='BTCUSDC', period='1h', limit=1)
            if top:
                futures_data['top_trader_ratio'] = float(top[0]['longShortRatio'])
            
            oi = self.client.futures_open_interest_hist(symbol='BTCUSDC', period='15m', limit=2)
            if oi and len(oi) >= 2:
                oi_now = float(oi[0]['sumOpenInterest'])
                oi_prev = float(oi[1]['sumOpenInterest'])
                futures_data['oi_change'] = (oi_now - oi_prev) / oi_prev if oi_prev > 0 else 0
            
            kline = self.client.get_klines(symbol=self.symbol, interval='15m', limit=1)
            if kline:
                volume = float(kline[0][5])
                taker_buy = float(kline[0][9])
                futures_data['taker_buy_ratio'] = taker_buy / volume if volume > 0 else 0.5
                
        except Exception as e:
            print(f"  [Futures] Data fetch error: {e}")
        
        return futures_data
    
    def get_wisdom_signal(self, df, current_price, prediction, news_score):
        try:
            futures_data = self._get_futures_data()
            
            market_data = {
                'price': current_price,
                'change_24h': float(df['close'].pct_change(96).iloc[-1] * 100) if len(df) > 96 else 0,
                'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50,
                'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0,
                'macd_signal': float(df['macd_signal'].iloc[-1]) if 'macd_signal' in df.columns else 0,
                'macd_hist': float(df['macd_hist'].iloc[-1]) if 'macd_hist' in df.columns else 0,
                'bb_position': float(df['bb_position'].iloc[-1]) if 'bb_position' in df.columns else 0.5,
                'atr_pct': float(df['atr_pct'].iloc[-1]) if 'atr_pct' in df.columns else 1,
                'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 1,
                'ema_cross': float(df['ema_cross'].iloc[-1]) if 'ema_cross' in df.columns else 0,
                'stoch_k': float(df['stoch_k'].iloc[-1]) if 'stoch_k' in df.columns else 50,
                'stoch_d': float(df['stoch_d'].iloc[-1]) if 'stoch_d' in df.columns else 50,
                'momentum_1h': float(df['momentum_1h'].iloc[-1]) if 'momentum_1h' in df.columns else 0,
                'momentum_4h': float(df['momentum_4h'].iloc[-1]) if 'momentum_4h' in df.columns else 0,
                'momentum_1d': float(df['momentum_1d'].iloc[-1]) if 'momentum_1d' in df.columns else 0,
                'funding_rate': futures_data['funding_rate'],
                'long_short_ratio': futures_data['long_short_ratio'],
                'long_account': futures_data['long_account'],
                'top_trader_ratio': futures_data['top_trader_ratio'],
                'taker_buy_ratio': futures_data['taker_buy_ratio'],
                'oi_change': futures_data['oi_change'],
                'ml_prediction': prediction,
                'ml_change': (prediction - current_price) / current_price * 100,
                'news_sentiment': self.last_news.get('sentiment', 'NEUTRAL') if self.last_news else 'NEUTRAL',
                'news_score': news_score,
                'current_position': self.position or 'NONE'
            }
            
            print(f"  [Futures] FR:{futures_data['funding_rate']:.4%} | L/S:{futures_data['long_short_ratio']:.2f} | Taker:{futures_data['taker_buy_ratio']:.2f} | OI:{futures_data['oi_change']:+.2%}")
            
            historical_context = self.learning_engine.get_prompt_context({
                'rsi': market_data.get('rsi', 50),
                'htf_bias': self.last_quant.get('htf_bias', 0) if self.last_quant else 0,
                'volatility_percentile': self.last_quant.get('vol_percentile', 50) if self.last_quant else 50
            })
            if historical_context:
                market_data['historical_patterns'] = historical_context
            
            wisdom_score, wisdom_confidence, reasoning = self.wisdom.get_wisdom_signal(market_data)
            full_analysis = self.wisdom.last_analysis or {}
            self.last_wisdom = {
                'score': wisdom_score, 
                'confidence': wisdom_confidence, 
                'reasoning': reasoning,
                'signal': full_analysis.get('signal', 'N/A'),
                'trade_quality': full_analysis.get('trade_quality', 'N/A'),
                'which_master_speaks': full_analysis.get('which_master_speaks', 'N/A'),
                'warnings': full_analysis.get('warnings', [])
            }
            return wisdom_score, wisdom_confidence
            
        except Exception as e:
            print(f"  [Wisdom] Error: {e}")
            return 0.0, 0.0
    
    def get_combined_signal(self, current_price, prediction, news_score, wisdom_score, wisdom_confidence,
                            quant_score=0, quant_confidence=0.5, htf_bias=0.0, rsi=50.0):
        """Combine signals from all pillars into final trading decision."""
        learned_weights = self.learning_engine.get_optimized_weights()
        ml_weight = learned_weights.get('ml', 0.305)
        news_weight = learned_weights.get('news', 0.105)
        wisdom_weight = learned_weights.get('wisdom', 0.205)
        quant_weight = learned_weights.get('quant', 0.33)
        htf_weight = learned_weights.get('htf', 0.025)
        
        tuned_params = self.learning_engine.get_tuned_parameters()
        
        ml_signal = (prediction - current_price) / current_price
        ml_direction = 'LONG' if ml_signal > 0.001 else 'SHORT' if ml_signal < -0.001 else 'HOLD'
        ml_strength = min(abs(ml_signal) * 100, 1.0)
        
        news = self.last_news or {}
        news_confidence = news.get('confidence', 0.5)
        if news_score > 0.1:
            news_direction = 'LONG'
        elif news_score < -0.1:
            news_direction = 'SHORT'
        else:
            news_direction = 'HOLD'
        news_strength = abs(news_score) * news_confidence
        
        if wisdom_score > 0.1:
            wisdom_direction = 'LONG'
        elif wisdom_score < -0.1:
            wisdom_direction = 'SHORT'
        else:
            wisdom_direction = 'HOLD'
        wisdom_strength = abs(wisdom_score) * wisdom_confidence
        
        if quant_score > 0.1:
            quant_direction = 'LONG'
        elif quant_score < -0.1:
            quant_direction = 'SHORT'
        else:
            quant_direction = 'HOLD'
        quant_strength = abs(quant_score) * quant_confidence
        
        if htf_bias > 0.3:
            htf_direction = 'LONG'
        elif htf_bias < -0.3:
            htf_direction = 'SHORT'
        else:
            htf_direction = 'HOLD'
        htf_strength = abs(htf_bias)
        
        directions = [ml_direction, news_direction, wisdom_direction, quant_direction, htf_direction]
        non_hold = [d for d in directions if d != 'HOLD']
        non_hold_count = len(non_hold)
        
        long_count = sum(1 for d in directions if d == 'LONG')
        short_count = sum(1 for d in directions if d == 'SHORT')
        
        if non_hold_count <= 1:
            threshold = 0.20
            threshold_reason = "single-pillar (0.20)"
        elif non_hold_count == 2 and len(set(non_hold)) > 1:
            threshold = 0.15
            threshold_reason = "split signals (0.15)"
        else:
            threshold = 0.08
            threshold_reason = "consensus (0.08)"
        
        z_score = self.last_quant.get('z_score', 0) if self.last_quant else 0
        if abs(z_score) > 1.5:
            threshold *= 0.7
            threshold_reason += f" + Z-score extreme ({z_score:+.1f}) → {threshold:.2f}"
        elif abs(z_score) > 1.0:
            threshold *= 0.85
            threshold_reason += f" + Z-score ({z_score:+.1f}) → {threshold:.2f}"
        
        dir_map = {'LONG': 1, 'HOLD': 0, 'SHORT': -1}
        
        ml_vote = dir_map[ml_direction] * ml_strength * ml_weight
        news_vote = dir_map[news_direction] * news_strength * news_weight
        wisdom_vote = dir_map[wisdom_direction] * wisdom_strength * wisdom_weight
        quant_vote = dir_map[quant_direction] * quant_strength * quant_weight
        htf_vote = dir_map[htf_direction] * htf_strength * htf_weight
        
        total_vote = ml_vote + news_vote + wisdom_vote + quant_vote + htf_vote
        
        extreme_rsi_low = tuned_params.get('extreme_rsi_low', 25)
        extreme_rsi_high = tuned_params.get('extreme_rsi_high', 75)
        htf_penalty_strong = tuned_params.get('htf_penalty_strong', 0.6)
        htf_penalty_moderate = tuned_params.get('htf_penalty_moderate', 0.8)
        
        extreme_oversold = rsi < extreme_rsi_low
        extreme_overbought = rsi > extreme_rsi_high
        extreme_condition = extreme_oversold or extreme_overbought
        
        if abs(z_score) > 1.5:
            if z_score > 1.5:
                extreme_oversold = True
            else:
                extreme_overbought = True
            extreme_condition = True
        
        htf_modifier = 1.0
        htf_msg = ""
        
        if extreme_oversold and total_vote > 0.02 and wisdom_score > 0.3:
            htf_modifier = 0.90
            htf_msg = f"🔥 EXTREME OVERSOLD (RSI={rsi:.1f}, Z={z_score:+.1f}) + Wisdom BULLISH - HTF penalty reduced to 10%"
        elif extreme_overbought and total_vote < -0.02 and wisdom_score < -0.3:
            htf_modifier = 0.90
            htf_msg = f"🔥 EXTREME OVERBOUGHT (RSI={rsi:.1f}, Z={z_score:+.1f}) + Wisdom BEARISH - HTF penalty reduced to 10%"
        elif htf_bias < -0.7 and total_vote > 0.05:
            htf_modifier = htf_penalty_strong
            htf_msg = f"⚠️ HTF OVERRIDE: Daily trend BEARISH ({htf_bias:+.2f}) against LONG - reducing by {(1-htf_penalty_strong)*100:.0f}%"
        elif htf_bias > 0.7 and total_vote < -0.05:
            htf_modifier = htf_penalty_strong
            htf_msg = f"⚠️ HTF OVERRIDE: Daily trend BULLISH ({htf_bias:+.2f}) against SHORT - reducing by {(1-htf_penalty_strong)*100:.0f}%"
        elif htf_bias < -0.5 and total_vote > 0:
            htf_modifier = htf_penalty_moderate
            htf_msg = f"⚠️ HTF CAUTION: Daily trend bearish ({htf_bias:+.2f}) - reducing by {(1-htf_penalty_moderate)*100:.0f}%"
        elif htf_bias > 0.5 and total_vote < 0:
            htf_modifier = htf_penalty_moderate
            htf_msg = f"⚠️ HTF CAUTION: Daily trend bullish ({htf_bias:+.2f}) - reducing by {(1-htf_penalty_moderate)*100:.0f}%"
        
        total_vote_htf = total_vote * htf_modifier
        
        print(f"   ┌──────────────────────────────────────────────────────────────┐")
        print(f"   │ 5-PILLAR SIGNAL BREAKDOWN (REBALANCED):                      │")
        print(f"   │   ML:     {ml_direction:6} × {ml_strength:.3f} × {ml_weight:.2f} = {ml_vote:+.4f}         │")
        print(f"   │   News:   {news_direction:6} × {news_strength:.3f} × {news_weight:.2f} = {news_vote:+.4f}         │")
        print(f"   │   Wisdom: {wisdom_direction:6} × {wisdom_strength:.3f} × {wisdom_weight:.2f} = {wisdom_vote:+.4f}         │")
        print(f"   │   Quant:  {quant_direction:6} × {quant_strength:.3f} × {quant_weight:.2f} = {quant_vote:+.4f}         │")
        print(f"   │   HTF:    {htf_direction:6} × {htf_strength:.3f} × {htf_weight:.2f} = {htf_vote:+.4f}         │")
        print(f"   │   ──────────────────────────────────────────────             │")
        print(f"   │   RAW VOTE: {total_vote:+.4f}                                     │")
        print(f"   │   Threshold: {threshold_reason}                          │")
        print(f"   └──────────────────────────────────────────────────────────────┘")
        
        if htf_msg:
            print(f"   {htf_msg}")
            print(f"   │   HTF Adjusted: {total_vote:+.4f} × {htf_modifier:.2f} = {total_vote_htf:+.4f}")
        
        whale_sentiment = self.last_whale.get('sentiment', 'NEUTRAL') if self.last_whale else 'NEUTRAL'
        whale_score = self.last_whale.get('score', 0) if self.last_whale else 0
        whale_net_flow = self.last_whale.get('net_flow', 0) if self.last_whale else 0
        
        whale_modifier = 1.0
        whale_msg = ""
        
        if total_vote_htf > 0.05:
            if whale_sentiment == 'BULLISH' and whale_score > 0.3:
                whale_modifier = 1.25
                whale_msg = f"🐋 WHALE BOOST: Withdrawals support LONG (+25%)"
            elif whale_sentiment == 'BEARISH' and whale_score < -0.3:
                whale_modifier = 0.5
                whale_msg = f"🐋 WHALE WARNING: {abs(whale_net_flow):.0f} BTC deposited - reducing (-50%)"
            elif whale_sentiment == 'BEARISH':
                whale_modifier = 0.75
                whale_msg = f"🐋 WHALE CAUTION: Deposits detected - reducing (-25%)"
        elif total_vote_htf < -0.05:
            if whale_sentiment == 'BEARISH' and whale_score < -0.3:
                whale_modifier = 1.25
                whale_msg = f"🐋 WHALE BOOST: Deposits support SHORT (+25%)"
            elif whale_sentiment == 'BULLISH' and whale_score > 0.3:
                whale_modifier = 0.5
                whale_msg = f"🐋 WHALE WARNING: {abs(whale_net_flow):.0f} BTC withdrawn - reducing (-50%)"
            elif whale_sentiment == 'BULLISH':
                whale_modifier = 0.75
                whale_msg = f"🐋 WHALE CAUTION: Withdrawals detected - reducing (-25%)"
        
        final_vote = total_vote_htf * whale_modifier
        
        if whale_msg:
            print(f"   {whale_msg}")
            print(f"   │   Final Vote: {total_vote_htf:+.4f} × {whale_modifier:.2f} = {final_vote:+.4f}")
        
        if final_vote > threshold:
            final_direction = 'LONG'
        elif final_vote < -threshold:
            final_direction = 'SHORT'
        else:
            final_direction = 'HOLD'
        
        total_strength = abs(final_vote)
        
        if long_count >= 3 or short_count >= 3:
            print(f"  ✓ STRONG CONSENSUS: {max(long_count, short_count)}/5 pillars agree")
            total_strength *= 1.3
        elif long_count >= 2 and short_count == 0:
            print(f"  ✓ CONSENSUS: {long_count} pillars support LONG")
            total_strength *= 1.1
        elif short_count >= 2 and long_count == 0:
            print(f"  ✓ CONSENSUS: {short_count} pillars support SHORT")
            total_strength *= 1.1
        elif long_count > 0 and short_count > 0:
            print(f"  ⚠ CONFLICT: {long_count} LONG vs {short_count} SHORT - reduced confidence")
            total_strength *= 0.6
        elif non_hold_count == 1:
            print(f"  ⚠ SINGLE PILLAR: Only 1 signal active - needs {threshold} threshold")
        
        if self.last_wisdom and final_direction != 'HOLD':
            reasoning = self.last_wisdom.get('reasoning', '')[:100]
            print(f"  🎯 {reasoning}")
        
        return final_direction, total_strength
    
    def execute_strategy(self):
        try:
            print(f"\n{'='*70}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 MARKET ANALYSIS")
            print(f"{'='*70}")
            
            self._reset_thinking()
            self._update_thinking('ml', 'running', {'message': 'Analyzing price data...'})
            
            df = self.get_market_data()
            current_price, prediction, ml_confidence, model_preds = self.predict(df)
            
            if prediction is None:
                print("  Not enough data")
                self._reset_thinking()
                return
            
            ml_direction = 'LONG' if prediction > current_price else 'SHORT' if prediction < current_price else 'HOLD'
            self._update_thinking('ml', 'complete', {
                'direction': ml_direction,
                'prediction': prediction,
                'confidence': ml_confidence,
                'models': model_preds
            })
            
            print(f"\n📊 ML MODEL PREDICTIONS:")
            print(f"   Current Price: ${current_price:,.2f}")
            if model_preds:
                for name, pred in model_preds.items():
                    diff = pred - current_price
                    diff_pct = (diff / current_price) * 100
                    direction = "🟢 LONG" if pred > current_price else "🔴 SHORT"
                    print(f"   {name:12}: ${pred:,.2f} ({diff_pct:+.3f}%) → {direction}")
                
                n_models = len(model_preds)
                directions = {k: 'LONG' if v > current_price else 'SHORT' for k, v in model_preds.items()}
                agreement = sum(1 for d in directions.values() if d == list(directions.values())[0])
                print(f"   Agreement: {agreement}/{n_models} models agree")
            
            self._update_thinking('news', 'running', {'message': 'Analyzing news sentiment...'})
            print(f"\n📰 NEWS ANALYSIS:")
            news_score, news_confidence = self.get_news_signal()
            
            news_sentiment = 'BULLISH' if news_score > 0.2 else 'BEARISH' if news_score < -0.2 else 'NEUTRAL'
            self._update_thinking('news', 'complete', {
                'sentiment': news_sentiment,
                'score': news_score,
                'confidence': news_confidence,
                'cached': self.last_news_fetch_time is not None
            })
            
            if self.last_news:
                print(f"   Headlines analyzed: {self.last_news.get('news_count', 0)}")
                for h in self.last_news.get('headlines', [])[:3]:
                    print(f"   • {h[:60]}...")
                print(f"   Sentiment: {self.last_news.get('sentiment', 'N/A')} (score: {news_score:+.2f})")
                print(f"   Confidence: {news_confidence:.0%}")
                if self.last_news.get('events'):
                    print(f"   Key Events: {', '.join(self.last_news.get('events', [])[:2])}")
            
            df_with_indicators = self.add_indicators(df.copy())
            latest = df_with_indicators.iloc[-1]
            print(f"\n📈 TECHNICAL INDICATORS:")
            print(f"   RSI: {latest.get('rsi', 0):.1f} {'(OVERSOLD)' if latest.get('rsi', 50) < 30 else '(OVERBOUGHT)' if latest.get('rsi', 50) > 70 else ''}")
            print(f"   MACD: {latest.get('macd', 0):.2f} | Signal: {latest.get('macd_signal', 0):.2f} | Hist: {latest.get('macd_hist', 0):.2f}")
            print(f"   BB Position: {latest.get('bb_position', 0.5):.2f} (0=lower band, 1=upper band)")
            print(f"   Volume Ratio: {latest.get('volume_ratio', 1):.2f}x avg")
            print(f"   Momentum: 1h={latest.get('momentum_1h', 0)*100:+.2f}% | 4h={latest.get('momentum_4h', 0)*100:+.2f}%")
            
            print(f"\n📊 FUTURES SENTIMENT:")
            print(f"   Funding Rate: {latest.get('fundingRate', 0)*100:.4f}%")
            print(f"   Long/Short Ratio: {latest.get('longShortRatio', 1):.2f} ({latest.get('longAccount', 0.5)*100:.0f}% long)")
            print(f"   Taker Buy Ratio: {latest.get('taker_buy_ratio', 0.5):.2f}")
            print(f"   OI Change: {latest.get('oi_change', 0)*100:+.2f}%")
            
            self._update_thinking('wisdom', 'running', {'message': 'Consulting trading masters...'})
            print(f"\n🧙 WISDOM ORACLE:")
            wisdom_score, wisdom_confidence = self.get_wisdom_signal(
                df_with_indicators, current_price, prediction, news_score
            )
            wisdom_direction = 'LONG' if wisdom_score > 0.1 else 'SHORT' if wisdom_score < -0.1 else 'HOLD'
            self._update_thinking('wisdom', 'complete', {
                'direction': wisdom_direction,
                'score': wisdom_score,
                'confidence': wisdom_confidence,
                'master': self.last_wisdom.get('which_master_speaks', '') if self.last_wisdom else '',
                'grade': self.last_wisdom.get('trade_quality', '') if self.last_wisdom else ''
            })
            if self.last_wisdom:
                print(f"   Signal: {self.last_wisdom.get('signal', 'N/A')}")
                print(f"   Score: {wisdom_score:+.2f} | Confidence: {wisdom_confidence:.0%}")
                print(f"   Grade: {self.last_wisdom.get('trade_quality', 'N/A')}")
                print(f"   Master: {self.last_wisdom.get('which_master_speaks', 'N/A')}")
                if self.last_wisdom.get('reasoning'):
                    print(f"   Reasoning: {self.last_wisdom.get('reasoning', '')[:80]}...")
                if self.last_wisdom.get('warnings'):
                    print(f"   ⚠️ Warnings: {', '.join(self.last_wisdom.get('warnings', [])[:2])}")
            
            self._update_thinking('quant', 'running', {'message': 'Running quantitative analysis...'})
            print(f"\n📐 QUANT ANALYSIS:")
            self.last_quant = self.quant.analyze(
                df_with_indicators, current_price, 
                self.entry_price, self.position, self.usdt_balance,
                symbol=self.symbol
            )
            quant_score = self.last_quant['signal_score']
            quant_confidence = self.last_quant['confidence']
            momentum = self.last_quant.get('momentum', {})
            quant_direction = 'LONG' if quant_score > 0.1 else 'SHORT' if quant_score < -0.1 else 'HOLD'
            self._update_thinking('quant', 'complete', {
                'direction': quant_direction,
                'score': quant_score,
                'zScore': self.last_quant.get('z_score', 0),
                'htfTrend': self.last_quant.get('htf_trend', 'NEUTRAL'),
                'swingLow': self.last_quant.get('swing_low', False),
                'swingHigh': self.last_quant.get('swing_high', False)
            })
            print(f"   Z-Score: {self.last_quant['z_score']:+.2f}")
            print(f"   Momentum: 5p={momentum.get('5p', 0):+.2f}% | 10p={momentum.get('10p', 0):+.2f}% | 20p={momentum.get('20p', 0):+.2f}%")
            print(f"   Volatility: {self.last_quant['volatility_regime']} ({self.last_quant['volatility_percentile']:.0f}th pctl)")
            print(f"   HTF Trend: {self.last_quant.get('htf_trend', 'N/A')} (bias: {self.last_quant.get('htf_bias', 0):+.2f})")
            
            swing_low = self.last_quant.get('swing_low', False)
            swing_high = self.last_quant.get('swing_high', False)
            swing_score = self.last_quant.get('swing_score', 0)
            recent_low = self.last_quant.get('recent_swing_low')
            recent_high = self.last_quant.get('recent_swing_high')
            
            swing_status = "🎯 SWING LOW" if swing_low else "🎯 SWING HIGH" if swing_high else "—"
            print(f"   Swing: {swing_status} (score: {swing_score:+.2f})")
            if recent_low and recent_high:
                dist_low = self.last_quant.get('distance_from_swing_low', 0)
                dist_high = self.last_quant.get('distance_from_swing_high', 0)
                print(f"   Range: Low ${recent_low:.0f} ({dist_low:+.1f}%) | High ${recent_high:.0f} ({dist_high:+.1f}%)")
            
            projected_entry = self.last_quant.get('projected_entry')
            entry_type = self.last_quant.get('entry_type', 'MARKET')
            if projected_entry and entry_type == 'LIMIT':
                offset_pct = self.last_quant.get('limit_offset_pct', 0)
                print(f"   📍 Projected Entry: ${projected_entry:.1f} ({entry_type}, {offset_pct:.2f}% offset)")
            
            print(f"   Kelly: {self.last_quant['kelly_fraction']:.1%} → ${self.last_quant['optimal_position_size']:.2f}")
            print(f"   Signal: {self.last_quant['signal']} (score: {quant_score:+.2f}, conf: {quant_confidence:.0%})")
            print(f"   Reasoning: {self.last_quant['reasoning']}")
            
            self._update_thinking('whale', 'running', {'message': 'Tracking whale activity...'})
            print(f"\n🐋 WHALE ACTIVITY:")
            self.last_whale = self.whale_tracker.analyze(current_price)
            whale_sentiment = self.last_whale.get('sentiment', 'NEUTRAL')
            whale_score = self.last_whale.get('score', 0)
            self._update_thinking('whale', 'complete', {
                'sentiment': whale_sentiment,
                'score': whale_score,
                'netFlow': self.last_whale.get('net_flow', 0)
            })
            print(f"   Exchange Flow: {whale_sentiment} (score: {whale_score:+.2f})")
            print(f"   Deposits: {self.last_whale.get('deposits', 0)} txs ({self.last_whale.get('deposit_btc', 0):.1f} BTC)")
            print(f"   Withdrawals: {self.last_whale.get('withdrawals', 0)} txs ({self.last_whale.get('withdrawal_btc', 0):.1f} BTC)")
            print(f"   Net Flow: {self.last_whale.get('net_flow', 0):+.1f} BTC (${self.last_whale.get('net_flow_usd', 0):,.0f})")
            if self.last_whale.get('reasoning'):
                print(f"   Analysis: {self.last_whale.get('reasoning')}")
            
            self.log_market_snapshot(df, self.last_news)
            
            print(f"\n🎯 SIGNAL CALCULATION:")
            htf_bias = self.last_quant.get('htf_bias', 0.0)
            current_rsi = latest.get('rsi', 50.0)
            signal, strength = self.get_combined_signal(
                current_price, prediction, news_score, wisdom_score, wisdom_confidence,
                quant_score, quant_confidence, htf_bias, current_rsi
            )
            
            self._last_ml_score = (prediction - current_price) / current_price
            self._last_ml_confidence = ml_confidence
            self._last_final_vote = strength
            self._last_rsi = current_rsi
            self._last_bb_position = latest.get('bb_position', 0.5)
            self._last_volume_ratio = latest.get('volume_ratio', 1.0)
            self._last_funding_rate = self.last_quant.get('funding_rate', 0) if self.last_quant else 0
            self._last_macd = latest.get('macd', 0)
            self._last_momentum_1h = latest.get('momentum_1h', 0)
            
            pred_pct = ((prediction / current_price) - 1) * 100
            
            overlord_decision = None
            if self.overlord and self.overlord.enabled:
                self._update_thinking('overlord', 'running', {'message': 'DeepSeek R1 making final decision...'})
                print(f"\n🤖 LLM OVERLORD (DeepSeek R1):")
                
                pos_pnl_pct = 0
                pos_pnl_leveraged = 0
                if self.position and self.entry_price:
                    if self.position == 'LONG':
                        pos_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                    else:
                        pos_pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                    pos_pnl_leveraged = pos_pnl_pct * self.leverage
                
                overlord_context = {
                    'current_price': current_price,
                    'position': self.position,
                    'position_info': {
                        'entry_price': self.entry_price,
                        'quantity': self.quantity,
                        'pnl_pct': pos_pnl_pct,
                        'pnl_leveraged': pos_pnl_leveraged
                    },
                    'ml': {
                        'direction': 'LONG' if prediction > current_price * 1.001 else 'SHORT' if prediction < current_price * 0.999 else 'HOLD',
                        'predicted_price': prediction,
                        'confidence': ml_confidence
                    },
                    'news': {
                        'direction': self.last_news.get('trading_suggestion', 'HOLD') if self.last_news else 'HOLD',
                        'score': news_score,
                        'confidence': news_confidence,
                        'summary': self.last_news.get('summary', '') if self.last_news else ''
                    },
                    'wisdom': {
                        'direction': self.last_wisdom.get('signal', 'HOLD') if self.last_wisdom else 'HOLD',
                        'score': wisdom_score,
                        'master': self.last_wisdom.get('which_master_speaks', '') if self.last_wisdom else '',
                        'insight': self.last_wisdom.get('analysis', '') if self.last_wisdom else ''
                    },
                    'quant': {
                        'signal': self.last_quant.get('signal', 'HOLD') if self.last_quant else 'HOLD',
                        'z_score': self.last_quant.get('z_score', 0) if self.last_quant else 0,
                        'momentum_5p': momentum.get('5p', 0),
                        'momentum_10p': momentum.get('10p', 0),
                        'momentum_20p': momentum.get('20p', 0),
                        'volatility_regime': self.last_quant.get('volatility_regime', 'NORMAL') if self.last_quant else 'NORMAL',
                        'volatility_pctl': self.last_quant.get('volatility_percentile', 50) if self.last_quant else 50,
                        'htf_trend': self.last_quant.get('htf_trend', 'NEUTRAL') if self.last_quant else 'NEUTRAL',
                        'htf_bias': htf_bias,
                        'swing_low': self.last_quant.get('swing_low', False) if self.last_quant else False,
                        'swing_high': self.last_quant.get('swing_high', False) if self.last_quant else False,
                        'swing_score': self.last_quant.get('swing_score', 0) if self.last_quant else 0,
                        'projected_entry': self.last_quant.get('projected_entry', current_price) if self.last_quant else current_price,
                        'entry_type': self.last_quant.get('entry_type', 'MARKET') if self.last_quant else 'MARKET'
                    },
                    'whale': {
                        'sentiment': self.last_whale.get('sentiment', 'NEUTRAL') if self.last_whale else 'NEUTRAL',
                        'net_flow': self.last_whale.get('net_flow', 0) if self.last_whale else 0,
                        'reasoning': self.last_whale.get('reasoning', '') if self.last_whale else ''
                    },
                    'ensemble': {
                        'direction': signal,
                        'strength': strength,
                        'threshold': 0.08
                    }
                }
                
                overlord_decision = self.overlord.make_decision(overlord_context)
                self.last_overlord = overlord_decision
                
                self._update_thinking('overlord', 'complete', {
                    'decision': overlord_decision.get('decision', 'HOLD'),
                    'confidence': overlord_decision.get('confidence', 0),
                    'reasoning': overlord_decision.get('reasoning', '')[:100],
                    'riskLevel': overlord_decision.get('risk_level', 'MEDIUM')
                })
                
                cached_str = " (cached)" if overlord_decision.get('cached') else ""
                print(f"   Decision: {overlord_decision['decision']} @ {overlord_decision['confidence']}% confidence{cached_str}")
                print(f"   Entry: {overlord_decision.get('entry_type', 'MARKET')}")
                print(f"   Risk: {overlord_decision.get('risk_level', 'MEDIUM')}")
                print(f"   Reasoning: {overlord_decision.get('reasoning', 'N/A')[:80]}")
                if overlord_decision.get('key_factors'):
                    print(f"   Key Factors: {', '.join(overlord_decision['key_factors'][:3])}")
                
                if not overlord_decision.get('passthrough'):
                    old_signal = signal
                    signal = overlord_decision['decision']
                    if signal != old_signal:
                        print(f"   🔄 OVERLORD DECISION: {old_signal} → {signal}")
            
            self._update_thinking('overlord', 'complete' if overlord_decision else 'skipped', 
                                  final_decision={'action': signal, 'strength': strength})
            
            print(f"\n{'='*70}")
            print(f"📌 FINAL DECISION: {signal} (strength: {strength:+.4f})")
            print(f"   Ensemble Prediction: ${prediction:,.2f} ({pred_pct:+.2f}%)")
            print(f"   ML Confidence: {ml_confidence:.0%}")
            if overlord_decision and not overlord_decision.get('passthrough'):
                print(f"   Overlord: {overlord_decision['decision']} @ {overlord_decision['confidence']}%")
            print(f"{'='*70}")
            
            import time as time_module
            time_module.sleep(3)
            self._reset_thinking()
            
            self.usdt_balance = self._get_futures_balance()
            
            news_sentiment = 'NEUTRAL'
            if news_score > 0.2:
                news_sentiment = 'BULLISH'
            elif news_score < -0.2:
                news_sentiment = 'BEARISH'
            
            ml_direction = 'HOLD'
            if prediction > current_price * 1.001:
                ml_direction = 'LONG'
            elif prediction < current_price * 0.999:
                ml_direction = 'SHORT'
            
            costs = self._get_trading_costs(hours=24)
            
            pos_pnl = 0
            pos_pnl_pct = 0
            if self.position and self.entry_price:
                if self.position == 'LONG':
                    pos_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                else:
                    pos_pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                pos_pnl = pos_pnl_pct * self.leverage
            
            self._update_dashboard(
                mode='LIVE' if not self.paper_trading else 'PAPER',
                symbol=self.symbol,
                balance=self.usdt_balance,
                currentPrice=current_price,
                position={
                    'side': self.position,
                    'quantity': self.quantity,
                    'entryPrice': self.entry_price or 0,
                    'pnl': pos_pnl,
                    'pnlPercent': pos_pnl_pct
                },
                signals={
                    'ml': {
                        'direction': ml_direction,
                        'confidence': int(ml_confidence * 100),
                        'prediction': prediction
                    },
                    'news': {
                        'sentiment': news_sentiment,
                        'score': news_score,
                        'confidence': int(news_confidence * 100)
                    },
                    'wisdom': {
                        'signal': self.last_wisdom.get('signal', 'HOLD') if self.last_wisdom else 'HOLD',
                        'grade': self.last_wisdom.get('trade_quality', 'C') if self.last_wisdom else 'C',
                        'master': self.last_wisdom.get('which_master_speaks', '') if self.last_wisdom else '',
                        'reasoning': self.last_wisdom.get('reasoning', '') if self.last_wisdom else '',
                        'specialty': self.last_wisdom.get('trader_specialty', '') if self.last_wisdom else '',
                        'style': self.last_wisdom.get('trader_style', '') if self.last_wisdom else '',
                        'selectionReason': self.last_wisdom.get('selection_reason', '') if self.last_wisdom else '',
                        'fullAnalysis': self.last_wisdom if self.last_wisdom else {},
                        'keyLevels': self.last_wisdom.get('key_levels', {}) if self.last_wisdom else {}
                    },
                    'quant': {
                        'signal': self.last_quant.get('signal', 'HOLD') if self.last_quant else 'HOLD',
                        'zScore': self.last_quant.get('z_score', 0) if self.last_quant else 0,
                        'momentum': sum(momentum.values()) / len(momentum) if momentum else 0,
                        'htfTrend': self.last_quant.get('htf_trend', 'NEUTRAL') if self.last_quant else 'NEUTRAL',
                        'htfBias': self.last_quant.get('htf_bias', 0) if self.last_quant else 0,
                        'volatilityRegime': self.last_quant.get('volatility_regime', 'NORMAL') if self.last_quant else 'NORMAL',
                        'volatilityPctl': self.last_quant.get('volatility_percentile', 50) if self.last_quant else 50,
                        'kellyFraction': self.last_quant.get('kelly_fraction', 0) if self.last_quant else 0,
                        'optimalSize': self.last_quant.get('optimal_position_size', 0) if self.last_quant else 0,
                        'reasoning': self.last_quant.get('reasoning', '') if self.last_quant else '',
                        'momentum5p': momentum.get('5p', 0) if momentum else 0,
                        'momentum10p': momentum.get('10p', 0) if momentum else 0,
                        'momentum20p': momentum.get('20p', 0) if momentum else 0
                    }
                },
                whale={
                    'sentiment': self.last_whale.get('sentiment', 'NEUTRAL') if self.last_whale else 'NEUTRAL',
                    'score': self.last_whale.get('score', 0) if self.last_whale else 0,
                    'deposits': self.last_whale.get('deposits', 0) if self.last_whale else 0,
                    'withdrawals': self.last_whale.get('withdrawals', 0) if self.last_whale else 0,
                    'depositBtc': self.last_whale.get('deposit_btc', 0) if self.last_whale else 0,
                    'withdrawalBtc': self.last_whale.get('withdrawal_btc', 0) if self.last_whale else 0,
                    'netFlow': self.last_whale.get('net_flow', 0) if self.last_whale else 0,
                    'netFlowUsd': self.last_whale.get('net_flow_usd', 0) if self.last_whale else 0,
                    'reasoning': self.last_whale.get('reasoning', '') if self.last_whale else '',
                    'largeTxs': self.last_whale.get('large_txs', [])[:5] if self.last_whale else [],
                    'hasAlert': abs(self.last_whale.get('net_flow', 0)) > 100 if self.last_whale else False
                },
                costs=costs,
                decision={'action': signal, 'strength': abs(strength)},
                entryMode={
                    'mode': self.entry_mode,
                    'scaledEnabled': self.scaled_entry_enabled,
                    'aggressiveThreshold': self.aggressive_signal_threshold,
                    'currentSignalStrength': abs(strength)
                },
                learning=self._get_learning_stats(),
                overlord={
                    'enabled': self.overlord.enabled if self.overlord else False,
                    'decision': self.last_overlord.get('decision', 'N/A') if self.last_overlord else 'N/A',
                    'confidence': self.last_overlord.get('confidence', 0) if self.last_overlord else 0,
                    'reasoning': self.last_overlord.get('reasoning', '') if self.last_overlord else '',
                    'entryType': self.last_overlord.get('entry_type', 'MARKET') if self.last_overlord else 'MARKET',
                    'riskLevel': self.last_overlord.get('risk_level', 'MEDIUM') if self.last_overlord else 'MEDIUM',
                    'keyFactors': self.last_overlord.get('key_factors', []) if self.last_overlord else [],
                    'cached': self.last_overlord.get('cached', False) if self.last_overlord else False,
                    'model': 'DeepSeek R1'
                } if self.overlord else None
            )
            
            self._add_dashboard_log(f"DECISION: {signal} | Price: ${current_price:.2f} | Strength: {strength:+.4f}")
            
            overlord_confidence = overlord_decision.get('confidence', 0) / 100 if overlord_decision else 0
            effective_strength = max(strength, overlord_confidence)
            
            if self.position is None:
                if signal == 'LONG':
                    self._open_long(current_price)
                elif signal == 'SHORT':
                    self._open_short(current_price)
            else:
                reversal_threshold = 0.2
                if overlord_decision and overlord_confidence >= 0.5:
                    reversal_threshold = 0.0
                
                if self.position == 'LONG' and signal == 'SHORT' and effective_strength > reversal_threshold:
                    print(f"  🔄 REVERSAL: LONG → SHORT (strength: {effective_strength:.2f}, overlord: {overlord_confidence:.0%})")
                    self._close_long(current_price, "OVERLORD REVERSAL" if overlord_decision else "SIGNAL REVERSAL")
                    self._open_short(current_price)
                elif self.position == 'SHORT' and signal == 'LONG' and effective_strength > reversal_threshold:
                    print(f"  🔄 REVERSAL: SHORT → LONG (strength: {effective_strength:.2f}, overlord: {overlord_confidence:.0%})")
                    self._close_short(current_price, "OVERLORD REVERSAL" if overlord_decision else "SIGNAL REVERSAL")
                    self._open_long(current_price)
                else:
                    self._manage_position(current_price, signal, strength)
            
            self._save_state()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_order_book_spread(self) -> dict:
        try:
            depth = self.client.futures_order_book(symbol=self.symbol, limit=20)
            best_bid = float(depth['bids'][0][0])
            best_ask = float(depth['asks'][0][0])
            spread = best_ask - best_bid
            spread_bps = (spread / best_bid) * 10000
            
            bid_depth = sum(float(b[1]) for b in depth['bids'][:10])
            ask_depth = sum(float(a[1]) for a in depth['asks'][:10])
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_bps': spread_bps,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': imbalance
            }
        except Exception as e:
            return None
    
    def _find_support_resistance(self, side: str, current_price: float) -> list:
        try:
            candles = self.client.futures_klines(symbol=self.symbol, interval='15m', limit=48)
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            
            levels = []
            
            for i in range(2, len(candles) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    levels.append(('support', lows[i]))
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    levels.append(('resistance', highs[i]))
            
            if side == 'BUY':
                relevant = [l[1] for l in levels if l[0] == 'support' and l[1] < current_price]
                relevant.sort(reverse=True)
            else:
                relevant = [l[1] for l in levels if l[0] == 'resistance' and l[1] > current_price]
                relevant.sort()
            
            return relevant[:3]
        except:
            return []
    
    def _calculate_scaled_entries(self, side: str, current_price: float, total_qty: float, vol_percentile: float) -> list:
        """Calculate scaled entry orders based on volatility."""
        entries = []
        
        ob = self._get_order_book_spread()
        if ob:
            spread_bps = ob['spread_bps']
            imbalance = ob['imbalance']
            print(f"  >> Order Book: Spread {spread_bps:.1f}bps | Imbalance {imbalance:+.2f}")
            
            if spread_bps > self.max_spread_bps:
                print(f"  >> ⚠️ Wide spread ({spread_bps:.1f}bps) - using MARKET order")
                return []
        
        sr_levels = self._find_support_resistance(side, current_price)
        if sr_levels:
            print(f"  >> S/R Levels: {', '.join([f'${l:.0f}' for l in sr_levels[:3]])}")
        
        vol_factor = vol_percentile / 100.0
        adjusted_offsets = [o * (0.5 + vol_factor) for o in self.tranche_offsets]
        
        min_notional = 100
        min_qty_per_tranche = min_notional / (current_price * self.leverage)
        
        for i, (offset, weight) in enumerate(zip(adjusted_offsets, self.tranche_weights)):
            qty = round(total_qty * weight, 3)
            if qty < 0.001 or qty < min_qty_per_tranche:
                continue
            
            if side == 'BUY':
                base_price = current_price * (1 - offset)
                if sr_levels and i < len(sr_levels):
                    sr_price = sr_levels[i] * 1.001
                    if sr_price < current_price and sr_price > base_price:
                        base_price = sr_price
                price = round(base_price, 1)
            else:
                base_price = current_price * (1 + offset)
                if sr_levels and i < len(sr_levels):
                    sr_price = sr_levels[i] * 0.999
                    if sr_price > current_price and sr_price < base_price:
                        base_price = sr_price
                price = round(base_price, 1)
            
            actual_offset = abs(price - current_price) / current_price
            entries.append((price, qty, actual_offset * 100))
        
        return entries
    
    def _calculate_adaptive_timeout(self, vol_percentile: float) -> int:
        vol_factor = vol_percentile / 100.0
        timeout = self.limit_timeout_max - (self.limit_timeout_max - self.limit_timeout_base) * vol_factor
        return int(timeout)
    
    def _place_entry_order(self, side, position_side, quantity, current_price):
        """Place entry order with smart order type selection."""
        import time
        
        volatility = self.last_quant.get('volatility_regime', 'MEDIUM') if self.last_quant else 'MEDIUM'
        vol_percentile = self.last_quant.get('vol_percentile', 50) if self.last_quant else 50
        signal_strength = abs(self.last_quant.get('signal_score', 0)) if self.last_quant else 0
        
        use_scaled = self.use_limit_orders and self.scaled_entry_enabled
        
        if volatility in ['HIGH', 'EXTREME']:
            print(f"  >> ⚡ {volatility} volatility → MARKET order (fast execution)")
            use_scaled = False
        elif signal_strength > 0.5:
            print(f"  >> 🎯 Strong signal ({signal_strength:.2f}) → MARKET order (high conviction)")
            use_scaled = False
        
        if use_scaled:
            entries = self._calculate_scaled_entries(side, current_price, quantity, vol_percentile)
            
            if not entries:
                use_scaled = False
            else:
                print(f"\n  >> 📊 SCALED ENTRY ({len(entries)} tranches):")
                for i, (price, qty, offset) in enumerate(entries):
                    print(f"     T{i+1}: {qty:.3f} BTC @ ${price:.1f} ({offset:.2f}% offset)")
        
        if use_scaled and entries:
            timeout = self._calculate_adaptive_timeout(vol_percentile)
            
            if self.entry_mode == 'auto':
                if signal_strength >= self.aggressive_signal_threshold:
                    effective_mode = 'aggressive'
                    mode_str = f"🏃 AUTO→AGGRESSIVE (signal {signal_strength:.2f} ≥ {self.aggressive_signal_threshold})"
                else:
                    effective_mode = 'patient'
                    mode_str = f"🧘 AUTO→PATIENT (signal {signal_strength:.2f} < {self.aggressive_signal_threshold})"
            else:
                effective_mode = self.entry_mode
                mode_str = "🏃 AGGRESSIVE" if effective_mode == 'aggressive' else "🧘 PATIENT"
            
            print(f"  >> Mode: {mode_str} | Timeout: {timeout}s")
            
            order_ids = []
            try:
                for price, qty, _ in entries:
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=side,
                        positionSide=position_side,
                        type='LIMIT',
                        timeInForce='GTC',
                        price=str(price),
                        quantity=str(qty)
                    )
                    order_ids.append(order['orderId'])
                
                print(f"  >> Placed {len(order_ids)} limit orders")
                self._active_orders = order_ids.copy()
                
                wait_interval = 5
                elapsed = 0
                total_filled = 0
                total_value = 0
                
                while elapsed < timeout:
                    time.sleep(wait_interval)
                    elapsed += wait_interval
                    
                    filled_this_round = 0
                    pending_count = 0
                    
                    for oid in order_ids[:]:
                        try:
                            status_resp = self.client.futures_get_order(symbol=self.symbol, orderId=oid)
                            status = status_resp.get('status', '')
                            exec_qty = float(status_resp.get('executedQty', 0))
                            
                            if status == 'FILLED':
                                avg_px = float(status_resp.get('avgPrice', 0))
                                if exec_qty > 0 and avg_px > 0:
                                    total_filled += exec_qty
                                    total_value += exec_qty * avg_px
                                order_ids.remove(oid)
                                filled_this_round += exec_qty
                            elif status in ['CANCELED', 'EXPIRED', 'REJECTED']:
                                order_ids.remove(oid)
                            else:
                                pending_count += 1
                        except:
                            pass
                    
                    if filled_this_round > 0:
                        print(f"  >> ✅ Filled {filled_this_round:.3f} BTC (total: {total_filled:.3f})")
                        
                        if effective_mode == 'aggressive' and total_filled >= quantity * self.aggressive_fill_threshold:
                            print(f"  >> 🏃 AGGRESSIVE MODE: {total_filled/quantity*100:.0f}% filled - canceling remaining orders")
                            for oid in order_ids:
                                try:
                                    self.client.futures_cancel_order(symbol=self.symbol, orderId=oid)
                                except:
                                    pass
                            order_ids.clear()
                            break
                    
                    if not order_ids:
                        break
                    
                    try:
                        now_price = float(self.client.futures_symbol_ticker(symbol=self.symbol)['price'])
                        price_change = (now_price - current_price) / current_price
                        
                        adverse = (side == 'BUY' and price_change < -self.adverse_move_cancel) or \
                                  (side == 'SELL' and price_change > self.adverse_move_cancel)
                        
                        if adverse:
                            print(f"  >> ⚠️ Adverse move {price_change*100:+.2f}% - canceling unfilled orders")
                            for oid in order_ids:
                                try:
                                    self.client.futures_cancel_order(symbol=self.symbol, orderId=oid)
                                except:
                                    pass
                            break
                        
                        favorable = (side == 'BUY' and price_change > 0.002) or \
                                    (side == 'SELL' and price_change < -0.002)
                        
                        if favorable and total_filled == 0:
                            print(f"  >> 🚀 Price moving favorably ({price_change*100:+.2f}%) - chasing with MARKET")
                            for oid in order_ids:
                                try:
                                    self.client.futures_cancel_order(symbol=self.symbol, orderId=oid)
                                except:
                                    pass
                            
                            remaining_qty = quantity - total_filled
                            if remaining_qty > 0.001:
                                chase_order = self.client.futures_create_order(
                                    symbol=self.symbol,
                                    side=side,
                                    positionSide=position_side,
                                    type='MARKET',
                                    quantity=str(round(remaining_qty, 3))
                                )
                                chase_qty = float(chase_order.get('executedQty', 0) or remaining_qty)
                                chase_price = float(chase_order.get('avgPrice', 0) or now_price)
                                total_filled += chase_qty
                                total_value += chase_qty * chase_price
                                print(f"  >> CHASE FILLED: {chase_qty:.3f} BTC @ ${chase_price:.2f}")
                            break
                        
                        print(f"  >> Waiting ({elapsed}s/{timeout}s) | Filled: {total_filled:.3f} | Pending: {pending_count} | Price: ${now_price:.1f} ({price_change*100:+.2f}%)")
                    except:
                        print(f"  >> Waiting ({elapsed}s/{timeout}s) | Filled: {total_filled:.3f} | Pending: {pending_count}")
                
                for oid in order_ids:
                    try:
                        self.client.futures_cancel_order(symbol=self.symbol, orderId=oid)
                    except:
                        pass
                
                self._active_orders = []
                
                if total_filled >= quantity * 0.5:
                    avg_price = total_value / total_filled if total_filled > 0 else current_price
                    savings = (current_price - avg_price) / current_price * 100 if side == 'BUY' else (avg_price - current_price) / current_price * 100
                    print(f"  >> 📈 SCALED ENTRY COMPLETE: {total_filled:.3f} BTC @ ${avg_price:.2f} (saved {savings:.2f}%)")
                    return total_filled, avg_price
                elif total_filled > 0:
                    avg_price = total_value / total_filled
                    remaining = quantity - total_filled
                    print(f"  >> Partial fill ({total_filled:.3f}), filling remaining {remaining:.3f} with MARKET...")
                    
                    try:
                        mkt_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=side,
                            positionSide=position_side,
                            type='MARKET',
                            quantity=str(round(remaining, 3))
                        )
                        mkt_qty = float(mkt_order.get('executedQty', 0) or remaining)
                        mkt_price = float(mkt_order.get('avgPrice', 0) or current_price)
                        
                        final_qty = total_filled + mkt_qty
                        final_value = total_value + (mkt_qty * mkt_price)
                        final_avg = final_value / final_qty
                        
                        print(f"  >> COMBINED FILL: {final_qty:.3f} BTC @ ${final_avg:.2f}")
                        return final_qty, final_avg
                    except Exception as e:
                        print(f"  >> Market fill failed: {e}")
                        return total_filled, avg_price
                else:
                    print(f"  >> No fills from scaled entry, falling back to MARKET")
                    
            except Exception as e:
                print(f"  >> Scaled entry failed: {e}")
                for oid in order_ids:
                    try:
                        self.client.futures_cancel_order(symbol=self.symbol, orderId=oid)
                    except:
                        pass
        
        print(f"  >> Placing MARKET order...")
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                positionSide=position_side,
                type='MARKET',
                quantity=str(quantity)
            )
            filled_qty = float(order.get('executedQty', 0) or order.get('cumQty', 0) or quantity)
            avg_price = float(order.get('avgPrice', 0) or order.get('price', 0) or current_price)
            if filled_qty == 0:
                filled_qty = quantity
            if avg_price == 0:
                avg_price = current_price
            print(f"  >> MARKET FILLED: {filled_qty:.3f} BTC @ ${avg_price:.2f}")
            return filled_qty, avg_price
        except Exception as e:
            print(f"  >> Market order failed: {e}")
            return None, None
    
    def _open_long(self, price):
        if self.usdt_balance < 10:
            print("  >> Insufficient balance")
            return
        
        stats = self.learning_engine.get_stats()
        kelly_fraction = stats.get('kelly_fraction', 0.5)
        if stats.get('total', 0) >= 10:
            print(f"  >> Kelly confidence: {kelly_fraction:.0%} (EV: {stats.get('expectancy', 0):.4f}, MDD: {stats.get('max_drawdown', 0):.1f}%)")
        
        margin_amount = self.usdt_balance * self.position_size_pct
        notional_value = margin_amount * self.leverage
        
        MIN_NOTIONAL = 120
        if notional_value < MIN_NOTIONAL:
            notional_value = MIN_NOTIONAL
            margin_amount = notional_value / self.leverage
            if margin_amount > self.usdt_balance:
                print(f"  >> Insufficient balance for min ${MIN_NOTIONAL} notional (need ${margin_amount:.2f})")
                return
        
        quantity = round(notional_value / price, 3)
        
        actual_notional = quantity * price
        if actual_notional < 100:
            quantity = round(100 / price + 0.001, 3)
        
        if quantity < 0.001:
            print(f"  >> Quantity too small: {quantity:.3f} BTC")
            return
        
        print(f"  >> LONG: {quantity:.3f} BTC @ ${price:.2f} ({self.leverage}x)")
        print(f"     Margin: ${margin_amount:.2f} | Notional: ${notional_value:.2f}")
        
        if not self.paper_trading:
            filled_qty, avg_price = self._place_entry_order(
                side='BUY',
                position_side='LONG',
                quantity=quantity,
                current_price=price
            )
            if filled_qty is None:
                return
            quantity = filled_qty
            price = avg_price
        
        self.position = 'LONG'
        self.entry_price = price
        self.quantity = quantity
        self._save_state()
        
        self.trade_counter += 1
        self.current_trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.trade_counter}"
        self.learning_engine.record_entry(self.current_trade_id, {
            'entry_price': price,
            'side': 'LONG',
            'signals': {
                'ml_score': getattr(self, '_last_ml_score', 0),
                'ml_confidence': getattr(self, '_last_ml_confidence', 0.5),
                'news_score': self.last_news.get('score', 0) if self.last_news else 0,
                'wisdom_score': self.last_wisdom.get('score', 0) if self.last_wisdom else 0,
                'wisdom_master': self.last_wisdom.get('which_master_speaks', '') if self.last_wisdom else '',
                'quant_score': self.last_quant.get('score', 0) if self.last_quant else 0,
                'htf_bias': self.last_quant.get('htf_bias', 0) if self.last_quant else 0,
                'final_vote': getattr(self, '_last_final_vote', 0)
            },
            'market_state': {
                'rsi': getattr(self, '_last_rsi', 50),
                'bb_position': getattr(self, '_last_bb_position', 0.5),
                'volume_ratio': getattr(self, '_last_volume_ratio', 1.0),
                'funding_rate': getattr(self, '_last_funding_rate', 0),
                'volatility_percentile': self.last_quant.get('vol_percentile', 50) if self.last_quant else 50,
                'macd': getattr(self, '_last_macd', 0),
                'momentum_1h': getattr(self, '_last_momentum_1h', 0)
            }
        })
        
        self._update_dashboard(
            position={'side': 'LONG', 'quantity': quantity, 'entryPrice': price, 'pnl': 0, 'pnlPercent': 0}
        )
        self._add_dashboard_log(f"OPENED LONG: {quantity:.4f} BTC @ ${price:.2f}")
        
        if not self.paper_trading:
            self._place_sl_tp_orders('LONG', price, quantity)
    
    def _open_short(self, price):
        if self.usdt_balance < 10:
            print("  >> Insufficient balance")
            return
        
        stats = self.learning_engine.get_stats()
        kelly_fraction = stats.get('kelly_fraction', 0.5)
        if stats.get('total', 0) >= 10:
            print(f"  >> Kelly confidence: {kelly_fraction:.0%} (EV: {stats.get('expectancy', 0):.4f}, MDD: {stats.get('max_drawdown', 0):.1f}%)")
        
        margin_amount = self.usdt_balance * self.position_size_pct
        notional_value = margin_amount * self.leverage
        
        MIN_NOTIONAL = 120
        if notional_value < MIN_NOTIONAL:
            notional_value = MIN_NOTIONAL
            margin_amount = notional_value / self.leverage
            if margin_amount > self.usdt_balance:
                print(f"  >> Insufficient balance for min ${MIN_NOTIONAL} notional (need ${margin_amount:.2f})")
                return
        
        quantity = round(notional_value / price, 3)
        
        actual_notional = quantity * price
        if actual_notional < 100:
            quantity = round(100 / price + 0.001, 3)
        
        if quantity < 0.001:
            print(f"  >> Quantity too small: {quantity:.3f} BTC")
            return
        
        print(f"  >> SHORT: {quantity:.3f} BTC @ ${price:.2f} ({self.leverage}x)")
        print(f"     Margin: ${margin_amount:.2f} | Notional: ${notional_value:.2f}")
        
        if not self.paper_trading:
            filled_qty, avg_price = self._place_entry_order(
                side='SELL',
                position_side='SHORT',
                quantity=quantity,
                current_price=price
            )
            if filled_qty is None:
                return
            quantity = filled_qty
            price = avg_price
        
        self.position = 'SHORT'
        self.entry_price = price
        self.quantity = quantity
        self._save_state()
        
        self.trade_counter += 1
        self.current_trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.trade_counter}"
        self.learning_engine.record_entry(self.current_trade_id, {
            'entry_price': price,
            'side': 'SHORT',
            'signals': {
                'ml_score': getattr(self, '_last_ml_score', 0),
                'ml_confidence': getattr(self, '_last_ml_confidence', 0.5),
                'news_score': self.last_news.get('score', 0) if self.last_news else 0,
                'wisdom_score': self.last_wisdom.get('score', 0) if self.last_wisdom else 0,
                'wisdom_master': self.last_wisdom.get('which_master_speaks', '') if self.last_wisdom else '',
                'quant_score': self.last_quant.get('score', 0) if self.last_quant else 0,
                'htf_bias': self.last_quant.get('htf_bias', 0) if self.last_quant else 0,
                'final_vote': getattr(self, '_last_final_vote', 0)
            },
            'market_state': {
                'rsi': getattr(self, '_last_rsi', 50),
                'bb_position': getattr(self, '_last_bb_position', 0.5),
                'volume_ratio': getattr(self, '_last_volume_ratio', 1.0),
                'funding_rate': getattr(self, '_last_funding_rate', 0),
                'volatility_percentile': self.last_quant.get('vol_percentile', 50) if self.last_quant else 50,
                'macd': getattr(self, '_last_macd', 0),
                'momentum_1h': getattr(self, '_last_momentum_1h', 0)
            }
        })
        
        self._update_dashboard(
            position={'side': 'SHORT', 'quantity': quantity, 'entryPrice': price, 'pnl': 0, 'pnlPercent': 0}
        )
        self._add_dashboard_log(f"OPENED SHORT: {quantity:.4f} BTC @ ${price:.2f}")
        
        if not self.paper_trading:
            self._place_sl_tp_orders('SHORT', price, quantity)
    
    def _calculate_dynamic_tp(self):
        """Calculate dynamic take profit based on market conditions."""
        base_tp = self.take_profit_min
        
        if self.last_wisdom:
            rr_ratio = self.last_wisdom.get('risk_reward_ratio', 1.5)
            rr_factor = min(rr_ratio / 3.0, 1.0)
            base_tp += (self.take_profit_max - self.take_profit_min) * rr_factor * 0.4
        
        if self.last_quant:
            signal_strength = abs(self.last_quant.get('signal_score', 0))
            if signal_strength > 0.5:
                base_tp += (self.take_profit_max - self.take_profit_min) * 0.3
        
        if self.last_quant:
            volatility = self.last_quant.get('volatility_regime', 'MEDIUM')
            if volatility == 'LOW':
                base_tp += (self.take_profit_max - self.take_profit_min) * 0.2
            elif volatility == 'EXTREME':
                base_tp = max(base_tp * 0.8, self.take_profit_min)
        
        dynamic_tp = max(self.take_profit_min, min(base_tp, self.take_profit_max))
        
        print(f"  >> Dynamic TP: {dynamic_tp*100:.2f}% (range: {self.take_profit_min*100:.1f}%-{self.take_profit_max*100:.1f}%)")
        return dynamic_tp
    
    def _place_sl_tp_orders(self, position_side, entry_price, quantity):
        """Place stop loss and take profit orders."""
        try:
            dynamic_tp_pct = self._calculate_dynamic_tp()
            
            if position_side == 'LONG':
                sl_price = round(entry_price * (1 - self.stop_loss_pct), 1)
                tp_price = round(entry_price * (1 + dynamic_tp_pct), 1)
                sl_side = 'SELL'
                tp_side = 'SELL'
            else:
                sl_price = round(entry_price * (1 + self.stop_loss_pct), 1)
                tp_price = round(entry_price * (1 - dynamic_tp_pct), 1)
                sl_side = 'BUY'
                tp_side = 'BUY'
            
            self.take_profit_pct = dynamic_tp_pct
            
            sl_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=sl_side,
                positionSide=position_side,
                type='STOP_MARKET',
                stopPrice=str(sl_price),
                quantity=str(quantity),
                workingType='MARK_PRICE'
            )
            sl_id = sl_order.get('orderId') or sl_order.get('clientOrderId') or 'placed'
            print(f"  >> SL Order: {sl_id} @ ${sl_price:.1f}")
            
            tp_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=tp_side,
                positionSide=position_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=str(tp_price),
                quantity=str(quantity),
                workingType='MARK_PRICE'
            )
            tp_id = tp_order.get('orderId') or tp_order.get('clientOrderId') or 'placed'
            print(f"  >> TP Order: {tp_id} @ ${tp_price:.1f}")
            
        except Exception as e:
            print(f"  >> SL/TP order failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _manage_position(self, current_price, signal=None, strength=0):
        if self.position == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            tp = self.entry_price * (1 + self.take_profit_pct)
            sl = self.entry_price * (1 - self.stop_loss_pct)
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
            tp = self.entry_price * (1 - self.take_profit_pct)
            sl = self.entry_price * (1 + self.stop_loss_pct)
        
        if self.trailing_stop_enabled and pnl_pct >= self.trailing_stop_activation * 100:
            if self.position == 'LONG':
                new_trailing = current_price * (1 - self.trailing_stop_distance)
                if self.current_trailing_stop is None or new_trailing > self.current_trailing_stop:
                    self.current_trailing_stop = new_trailing
                    print(f"  >> 📈 TRAILING STOP updated: ${self.current_trailing_stop:.2f}")
                sl = max(sl, self.current_trailing_stop)
            else:
                new_trailing = current_price * (1 + self.trailing_stop_distance)
                if self.current_trailing_stop is None or new_trailing < self.current_trailing_stop:
                    self.current_trailing_stop = new_trailing
                    print(f"  >> 📈 TRAILING STOP updated: ${self.current_trailing_stop:.2f}")
                sl = min(sl, self.current_trailing_stop)
        
        leveraged_pnl = pnl_pct * self.leverage
        trail_str = f" | Trail ${self.current_trailing_stop:.2f}" if self.current_trailing_stop else ""
        print(f"  {self.position}: Entry ${self.entry_price:.2f} | TP ${tp:.2f} | SL ${sl:.2f}{trail_str} | PnL: {pnl_pct:+.2f}% ({leveraged_pnl:+.1f}% w/ {self.leverage}x)")
        self._show_position_costs()
        
        self._update_dashboard(
            currentPrice=current_price,
            position={
                'side': self.position,
                'quantity': self.quantity,
                'entryPrice': self.entry_price,
                'pnl': leveraged_pnl,
                'pnlPercent': pnl_pct
            }
        )
        
        if self.signal_based_exit and signal and pnl_pct > self.profit_take_threshold * 100:
            opposite_signal = (self.position == 'LONG' and signal == 'SHORT') or \
                              (self.position == 'SHORT' and signal == 'LONG')
            if opposite_signal and strength >= self.reversal_threshold:
                print(f"  >> 🔄 SIGNAL-BASED EXIT: {signal} signal with strength {strength:.2f} while in profit")
                if self.position == 'LONG':
                    self._close_long(current_price, f"SIGNAL EXIT ({signal})")
                else:
                    self._close_short(current_price, f"SIGNAL EXIT ({signal})")
                return
        
        if self.position == 'LONG':
            if current_price >= tp:
                self._close_long(current_price, "TAKE PROFIT")
            elif current_price <= sl:
                self._close_long(current_price, "TRAILING STOP" if self.current_trailing_stop and sl == self.current_trailing_stop else "STOP LOSS")
        else:
            if current_price <= tp:
                self._close_short(current_price, "TAKE PROFIT")
            elif current_price >= sl:
                self._close_short(current_price, "TRAILING STOP" if self.current_trailing_stop and sl == self.current_trailing_stop else "STOP LOSS")
    
    def _cancel_open_orders(self):
        try:
            result = self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            print(f"  >> Cancelled open orders")
        except Exception as e:
            print(f"  >> Cancel orders failed: {e}")
    
    def _close_long(self, price, reason):
        pnl_pct = (price - self.entry_price) / self.entry_price * 100
        leveraged_pnl = pnl_pct * self.leverage
        print(f"  >> CLOSE LONG @ ${price:.2f} | {reason} | PnL: {pnl_pct:+.2f}% ({leveraged_pnl:+.1f}% with {self.leverage}x)")
        
        if not self.paper_trading:
            self._cancel_open_orders()
            
            actual_pos, actual_qty, _ = self._get_futures_position()
            if actual_pos != 'LONG' or actual_qty < 0.001:
                print(f"  >> Position already closed by SL/TP order")
            else:
                try:
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side='SELL',
                        positionSide='LONG',
                        type='MARKET',
                        quantity=actual_qty
                    )
                    print(f"  >> Closed: Order {order['orderId']}")
                except Exception as e:
                    print(f"  >> Failed: {e}")
        
        if hasattr(self, 'current_trade_id') and self.current_trade_id:
            self.learning_engine.record_exit(self.current_trade_id, price, pnl_pct)
            self.current_trade_id = None
        
        self.position = None
        self.entry_price = None
        self.quantity = 0
        self.current_trailing_stop = None
        self._save_state()
        self._update_dashboard(position={'side': None, 'quantity': 0, 'entryPrice': 0, 'pnl': 0, 'pnlPercent': 0})
        self._add_dashboard_log(f"CLOSED LONG: {reason} | PnL: {leveraged_pnl:+.1f}%")
    
    def _close_short(self, price, reason):
        pnl_pct = (self.entry_price - price) / self.entry_price * 100
        leveraged_pnl = pnl_pct * self.leverage
        print(f"  >> CLOSE SHORT @ ${price:.2f} | {reason} | PnL: {pnl_pct:+.2f}% ({leveraged_pnl:+.1f}% with {self.leverage}x)")
        
        if not self.paper_trading:
            self._cancel_open_orders()
            
            actual_pos, actual_qty, _ = self._get_futures_position()
            if actual_pos != 'SHORT' or actual_qty < 0.001:
                print(f"  >> Position already closed by SL/TP order")
            else:
                try:
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side='BUY',
                        positionSide='SHORT',
                        type='MARKET',
                        quantity=actual_qty
                    )
                    print(f"  >> Closed: Order {order['orderId']}")
                except Exception as e:
                    print(f"  >> Failed: {e}")
        
        if hasattr(self, 'current_trade_id') and self.current_trade_id:
            self.learning_engine.record_exit(self.current_trade_id, price, pnl_pct)
            self.current_trade_id = None
        
        self.position = None
        self.entry_price = None
        self.quantity = 0
        self.current_trailing_stop = None
        self._save_state()
        self._update_dashboard(position={'side': None, 'quantity': 0, 'entryPrice': 0, 'pnl': 0, 'pnlPercent': 0})
        self._add_dashboard_log(f"CLOSED SHORT: {reason} | PnL: {leveraged_pnl:+.1f}%")
    
    def _update_thinking(self, stage: str, status: str, result: dict = None, active: bool = True, final_decision: dict = None):
        if DASHBOARD_AVAILABLE:
            try:
                thinking_state = {
                    'active': active,
                    'stage': stage,
                    'stages': getattr(self, '_thinking_stages', {
                        'ml': {'status': 'pending', 'result': None},
                        'news': {'status': 'pending', 'result': None},
                        'wisdom': {'status': 'pending', 'result': None},
                        'quant': {'status': 'pending', 'result': None},
                        'whale': {'status': 'pending', 'result': None},
                        'overlord': {'status': 'pending', 'result': None}
                    }),
                    'finalDecision': final_decision
                }
                thinking_state['stages'][stage] = {'status': status, 'result': result}
                self._thinking_stages = thinking_state['stages']
                dashboard_server.update_state(thinking=thinking_state)
            except:
                pass
    
    def _reset_thinking(self):
        if DASHBOARD_AVAILABLE:
            try:
                self._thinking_stages = {
                    'ml': {'status': 'pending', 'result': None},
                    'news': {'status': 'pending', 'result': None},
                    'wisdom': {'status': 'pending', 'result': None},
                    'quant': {'status': 'pending', 'result': None},
                    'whale': {'status': 'pending', 'result': None},
                    'overlord': {'status': 'pending', 'result': None}
                }
                dashboard_server.update_state(thinking={
                    'active': False,
                    'stage': '',
                    'stages': self._thinking_stages,
                    'finalDecision': None
                })
            except:
                pass
    
    def _update_dashboard(self, **kwargs):
        if DASHBOARD_AVAILABLE:
            try:
                dashboard_server.update_state(**kwargs)
                screenshots = dashboard_server.get_latest_screenshots()
                dashboard_server.update_state(screenshots=screenshots)
            except Exception as e:
                pass
    
    def _add_dashboard_log(self, message):
        if DASHBOARD_AVAILABLE:
            try:
                dashboard_server.add_log(message)
            except:
                pass
    
    def _get_learning_stats(self):
        try:
            stats = self.learning_engine.get_stats()
            ml_status = self.learning_engine.get_ml_retrain_status()
            closed_trades = ml_status.get('total_trades', 0)
            all_trades = ml_status.get('all_trades', 0)
            min_required = ml_status.get('min_required', 50)
            return {
                'totalTrades': closed_trades,
                'allTrades': all_trades,
                'minForRetrain': min_required,
                'tradesUntilRetrain': max(0, min_required - closed_trades),
                'winRate': stats.get('win_rate', 0),
                'lastRetrain': ml_status.get('last_retrain'),
                'readyToRetrain': ml_status.get('ready_to_retrain', False),
                'kellyFraction': stats.get('kelly_fraction', 0.5),
                'expectancy': stats.get('expectancy', 0),
                'maxDrawdown': stats.get('max_drawdown', 0),
                'sharpeRatio': stats.get('sharpe_ratio', 0),
                'profitFactor': stats.get('profit_factor', 0)
            }
        except Exception as e:
            return {
                'totalTrades': 0,
                'allTrades': 0,
                'minForRetrain': 50,
                'tradesUntilRetrain': 50,
                'winRate': 0,
                'lastRetrain': None,
                'readyToRetrain': False,
                'kellyFraction': 0.5,
                'expectancy': 0,
                'maxDrawdown': 0,
                'sharpeRatio': 0,
                'profitFactor': 0
            }
    
    def _update_position_pnl(self):
        if self.paper_trading or not self.position:
            return
        try:
            current_price = float(self.client.futures_symbol_ticker(symbol=self.symbol)['price'])
            if self.position == 'LONG':
                pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            else:
                pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
            leveraged_pnl = pnl_pct * self.leverage
            
            self._update_dashboard(
                currentPrice=current_price,
                position={
                    'side': self.position,
                    'quantity': self.quantity,
                    'entryPrice': self.entry_price,
                    'pnl': leveraged_pnl,
                    'pnlPercent': pnl_pct
                }
            )
        except Exception as e:
            pass
    
    def run(self):
        if DASHBOARD_AVAILABLE:
            dashboard_server.set_bot_instance(self)
            dashboard_thread = threading.Thread(target=dashboard_server.run_server, daemon=True)
            dashboard_thread.start()
            print("📊 Dashboard: http://localhost:3000 (API: http://localhost:5000)")
        
        print("\nStarting bot... Press Ctrl+C to stop.")
        print("Full Pipeline (ML + News + Wisdom + Quant): Every 5 minutes")
        print("Trailing Stop + Signal-Based Exit: ENABLED\n")
        
        self.execute_strategy()
        
        schedule.every(5).minutes.do(self.execute_strategy)
        
        schedule.every(30).minutes.do(self._sync_time)
        
        schedule.every(1).minutes.do(self._update_position_pnl)
        
        while self.running:
            schedule.run_pending()
            time.sleep(1)
        
        print("\n" + "="*60)
        print("BOT SHUTDOWN")
        print("="*60)
        self._save_state()

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║      ADVANCED BOT - News + ML Ensemble                     ║
    ║      Qwen LLM + Transformer + XGBoost                      ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    bot = AdvancedBot()
    bot.run()