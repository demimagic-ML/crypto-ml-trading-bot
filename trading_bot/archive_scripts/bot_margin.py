"""Margin trading bot."""
import os
import sys
import pickle
import time
import signal
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator
import schedule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from binance.client import Client

class MarginBot:
    """Margin trading bot with leverage support."""
    
    def __init__(self):
        self.symbol = 'BTCUSDC'
        self.interval = Client.KLINE_INTERVAL_15MINUTE
        self.time_steps = 60
        
        self.paper_trading = os.getenv('TRADING_MODE', 'paper').lower() == 'paper'
        
        self.position = None
        self.entry_price = None
        self.quantity = 0
        
        self.take_profit_pct = 0.01
        self.stop_loss_pct = 0.005
        
        self.leverage = 3
        
        self.position_size_pct = 0.90
        
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.client = Client(api_key, api_secret, requests_params={"timeout": 30})
        
        self._sync_time()
        
        self._load_model()
        
        self.usdc_balance = self._get_margin_balance('USDC')
        self.btc_balance = self._get_margin_balance('BTC')
        
        self.state_file = 'state/margin_bot_state.json'
        self._load_state()
        
        self._check_and_fix_open_positions()
        
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        mode_str = 'PAPER' if self.paper_trading else '🔴 LIVE'
        print(f"\n{'='*50}")
        print(f"MARGIN BOT - LONG/SHORT")
        print(f"{'='*50}")
        print(f"Mode: {mode_str}")
        print(f"Symbol: {self.symbol}")
        print(f"Interval: 15m")
        print(f"USDC Balance: ${self.usdc_balance:.2f}")
        print(f"BTC Balance: {self.btc_balance:.6f}")
        print(f"Take Profit: {self.take_profit_pct*100}%")
        print(f"Stop Loss: {self.stop_loss_pct*100}%")
        print(f"Leverage: {self.leverage}x")
        print(f"{'='*50}\n")
    
    def _signal_handler(self, signum, frame):
        print("\nShutdown signal received...")
        self.running = False
    
    def _sync_time(self):
        try:
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            self.client.timestamp_offset = server_time['serverTime'] - local_time
            print(f"Time synchronized. Offset: {self.client.timestamp_offset}ms")
        except Exception as e:
            print(f"Time sync failed: {e}")
    
    def _get_margin_balance(self, asset):
        if self.paper_trading:
            return 50.0 if asset == 'USDT' else 0.0
        try:
            account = self.client.get_margin_account()
            for item in account['userAssets']:
                if item['asset'] == asset:
                    return float(item['free'])
            return 0.0
        except Exception as e:
            print(f"Error getting margin balance: {e}")
            return 0.0
    
    def _load_model(self):
        try:
            self.model = load_model('models/model_15m.h5')
            
            with open('models/feature_scaler_15m.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            
            with open('models/close_scaler_15m.pkl', 'rb') as f:
                self.close_scaler = pickle.load(f)
            
            print("Model and scalers loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)
    
    def _load_state(self):
        os.makedirs('state', exist_ok=True)
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.position = state.get('position')
            self.entry_price = state.get('entry_price')
            self.quantity = state.get('quantity', 0)
            print(f"Loaded state: Position={self.position}, Qty={self.quantity}")
        except:
            print("No saved state found, starting fresh")
    
    def _check_and_fix_open_positions(self):
        if self.paper_trading:
            return
            
        try:
            account = self.client.get_margin_account()
            btc_free = 0
            btc_borrowed = 0
            btc_interest = 0
            
            for asset in account['userAssets']:
                if asset['asset'] == 'BTC':
                    btc_free = float(asset['free'])
                    btc_borrowed = float(asset['borrowed'])
                    btc_interest = float(asset['interest'])
                    break
            
            if btc_borrowed > 0:
                print(f"  Found open BTC loan: {btc_borrowed:.8f} BTC")
                print(f"  Interest: {btc_interest:.8f} BTC")
                print(f"  Free BTC: {btc_free:.8f} BTC")
                
                repay_amount = min(btc_free, btc_borrowed + btc_interest)
                if repay_amount > 0:
                    print(f"  Repaying {repay_amount:.8f} BTC...")
                    self.client.repay_margin_loan(
                        asset='BTC',
                        amount=repay_amount,
                        isIsolated='FALSE'
                    )
                    print(f"  >> Repaid {repay_amount:.8f} BTC loan")
                    
                    self.position = None
                    self.entry_price = None
                    self.quantity = 0
                    self._save_state()
                    
                    self.usdc_balance = self._get_margin_balance('USDC')
                    self.btc_balance = self._get_margin_balance('BTC')
            
            elif self.position == 'SHORT' and btc_borrowed == 0:
                print(f"  State shows SHORT but no borrowed BTC - clearing stale state")
                self.position = None
                self.entry_price = None
                self.quantity = 0
                self._save_state()
                
        except Exception as e:
            print(f"  Error checking positions: {e}")
    
    def _save_state(self):
        state = {
            'position': self.position,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'last_save': datetime.utcnow().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_market_data(self, limit=100):
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def predict(self, df):
        current_price = float(df['close'].iloc[-1])
        
        df = df.copy()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df = df.dropna()
        
        if len(df) < self.time_steps:
            return current_price, None
        
        df_scaled = df.copy()
        df_scaled[['open', 'high', 'low', 'volume', 'rsi']] = self.feature_scaler.transform(
            df[['open', 'high', 'low', 'volume', 'rsi']]
        )
        
        X = df_scaled[['open', 'high', 'low', 'volume', 'rsi']].tail(self.time_steps).values
        X = X.reshape(1, self.time_steps, 5)
        
        prediction_scaled = self.model.predict(X, verbose=0)
        prediction = self.close_scaler.inverse_transform(prediction_scaled)
        
        return current_price, float(prediction[0][0])
    
    def execute_strategy(self):
        try:
            df = self.get_market_data()
            current_price, prediction = self.predict(df)
            
            if prediction is None:
                print("Not enough data, skipping...")
                return
            
            self.usdc_balance = self._get_margin_balance('USDC')
            
            pred_diff = ((prediction / current_price) - 1) * 100
            signal = "LONG" if prediction > current_price else "SHORT"
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Price: ${current_price:.2f} | "
                  f"Pred: ${prediction:.2f} ({pred_diff:+.2f}%) | "
                  f"Signal: {signal}")
            
            if self.position is None:
                if prediction > current_price:
                    self._open_long(current_price)
                else:
                    self._open_short(current_price)
            
            else:
                self._manage_position(current_price)
            
            self._save_state()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _open_long(self, price):
        if self.usdc_balance < 10:
            print("  >> Insufficient USDC balance")
            return
        
        amount = self.usdc_balance * self.position_size_pct * self.leverage
        quantity = amount / price
        
        quantity = round(quantity, 5)
        
        print(f"  >> LONG: Buy {quantity:.5f} BTC @ ${price:.2f}")
        
        if not self.paper_trading:
            try:
                order = self.client.create_margin_order(
                    symbol=self.symbol,
                    side='BUY',
                    type='MARKET',
                    quoteOrderQty=amount,
                    isIsolated='FALSE'
                )
                quantity = float(order['executedQty'])
                price = float(order['cummulativeQuoteQty']) / quantity
                print(f"  >> Order executed: {order['orderId']}")
            except Exception as e:
                print(f"  >> Order failed: {e}")
                return
        
        self.position = 'LONG'
        self.entry_price = price
        self.quantity = quantity
        print(f"  >> LONG position opened @ ${price:.2f}")
    
    def _open_short(self, price):
        if self.usdc_balance < 10:
            print("  >> Insufficient USDC balance for margin")
            return
        
        amount = self.usdc_balance * self.position_size_pct * self.leverage
        quantity = amount / price
        quantity = round(quantity, 5)
        
        print(f"  >> SHORT: Sell {quantity:.5f} BTC @ ${price:.2f}")
        
        if not self.paper_trading:
            try:
                self.client.create_margin_loan(
                    asset='BTC',
                    amount=quantity,
                    isIsolated='FALSE'
                )
                print(f"  >> Borrowed {quantity:.5f} BTC")
                
                order = self.client.create_margin_order(
                    symbol=self.symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity,
                    isIsolated='FALSE'
                )
                price = float(order['cummulativeQuoteQty']) / float(order['executedQty'])
                print(f"  >> Order executed: {order['orderId']}")
            except Exception as e:
                print(f"  >> Order failed: {e}")
                return
        
        self.position = 'SHORT'
        self.entry_price = price
        self.quantity = quantity
        print(f"  >> SHORT position opened @ ${price:.2f}")
    
    def _manage_position(self, current_price):
        if self.position == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            tp_price = self.entry_price * (1 + self.take_profit_pct)
            sl_price = self.entry_price * (1 - self.stop_loss_pct)
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
            tp_price = self.entry_price * (1 - self.take_profit_pct)
            sl_price = self.entry_price * (1 + self.stop_loss_pct)
        
        print(f"  {self.position}: Entry ${self.entry_price:.2f} | "
              f"TP ${tp_price:.2f} | SL ${sl_price:.2f} | "
              f"PnL: {pnl_pct:+.2f}%")
        
        if self.position == 'LONG':
            if current_price >= tp_price:
                self._close_long(current_price, "TAKE PROFIT")
            elif current_price <= sl_price:
                self._close_long(current_price, "STOP LOSS")
        else:
            if current_price <= tp_price:
                self._close_short(current_price, "TAKE PROFIT")
            elif current_price >= sl_price:
                self._close_short(current_price, "STOP LOSS")
    
    def _close_long(self, price, reason):
        pnl_pct = (price - self.entry_price) / self.entry_price * 100
        pnl_usd = self.quantity * (price - self.entry_price)
        
        print(f"  >> CLOSE LONG: Sell {self.quantity:.5f} BTC @ ${price:.2f}")
        print(f"  >> {reason} | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
        
        if not self.paper_trading:
            try:
                order = self.client.create_margin_order(
                    symbol=self.symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=self.quantity,
                    isIsolated='FALSE'
                )
                print(f"  >> Order executed: {order['orderId']}")
            except Exception as e:
                print(f"  >> Order failed: {e}")
                return
        
        self.position = None
        self.entry_price = None
        self.quantity = 0
        print(f"  >> Position closed")
    
    def _close_short(self, price, reason):
        pnl_pct = (self.entry_price - price) / self.entry_price * 100
        pnl_usd = self.quantity * (self.entry_price - price)
        
        print(f"  >> CLOSE SHORT: Buy {self.quantity:.5f} BTC @ ${price:.2f}")
        print(f"  >> {reason} | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
        
        if not self.paper_trading:
            try:
                order = self.client.create_margin_order(
                    symbol=self.symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=self.quantity,
                    isIsolated='FALSE'
                )
                print(f"  >> Order executed: {order['orderId']}")
                
                account = self.client.get_margin_account()
                btc_free = 0
                btc_borrowed = 0
                for asset in account['userAssets']:
                    if asset['asset'] == 'BTC':
                        btc_free = float(asset['free'])
                        btc_borrowed = float(asset['borrowed'])
                        btc_interest = float(asset['interest'])
                        break
                
                repay_amount = min(btc_free, btc_borrowed + btc_interest)
                if repay_amount > 0:
                    self.client.repay_margin_loan(
                        asset='BTC',
                        amount=repay_amount,
                        isIsolated='FALSE'
                    )
                    print(f"  >> Repaid {repay_amount:.8f} BTC loan")
                
            except Exception as e:
                print(f"  >> Order failed: {e}")
                return
        
        self.position = None
        self.entry_price = None
        self.quantity = 0
        print(f"  >> Position closed")
    
    def run(self):
        print("\nStarting bot... Press Ctrl+C to stop.")
        print("Checking every 15 minutes.\n")
        
        self.execute_strategy()
        
        schedule.every(15).minutes.do(self.execute_strategy)
        schedule.every(30).minutes.do(self._sync_time)
        
        while self.running:
            schedule.run_pending()
            time.sleep(1)
        
        print("\n" + "="*50)
        print("BOT SHUTDOWN")
        print("="*50)
        print(f"Position: {self.position}")
        if self.position:
            print(f"Entry: ${self.entry_price:.2f}, Qty: {self.quantity:.5f}")
        print("="*50)
        self._save_state()

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          MARGIN BOT - LONG/SHORT                          ║
    ║          Cross Margin Trading                             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    bot = MarginBot()
    bot.run()

if __name__ == '__main__':
    main()