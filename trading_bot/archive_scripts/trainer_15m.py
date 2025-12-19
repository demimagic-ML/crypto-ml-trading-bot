"""15-minute interval trainer (archived)."""
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import pickle
from ta.momentum import RSIIndicator
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

client = Client(API_KEY, API_SECRET)

def get_historical_data(symbol, interval, start_time):
    print(f"Fetching {symbol} data from {start_time}...")
    klines = client.get_historical_klines(symbol, interval, start_time)
    data = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    return data

symbol = 'BTCUSDC'
interval = Client.KLINE_INTERVAL_15MINUTE
start_time = '1 year ago UTC'
TIME_STEPS = 60

print(f"""
{'='*50}
IMPROVED TRAINER
{'='*50}
Symbol: {symbol}
Interval: 15m (less noise than 1m)
Time Steps: {TIME_STEPS} (sees {TIME_STEPS * 15 / 60:.0f} hours of history)
Period: {start_time}
{'='*50}
""")