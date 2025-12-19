import os
import sys
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

app = Flask(__name__)

bot_instance = None

def set_bot_instance(bot):
    global bot_instance
    bot_instance = bot
CORS(app)

bot_state = {
    'connected': True,
    'mode': 'LIVE',
    'symbol': 'BTCUSDC',
    'balance': 0,
    'position': {
        'side': None,
        'quantity': 0,
        'entryPrice': 0,
        'pnl': 0,
        'pnlPercent': 0
    },
    'signals': {
        'ml': {'direction': 'HOLD', 'confidence': 0, 'prediction': 0},
        'news': {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 0},
        'wisdom': {
            'signal': 'HOLD', 
            'grade': 'C', 
            'master': '', 
            'reasoning': '',
            'specialty': '',
            'style': '',
            'selectionReason': '',
            'fullAnalysis': {},
            'keyLevels': {}
        },
        'quant': {'signal': 'HOLD', 'zScore': 0, 'momentum': 0, 'htfTrend': 'NEUTRAL', 'htfBias': 0}
    },
    'whale': {
        'sentiment': 'NEUTRAL',
        'score': 0,
        'deposits': 0,
        'withdrawals': 0,
        'depositBtc': 0,
        'withdrawalBtc': 0,
        'netFlow': 0,
        'netFlowUsd': 0,
        'reasoning': '',
        'largeTxs': [],
        'hasAlert': False
    },
    'whaleHistory': [],
    'costs': {'commission': 0, 'funding': 0, 'total': 0},
    'screenshots': [],
    'lastUpdate': '',
    'currentPrice': 0,
    'decision': {'action': 'HOLD', 'strength': 0},
    'entryMode': {
        'mode': 'auto',
        'scaledEnabled': True,
        'aggressiveThreshold': 0.35,
        'currentSignalStrength': 0
    },
    'learning': {
        'totalTrades': 0,
        'minForRetrain': 50,
        'tradesUntilRetrain': 50,
        'winRate': 0,
        'lastRetrain': None,
        'readyToRetrain': False
    },
    'overlord': None,
    'thinking': {
        'active': False,
        'stage': '',
        'stages': {
            'ml': {'status': 'pending', 'result': None},
            'news': {'status': 'pending', 'result': None},
            'wisdom': {'status': 'pending', 'result': None},
            'quant': {'status': 'pending', 'result': None},
            'whale': {'status': 'pending', 'result': None},
            'overlord': {'status': 'pending', 'result': None}
        },
        'finalDecision': None
    }
}

whale_history = []
MAX_WHALE_HISTORY = 50

logs_buffer = []
MAX_LOGS = 200

def add_log(message):
    global logs_buffer
    timestamp = datetime.now().strftime('%H:%M:%S')
    logs_buffer.append(f"[{timestamp}] {message}")
    if len(logs_buffer) > MAX_LOGS:
        logs_buffer = logs_buffer[-MAX_LOGS:]

def update_state(**kwargs):
    global bot_state, whale_history
    for key, value in kwargs.items():
        if key in bot_state:
            if isinstance(bot_state[key], dict) and isinstance(value, dict):
                bot_state[key].update(value)
            else:
                bot_state[key] = value
    bot_state['lastUpdate'] = datetime.now().strftime('%H:%M:%S')
    
    if 'whale' in kwargs:
        whale_data = kwargs['whale']
        if whale_data.get('hasAlert') or abs(whale_data.get('netFlow', 0)) > 50:
            history_entry = {
                **whale_data,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            whale_history.insert(0, history_entry)
            if len(whale_history) > MAX_WHALE_HISTORY:
                whale_history = whale_history[:MAX_WHALE_HISTORY]
            bot_state['whaleHistory'] = whale_history

@app.route('/api/status')
def get_status():
    return jsonify(convert_numpy(bot_state))

@app.route('/api/logs')
def get_logs():
    return jsonify({'logs': logs_buffer[-100:]})

@app.route('/api/close-position', methods=['POST'])
def close_position():
    global bot_instance
    
    if bot_instance is None:
        return jsonify({'success': False, 'error': 'Bot not connected'}), 500
    
    try:
        actual_pos, actual_qty, _ = bot_instance._get_futures_position()
        
        if bot_instance.position is None and actual_pos is None:
            return jsonify({'success': False, 'error': 'No position to close'}), 400
        
        if actual_pos is None or actual_qty < 0.001:
            if bot_instance.position is not None:
                add_log(f"🔄 Position already closed on exchange, syncing state...")
                bot_instance.position = None
                bot_instance.entry_price = None
                bot_instance.quantity = 0
                bot_instance._save_state()
            return jsonify({'success': False, 'error': 'Position already closed on exchange'}), 400
        
        try:
            bot_instance._cancel_open_orders()
            add_log("🧹 Cancelled open TP/SL orders")
        except Exception as e:
            add_log(f"⚠️ Error cancelling orders: {e}")
        
        current_price = float(bot_instance.client.futures_symbol_ticker(symbol=bot_instance.symbol)['price'])
        position_side = bot_instance.position
        
        if position_side == 'LONG':
            result = bot_instance._close_long(current_price, reason="Manual close via UI")
        else:
            result = bot_instance._close_short(current_price, reason="Manual close via UI")
        
        add_log(f"🖐️ Manual close: {position_side} @ ${current_price:.2f}")
        
        return jsonify({
            'success': True, 
            'message': f'Closed {position_side} position at ${current_price:.2f}',
            'exitPrice': current_price
        })
        
    except Exception as e:
        add_log(f"❌ Manual close error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/screenshots/<path:filename>')
def get_screenshot(filename):
    screenshots_dir = os.path.join(os.path.dirname(__file__), 'data', 'screenshots')
    return send_from_directory(screenshots_dir, filename)

@app.route('/screenshots/<path:path>')
def serve_screenshot(path):
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'screenshots')
    return send_from_directory(base_dir, path)

def get_latest_screenshots():
    screenshots_dir = os.path.join(os.path.dirname(__file__), 'data', 'screenshots')
    today = datetime.now().strftime('%Y-%m-%d')
    today_dir = os.path.join(screenshots_dir, today, 'news')
    
    if not os.path.exists(today_dir):
        return []
    
    files = sorted([f for f in os.listdir(today_dir) if f.endswith('.png')], reverse=True)
    return [f'/screenshots/{today}/news/{f}' for f in files[:3]]

def run_server(port=5000):
    print(f"Dashboard API server starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_server()
