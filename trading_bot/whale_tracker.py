"""Whale tracker module for monitoring large Bitcoin transactions."""

import requests
import time
import os
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class WhaleTracker:
    
    EXCHANGE_ADDRESSES = {
        '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo': 'Binance',
        '3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6': 'Binance',
        'bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h': 'Binance',
        '3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS': 'Coinbase',
        '1CWaFhgGnBSKKzfmxNSFg3PVZv3yWFjGAa': 'Coinbase',
        '3AfSVihHiyBgL7cV5LmSqLZqMxLncH5u1m': 'Kraken',
        '3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r': 'Bitfinex',
        '1FWQiwK27EnGXb6BiBMRLJvunJQZZPMcGd': 'FTX',
        '1KYiKJEfdJtap9QX2v9BXJMpz2SfU4pgZw': 'Gemini',
        'bc1q2s3rjwvam9dt2ftt4sqxqjf3twav0gdnv86pmh': 'OKX',
    }
    
    def __init__(self):
        self.blockcypher_base = "https://api.blockcypher.com/v1/btc/main"
        self.blockchain_base = "https://blockchain.info"
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 300
        self.min_whale_btc = 100
        self.last_request_time = 0
        self.rate_limit_delay = 0.5
        self.sound_enabled = True
        self.last_alert_time = 0
        self.alert_cooldown = 300
    
    def _play_alert_sound(self, alert_type: str = 'whale'):
        if not self.sound_enabled:
            return
        
        if time.time() - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = time.time()
        
        def play():
            try:
                
                sound_paths = [
                    '/usr/share/sounds/freedesktop/stereo/complete.oga',
                    '/usr/share/sounds/freedesktop/stereo/bell.oga',
                    '/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga',
                    '/usr/share/sounds/gnome/default/alerts/bark.ogg',
                    '/usr/share/sounds/ubuntu/stereo/bell.ogg',
                ]
                
                for sound_path in sound_paths:
                    if os.path.exists(sound_path):
                        for _ in range(3):
                            subprocess.run(['paplay', sound_path], 
                                         capture_output=True, timeout=2)
                            time.sleep(0.3)
                        print(f"  🔊 WHALE ALERT SOUND PLAYED!")
                        return
                
                for _ in range(5):
                    print('\a', end='', flush=True)
                    time.sleep(0.2)
                print(f"  🔊 WHALE ALERT (beep)!")
                
            except Exception as e:
                print(f"  [Sound] Could not play alert: {e}")
        
        thread = threading.Thread(target=play, daemon=True)
        thread.start()
        
    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_recent_large_transactions(self, minutes: int = 15) -> List[Dict]:
        """Get recent large Bitcoin transactions from mempool and blocks."""
        cache_key = f"large_txs_{minutes}m"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        all_txs = []
        cutoff_time = time.time() - (minutes * 60)
        
        try:
            self._rate_limit()
            url = f"{self.blockchain_base}/unconfirmed-transactions?format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                mempool_txs = data.get('txs', [])
                all_txs.extend(mempool_txs)
                print(f"  [Whale] Mempool: {len(mempool_txs)} transactions")
            
            self._rate_limit()
            blocks_url = f"{self.blockchain_base}/blocks?format=json"
            blocks_resp = requests.get(blocks_url, timeout=10)
            
            if blocks_resp.status_code == 200:
                blocks_data = blocks_resp.json()
                recent_blocks = blocks_data.get('blocks', [])[:3]
                
                for block in recent_blocks:
                    block_time = block.get('time', 0)
                    if block_time >= cutoff_time:
                        self._rate_limit()
                        block_hash = block.get('hash', '')
                        block_url = f"{self.blockchain_base}/rawblock/{block_hash}"
                        block_resp = requests.get(block_url, timeout=15)
                        
                        if block_resp.status_code == 200:
                            block_data = block_resp.json()
                            block_txs = block_data.get('tx', [])
                            all_txs.extend(block_txs)
                            print(f"  [Whale] Block {block.get('height', '?')}: {len(block_txs)} transactions")
            
            txs = all_txs
            print(f"  [Whale] Total scanning: {len(txs)} transactions from last {minutes} min")
            
            large_txs = []
            for tx in txs:
                total_btc = sum(out.get('value', 0) for out in tx.get('out', [])) / 1e8
                
                if total_btc >= self.min_whale_btc:
                    from_exchange = None
                    to_exchange = None
                    
                    for inp in tx.get('inputs', []):
                        addr = inp.get('prev_out', {}).get('addr', '')
                        if addr in self.EXCHANGE_ADDRESSES:
                            from_exchange = self.EXCHANGE_ADDRESSES[addr]
                            break
                    
                    for out in tx.get('out', []):
                        addr = out.get('addr', '')
                        if addr in self.EXCHANGE_ADDRESSES:
                            to_exchange = self.EXCHANGE_ADDRESSES[addr]
                            break
                    
                    large_txs.append({
                        'hash': tx.get('hash', '')[:16] + '...',
                        'btc': round(total_btc, 2),
                        'usd_approx': 0,
                        'from_exchange': from_exchange,
                        'to_exchange': to_exchange,
                        'time': datetime.fromtimestamp(tx.get('time', 0)).strftime('%H:%M:%S'),
                        'type': self._classify_tx(from_exchange, to_exchange)
                    })
            
            large_txs.sort(key=lambda x: x['btc'], reverse=True)
            
            self.cache[cache_key] = large_txs[:10]
            self.cache_time = datetime.now()
            
            return large_txs[:10]
            
        except Exception as e:
            print(f"  [Whale] Tracker error: {e}")
            return []
    
    def _classify_tx(self, from_exchange: Optional[str], to_exchange: Optional[str]) -> str:
        if from_exchange and not to_exchange:
            return 'WITHDRAWAL'
        elif to_exchange and not from_exchange:
            return 'DEPOSIT'
        elif from_exchange and to_exchange:
            return 'EXCHANGE_TRANSFER'
        else:
            return 'UNKNOWN'
    
    def get_exchange_flow_sentiment(self) -> Dict:
        """Calculate exchange flow sentiment from whale transactions."""
        txs = self.get_recent_large_transactions(minutes=15)
        
        if not txs:
            return {
                'sentiment': 'NEUTRAL',
                'score': 0.0,
                'deposits': 0,
                'withdrawals': 0,
                'deposit_btc': 0,
                'withdrawal_btc': 0,
                'net_flow': 0,
                'large_txs': []
            }
        
        deposits = [tx for tx in txs if tx['type'] == 'DEPOSIT']
        withdrawals = [tx for tx in txs if tx['type'] == 'WITHDRAWAL']
        
        deposit_btc = sum(tx['btc'] for tx in deposits)
        withdrawal_btc = sum(tx['btc'] for tx in withdrawals)
        net_flow = withdrawal_btc - deposit_btc
        
        total = deposit_btc + withdrawal_btc
        if total > 0:
            score = net_flow / total
        else:
            score = 0.0
        
        if score > 0.3:
            sentiment = 'BULLISH'
        elif score < -0.3:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'score': round(score, 2),
            'deposits': len(deposits),
            'withdrawals': len(withdrawals),
            'deposit_btc': round(deposit_btc, 2),
            'withdrawal_btc': round(withdrawal_btc, 2),
            'net_flow': round(net_flow, 2),
            'large_txs': txs[:5]
        }
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self.cache or self.cache_time is None:
            return False
        elapsed = (datetime.now() - self.cache_time).seconds
        return elapsed < self.cache_duration
    
    def analyze(self, current_price: float = 100000) -> Dict:
        """Analyze whale activity and return sentiment data."""
        flow = self.get_exchange_flow_sentiment()
        
        for tx in flow.get('large_txs', []):
            tx['usd_approx'] = round(tx['btc'] * current_price, 0)
        
        flow['deposit_usd'] = round(flow['deposit_btc'] * current_price, 0)
        flow['withdrawal_usd'] = round(flow['withdrawal_btc'] * current_price, 0)
        flow['net_flow_usd'] = round(flow['net_flow'] * current_price, 0)
        
        reasons = []
        if flow['withdrawals'] > flow['deposits']:
            reasons.append(f"{flow['withdrawals']} withdrawals vs {flow['deposits']} deposits")
        elif flow['deposits'] > flow['withdrawals']:
            reasons.append(f"{flow['deposits']} deposits vs {flow['withdrawals']} withdrawals")
        
        if abs(flow['net_flow']) > 100:
            direction = "OUT of" if flow['net_flow'] > 0 else "INTO"
            reasons.append(f"Net {abs(flow['net_flow']):.0f} BTC {direction} exchanges")
            self._play_alert_sound('whale')
        
        flow['reasoning'] = " | ".join(reasons) if reasons else "No significant whale activity"
        
        return flow

def test_whale_tracker():
    print("=" * 60)
    print("WHALE TRACKER TEST")
    print("=" * 60)
    
    tracker = WhaleTracker()
    result = tracker.analyze(current_price=100000)
    
    print(f"\nExchange Flow Sentiment: {result['sentiment']}")
    print(f"Score: {result['score']:+.2f}")
    print(f"Deposits: {result['deposits']} txs ({result['deposit_btc']:.2f} BTC)")
    print(f"Withdrawals: {result['withdrawals']} txs ({result['withdrawal_btc']:.2f} BTC)")
    print(f"Net Flow: {result['net_flow']:+.2f} BTC (${result['net_flow_usd']:,.0f})")
    print(f"Reasoning: {result['reasoning']}")
    
    if result['large_txs']:
        print(f"\nTop Whale Transactions:")
        for tx in result['large_txs'][:5]:
            print(f"  • {tx['btc']:.2f} BTC (${tx['usd_approx']:,.0f}) - {tx['type']} @ {tx['time']}")

if __name__ == '__main__':
    test_whale_tracker()
