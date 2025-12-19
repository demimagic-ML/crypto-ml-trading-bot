import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from binance.client import Client

api_key = os.getenv('BINANCE_API_KEY', '')
api_secret = os.getenv('BINANCE_API_SECRET', '')

client = Client(api_key, api_secret)

print("Fetching cross margin pairs...")
try:
    margin_info = client.get_margin_all_pairs()
    
    btc_pairs = [p for p in margin_info if 'BTC' in p['base'] or 'BTC' in p['symbol']]
    
    print(f"\nFound {len(margin_info)} total margin pairs")
    print(f"\nBTC-related margin pairs:")
    print("-" * 40)
    
    for pair in btc_pairs[:20]:
        print(f"  {pair['symbol']} - Base: {pair['base']}, Quote: {pair['quote']}")
    
    print("\n" + "="*40)
    print("Checking specific pairs:")
    print("="*40)
    
    for symbol in ['BTCUSDT', 'BTCUSDC', 'BTCBUSD']:
        found = any(p['symbol'] == symbol for p in margin_info)
        status = "✅ Available" if found else "❌ Not available"
        print(f"  {symbol}: {status}")
    
    print("\n" + "="*40)
    print("Your Margin Account Balances:")
    print("="*40)
    
    account = client.get_margin_account()
    for asset in account['userAssets']:
        free = float(asset['free'])
        borrowed = float(asset['borrowed'])
        if free > 0 or borrowed > 0:
            print(f"  {asset['asset']}: Free={free:.6f}, Borrowed={borrowed:.6f}")

except Exception as e:
    print(f"Error: {e}")
