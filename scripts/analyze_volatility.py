import json
import numpy as np

LOG_FILE = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/logs/audit_log.json'

def analyze():
    with open(LOG_FILE, 'r') as f:
        data = json.load(f)
    
    btc_prices = []
    eth_prices = []
    sol_prices = []
    
    for entry in data:
        p = entry.get('prices', {})
        if 'BTC-USD' in p: btc_prices.append(p['BTC-USD'])
        if 'ETH-USD' in p: eth_prices.append(p['ETH-USD'])
        if 'SOL-USD' in p: sol_prices.append(p['SOL-USD'])
    
    # Stable reference from CoinGecko (at audit start approx)
    # BTC: 75000, ETH: 2344.55, SOL: 89.12
    
    results = {
        'BTC': {
            'mean': np.mean(btc_prices),
            'std': np.std(btc_prices),
            'range': np.ptp(btc_prices),
            'variance_pct': (np.std(btc_prices) / np.mean(btc_prices)) * 100
        },
        'ETH': {
            'mean': np.mean(eth_prices),
            'std': np.std(eth_prices),
            'range': np.ptp(eth_prices),
            'variance_pct': (np.std(eth_prices) / np.mean(eth_prices)) * 100
        },
        'SOL': {
            'mean': np.mean(sol_prices),
            'std': np.std(sol_prices),
            'range': np.ptp(sol_prices),
            'variance_pct': (np.std(sol_prices) / np.mean(sol_prices)) * 100
        }
    }
    
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    analyze()
