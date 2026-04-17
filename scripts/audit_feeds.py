import asyncio
import json
import numpy as np
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Kraken, Gemini

SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
METRICS_FILE = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/logs/real_time_metrics.json'

data = {s: {'KRAKEN': [], 'GEMINI': [], 'ENGINE': []} for s in SYMBOLS}

async def trade_cb(trade, receipt_timestamp):
    sym = str(trade.symbol)
    exch = str(trade.exchange)
    data[sym][exch].append(float(trade.price))

async def run_audit():
    fh = FeedHandler()
    fh.add_feed(Kraken(symbols=SYMBOLS, channels=[TRADES], callbacks={TRADES: trade_cb}))
    fh.add_feed(Gemini(symbols=SYMBOLS, channels=[TRADES], callbacks={TRADES: trade_cb}))
    
    # Run in background
    loop = asyncio.get_running_loop()
    for f in fh.feeds:
        f.start(loop)
        
    print('Auditing Kraken vs Gemini vs Engine for 60 seconds...')
    for _ in range(60):
        try:
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
                engine_prices = metrics.get('live_prices', {})
                for s in SYMBOLS:
                    if s in engine_prices:
                        data[s]['ENGINE'].append(engine_prices[s])
        except:
            pass
        await asyncio.sleep(1)
    
    analysis = {}
    for s in SYMBOLS:
        analysis[s] = {}
        for src in ['KRAKEN', 'GEMINI', 'ENGINE']:
            prices = data[s][src]
            if prices:
                analysis[s][src] = {
                    'count': len(prices),
                    'mean': np.mean(prices),
                    'std': np.std(prices),
                    'range': np.ptp(prices) if len(prices) > 0 else 0
                }
            else:
                analysis[s][src] = 'No data'
    
    print(json.dumps(analysis, indent=4))

if __name__ == '__main__':
    try:
        asyncio.run(run_audit())
    except Exception as e:
        print(f'Error: {e}')
