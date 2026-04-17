import asyncio
import time
import json
import numpy as np
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Kraken, Gemini

SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
results = {s: {'KRAKEN': [], 'GEMINI': []} for s in SYMBOLS}

async def trade_callback(trade, receipt_timestamp):
    sym = str(trade.symbol)
    exch = str(trade.exchange)
    price = float(trade.price)
    if sym in results:
        results[sym][exch].append(price)

async def run_audit():
    print('Auditing Kraken vs Gemini for 60 seconds...')
    fh = FeedHandler()
    fh.add_feed(Kraken(symbols=SYMBOLS, channels=[TRADES], callbacks={TRADES: trade_callback}))
    fh.add_feed(Gemini(symbols=SYMBOLS, channels=[TRADES], callbacks={TRADES: trade_callback}))
    
    # Run in background
    stop_event = asyncio.Event()
    
    async def stop_after_delay():
        await asyncio.sleep(60)
        fh.stop()
        print('Audit finished.')

    await asyncio.gather(fh.run(), stop_after_delay())

    summary = {}
    for sym in SYMBOLS:
        kr = results[sym]['KRAKEN']
        ge = results[sym]['GEMINI']
        
        if kr and ge:
            mid_diff = np.abs(np.mean(kr) - np.mean(ge))
            kr_std = np.std(kr)
            ge_std = np.std(ge)
            summary[sym] = {
                'kraken_avg': np.mean(kr),
                'gemini_avg': np.mean(ge),
                'abs_diff': mid_diff,
                'kraken_vol': kr_std,
                'gemini_vol': ge_std,
                'natural_variance_pct': (mid_diff / np.mean(kr)) * 100
            }
        else:
            summary[sym] = 'Insufficient data'

    print(json.dumps(summary, indent=4))

if __name__ == '__main__':
    asyncio.run(run_audit())
