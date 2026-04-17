from cryptofeed.exchanges import Gemini
from cryptofeed.defines import TRADES, L2_BOOK
import asyncio

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def callback(data, ts):

    print(f"Callback data: {data}")

async def test():
    try:
        print("Initializing Gemini with symbols and callbacks...")
        SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        channels = [TRADES, L2_BOOK]
        callbacks = {TRADES: callback, L2_BOOK: callback}
        g = Gemini(symbols=SYMBOLS, channels=channels, callbacks=callbacks)
        print("Gemini initialized.")
    except Exception as e:
        print(f"Gemini failed: {e}")

asyncio.run(test())
