from cryptofeed.exchanges import Kraken, Gemini
from cryptofeed import FeedHandler
import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def test():
    try:
        print("Initializing FeedHandler...")
        fh = FeedHandler()
        SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        
        print("Adding Kraken...")
        fh.add_feed(Kraken(symbols=SYMBOLS))
        print("Kraken added.")
        
        print("Adding Gemini...")
        fh.add_feed(Gemini(symbols=SYMBOLS))
        print("Gemini added.")
        
    except Exception as e:
        print(f"Failed: {e}")

asyncio.run(test())
