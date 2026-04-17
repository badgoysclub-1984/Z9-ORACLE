from cryptofeed import FeedHandler
import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

try:
    fh = FeedHandler()
    fh.loop = loop
    print("FeedHandler loop set manually.")
    fh.run(start_loop=False)
    print("fh.run(start_loop=False) worked.")
except Exception as e:
    print(f"Failed: {e}")
