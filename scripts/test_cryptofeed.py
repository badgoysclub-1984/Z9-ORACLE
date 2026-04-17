from cryptofeed import FeedHandler
import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

try:
    fh = FeedHandler()
    print("FeedHandler() works with loop set")
except Exception as e:
    print(f"FeedHandler() failed even with loop set: {e}")
