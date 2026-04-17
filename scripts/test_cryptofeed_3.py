from cryptofeed import FeedHandler
import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def main_async():
    fh = FeedHandler()
    print("fh.run_async starting...")
    # Since we don't have feeds, it might exit immediately or wait
    # Let's see if it crashes.
    try:
        await fh.run_async()
        print("fh.run_async finished.")
    except Exception as e:
        print(f"fh.run_async failed: {e}")

asyncio.run(main_async())
