from cryptofeed.exchanges import Gemini
import asyncio

async def test():
    try:
        print("Initializing Gemini...")
        g = Gemini(symbols=['BTC-USD'])
        print("Gemini initialized.")
    except Exception as e:
        print(f"Gemini failed: {e}")

asyncio.run(test())
