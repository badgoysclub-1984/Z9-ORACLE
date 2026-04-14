import asyncio
import sys
import uvloop
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, L2_BOOK
from cryptofeed.exchanges import Kraken

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def trade_callback(trade, receipt_timestamp):
    print(f'[{trade.exchange}] {trade.symbol} | Trade {trade.price}')
    sys.stdout.flush()

async def book_callback(book, receipt_timestamp):
    print(f'[{book.exchange}] {book.symbol} | Book Update')
    sys.stdout.flush()

def main():
    print('Initializing HFT Oracle Engine...')
    sys.stdout.flush()
    fh = FeedHandler()
    channels = [TRADES, L2_BOOK]
    callbacks = {TRADES: trade_callback, L2_BOOK: book_callback}
    
    fh.add_feed(Kraken(symbols=['BTC-USD'], channels=channels, callbacks=callbacks))
    
    print('Connections established. Streaming...')
    sys.stdout.flush()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    fh.run()

if __name__ == '__main__':
    main()
