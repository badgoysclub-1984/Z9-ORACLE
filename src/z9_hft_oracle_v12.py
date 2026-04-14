import asyncio
import numpy as np
import onnxruntime as ort
import time
import logging
from collections import deque
import os
import csv
import json
from datetime import datetime
import sys
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, L2_BOOK
from cryptofeed.exchanges import Kraken, Gemini

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('Z9Oracle')

PROJECT_DIR = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE'
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
TRADE_LOG_PATH = os.path.join(LOG_DIR, 'trade_history.csv')
METRICS_JSON_PATH = os.path.join(LOG_DIR, 'real_time_metrics.json')
DEFAULT_ONNX_PATH = os.path.join(PROJECT_DIR, 'models/skyrmatron_v91_w4.onnx')

# Simulation Constants
LEVERAGE_BASE = 15.0
MAX_POSITION_USD = 500
STOP_LOSS_PCT = 0.003
TAKE_PROFIT_PCT = 0.012
Z9_CONFIDENCE_THRESHOLD = 0.85

class MetricsManager:
    def __init__(self):
        self.trades = []
        self.balance = 1000.0
        self.peak_balance = 1000.0
        self.metrics = self._get_default_metrics()
        self._ensure_logs()

    def _get_default_metrics(self):
        return {
            'accuracy': 0.0,
            'sharpe': 0.0,
            'drawdown': 0.0,
            'win_loss_ratio': 0.0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'current_balance': 1000.0,
            'win_pct': 0.0,
            'last_update': datetime.now().isoformat(),
            'live_prices': {'BTC-USD': 0, 'ETH-USD': 0, 'SOL-USD': 0}
        }

    def _ensure_logs(self):
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        with open(TRADE_LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'coin', 'type', 'leverage', 'entry_price', 'exit_price', 'pnl_abs', 'pnl_pct', 'status', 'z9_confidence'])
        self.save_metrics()

    def record_trade(self, coin, trade_type, leverage, entry_price, exit_price, pnl_abs, pnl_pct, confidence):
        timestamp = datetime.now().isoformat()
        self.balance += float(pnl_abs)
        if self.balance > self.peak_balance: self.peak_balance = self.balance
        trade_data = [timestamp, coin, trade_type, leverage, entry_price, exit_price, pnl_abs, pnl_pct, 'Closed', confidence]
        self.trades.append(trade_data)
        with open(TRADE_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(trade_data)
        self.update_metrics()

    def update_metrics(self):
        pnls = [float(t[6]) for t in self.trades] if self.trades else []
        win_trades = [p for p in pnls if p > 0]
        loss_trades = [p for p in pnls if p <= 0]
        
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['win_pct'] = (len(win_trades) / len(self.trades)) * 100 if self.trades else 0
        self.metrics['accuracy'] = len(win_trades) / len(self.trades) if self.trades else 0
        self.metrics['total_pnl'] = sum(pnls)
        self.metrics['current_balance'] = self.balance
        self.metrics['last_update'] = datetime.now().isoformat()
        
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = abs(np.mean(loss_trades)) if loss_trades else 1e-8
        self.metrics['win_loss_ratio'] = avg_win / avg_loss
        
        if len(pnls) > 1:
            std = np.std(pnls)
            self.metrics['sharpe'] = (np.mean(pnls) / (std + 1e-8)) * np.sqrt(365 * 24 * 60)
        else:
            self.metrics['sharpe'] = 0.0
            
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.metrics['drawdown'] = drawdown
        self.save_metrics()

    def save_metrics(self):
        try:
            with open(METRICS_JSON_PATH, 'w') as f: json.dump(self.metrics, f, indent=4)
        except Exception as e: logger.error(f'Error saving metrics: {e}')

class Z9OracleHFT:
    def __init__(self, onnx_path=DEFAULT_ONNX_PATH):
        self.has_model = False
        if os.path.exists(onnx_path):
            try:
                self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.has_model = True
                print("[Z9] ONNX Model loaded.", flush=True)
            except Exception as e: print(f"[Z9] ONNX Error: {e}", flush=True)
        
        self.metrics_mgr = MetricsManager()
        self.price_buffer = deque(maxlen=512)
        self.latest_prices = {'BTC-USD': 0, 'ETH-USD': 0, 'SOL-USD': 0}
        self.open_trades = []
        self.last_signal_ts = 0

    def z9_predict(self, buffer):
        if not self.has_model:
            direction = 1 if buffer[-1] > buffer[0] else -1
            return direction, np.random.uniform(0.7, 0.95)
        try:
            input_data = np.array(list(buffer), dtype=np.float32).reshape(1, 512, 1)
            input_data_full = np.tile(input_data, (1, 1, 36))
            output = self.session.run(['output'], {'input': input_data_full})[0][0]
            direction = int(np.sign(output[0]))
            coherence = 1.0 - abs(output.mean() % 9) / 9.0
            return direction, max(0.1, min(0.99, coherence))
        except: return 1, 0.5

    async def trade_callback(self, trade, receipt_timestamp):
        print(f"[{trade.exchange}] Trade {trade.symbol} | {trade.side} {trade.amount} @ {trade.price} | Time: {trade.timestamp}", flush=True)
        
        coin = str(trade.symbol)
        price = float(trade.price)
        self.latest_prices[coin] = price
        self.metrics_mgr.metrics['live_prices'] = self.latest_prices
        
        if coin == 'BTC-USD': self.price_buffer.append(price)
        
        for op in self.open_trades[:]:
            if op['coin'] == coin:
                pnl_pct = (price - op['entry_price']) / op['entry_price'] if op['side'] == 'Long' else (op['entry_price'] - price) / op['entry_price']
                if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= -STOP_LOSS_PCT or (time.time() - op['entry_ts']) >= 60:
                    pnl_abs = MAX_POSITION_USD * (pnl_pct * op['leverage'])
                    self.metrics_mgr.record_trade(op['coin'], op['side'], op['leverage'], op['entry_price'], price, pnl_abs, pnl_pct * op['leverage'], op['confidence'])
                    self.open_trades.remove(op)
                    print(f"[SIM] Position Closed: {op['side']} {coin} | PnL: ${pnl_abs:.2f}", flush=True)

        if len(self.price_buffer) == 512 and time.time() - self.last_signal_ts > 10:
            direction, conf = self.z9_predict(self.price_buffer)
            if conf > Z9_CONFIDENCE_THRESHOLD:
                target_coin = 'ETH-USD' if direction > 0 else 'SOL-USD'
                target_price = self.latest_prices.get(target_coin, 0)
                if target_price > 0:
                    side = 'Long' if direction > 0 else 'Short'
                    leverage = int(LEVERAGE_BASE * conf)
                    self.open_trades.append({'coin': target_coin, 'side': side, 'entry_price': target_price, 'leverage': leverage, 'confidence': conf, 'entry_ts': time.time()})
                    print(f"[Z9-SIGNAL] {side} {target_coin} @ {target_price:.2f} (Conf: {conf:.3f})", flush=True)
                    self.last_signal_ts = time.time()
                    
        self.metrics_mgr.save_metrics()

    async def book_callback(self, book, receipt_timestamp):
        top_bid = next(iter(book.book.bids)) if book.book.bids else None
        top_ask = next(iter(book.book.asks)) if book.book.asks else None
        
        if top_bid and top_ask:
            self.latest_prices[book.symbol] = (float(top_bid) + float(top_ask)) / 2
            self.metrics_mgr.metrics['live_prices'] = self.latest_prices
            if not hasattr(self, '_book_count'): self._book_count = 0
            self._book_count += 1
            if self._book_count % 10 == 0:
                self.metrics_mgr.save_metrics()

def main():
    print("Initializing Z9 HFT Oracle Engine...", flush=True)
    oracle = Z9OracleHFT()
    print("Oracle initialized.", flush=True)
    fh = FeedHandler()
    print("FeedHandler initialized.", flush=True)
    channels = [TRADES, L2_BOOK]
    callbacks = {TRADES: oracle.trade_callback, L2_BOOK: oracle.book_callback}
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    print("Adding Kraken...", flush=True)
    fh.add_feed(Kraken(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], channels=channels, callbacks=callbacks))
    
    print("Connections established. Streaming Kraken L2/Trade data...", flush=True)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fh.run()
    except KeyboardInterrupt:
        print("\nStopping oracle process...", flush=True)

if __name__ == '__main__':
    main()
