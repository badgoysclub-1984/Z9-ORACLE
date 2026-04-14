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
import uvloop
import sys
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, L2_BOOK
from cryptofeed.exchanges import Kraken, Gemini

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
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
Z9_CONFIDENCE_THRESHOLD = 0.89
HINDSIGHT_BUFFER_SIZE = 32
OPTIMIZE_EVERY = 100

class MetricsManager:
    def __init__(self):
        self.trades = []
        self.balance = 1000.0
        self.peak_balance = 1000.0
        self.metrics = {
            'accuracy': 0.0,
            'sharpe': 0.0,
            'drawdown': 0.0,
            'win_loss_ratio': 0.0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'current_balance': 1000.0,
            'win_pct': 0.0,
            'last_update': datetime.now().isoformat()
        }
        self._ensure_logs()

    def _ensure_logs(self):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        if not os.path.exists(TRADE_LOG_PATH):
            with open(TRADE_LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'coin', 'type', 'leverage', 'entry_price', 'exit_price', 'pnl_abs', 'pnl_pct', 'status', 'z9_confidence'])
        self.save_metrics()

    def record_trade(self, coin, trade_type, leverage, entry_price, exit_price, pnl_abs, pnl_pct, confidence):
        timestamp = datetime.now().isoformat()
        self.balance += float(pnl_abs)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        trade_data = [timestamp, coin, trade_type, leverage, entry_price, exit_price, pnl_abs, pnl_pct, 'Closed', confidence]
        self.trades.append(trade_data)
        
        with open(TRADE_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(trade_data)
        
        self.update_metrics()
        logger.info(f'TRADE RECORDED: {trade_type} {coin} | PnL: ${pnl_abs:.2f} ({pnl_pct*100:.2f}%) | Bal: ${self.balance:.2f}')

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
            self.metrics['sharpe'] = 0
            
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.metrics['drawdown'] = drawdown
        self.save_metrics()

    def save_metrics(self):
        try:
            with open(METRICS_JSON_PATH, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            logger.error(f'Error saving metrics: {e}')

class Z9OracleHFT:
    def __init__(self, onnx_path=DEFAULT_ONNX_PATH):
        self.has_model = False
        if os.path.exists(onnx_path):
            try:
                self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.has_model = True
                logger.info('ONNX Model loaded successfully.')
            except Exception as e:
                logger.error(f'ONNX Load Error: {e}')
        else:
            logger.warning('ONNX Model missing. Running in Simulated Signal mode.')

        self.metrics_mgr = MetricsManager()
        self.price_buffer = deque(maxlen=512)
        self.latest_prices = {'BTC-USD': 0, 'ETH-USD': 0, 'SOL-USD': 0}
        self.open_trades = []
        self.last_signal_ts = 0
        self.trade_count = 0
        self.hindsight = deque(maxlen=HINDSIGHT_BUFFER_SIZE)
        self.meta_params = {'threshold': Z9_CONFIDENCE_THRESHOLD, 'leverage_base': LEVERAGE_BASE}

    def z9_predict(self, buffer):
        if not self.has_model:
            direction = 1 if buffer[-1] > buffer[0] else -1
            confidence = np.random.uniform(0.7, 0.95)
            return direction, confidence
        try:
            input_data = np.array(list(buffer), dtype=np.float32).reshape(1, 512, 1)
            input_data_full = np.tile(input_data, (1, 1, 36))
            output = self.session.run(['output'], {'input': input_data_full})[0][0]
            direction = int(np.sign(output[0]))
            coherence = 1.0 - abs(output.mean() % 9) / 9.0
            return direction, max(0.1, min(0.99, coherence))
        except:
            return 1, 0.5

    def retrocausal_correction(self):
        if len(self.hindsight) < 8: return
        total_error = sum(1 if outcome < 0 else -0.5 for _, outcome in self.hindsight)
        correction_factor = 1.0 - (total_error % 9) * 0.02
        self.meta_params['threshold'] *= correction_factor
        self.meta_params['threshold'] = max(0.05, min(0.95, self.meta_params['threshold']))
        self.meta_params['leverage_base'] = max(8, min(25, self.meta_params['leverage_base'] * correction_factor))

    def z9_quantum_swarm(self, features):
        try:
            feats = np.array(features)
            best_energy = np.inf
            best_dir = 1
            for d in [1, -1]:
                energy = np.sum(feats * d) 
                if energy < best_energy:
                    best_energy = energy
                    best_dir = d
            return best_dir
        except:
            return 1

    async def handle_trade_event(self, trade, receipt_timestamp):
        coin = str(trade.symbol)
        price = float(trade.price)
        self.latest_prices[coin] = price
        if coin == 'BTC-USD':
            self.price_buffer.append(price)
            if len(self.price_buffer) % 500 == 0:
                logger.info(f'BTC Price Update: {price} | Buffer: {len(self.price_buffer)}')
        
        for op in self.open_trades[:]:
            if op['coin'] == coin:
                pnl_pct = (price - op['entry_price']) / op['entry_price'] if op['side'] == 'Long' else (op['entry_price'] - price) / op['entry_price']
                elapsed = time.time() - op['entry_ts']
                
                if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= -STOP_LOSS_PCT or elapsed >= 60:
                    pnl_abs = MAX_POSITION_USD * (pnl_pct * op['leverage'])
                    self.metrics_mgr.record_trade(op['coin'], op['side'], op['leverage'], op['entry_price'], price, pnl_abs, pnl_pct * op['leverage'], op['confidence'])
                    self.open_trades.remove(op)
                    self.hindsight.append((1 if pnl_pct > 0 else -1, pnl_pct))
                    
                    if len(self.hindsight) % OPTIMIZE_EVERY == 0:
                        self.retrocausal_correction()

        if len(self.price_buffer) == 512 and time.time() - self.last_signal_ts > 10:
            direction, z9_conf = self.z9_predict(self.price_buffer)
            swarm_boost = self.z9_quantum_swarm(list(self.price_buffer)[-9:])
            combined_conf = z9_conf * (1 + 0.3 * (1 if swarm_boost == direction else -0.5))
            
            if combined_conf > self.meta_params['threshold']:
                target_coin = 'ETH-USD' if direction > 0 else 'SOL-USD'
                target_price = self.latest_prices.get(target_coin, 0)
                if target_price > 0:
                    side = 'Long' if direction > 0 else 'Short'
                    leverage = int(self.meta_params['leverage_base'] * combined_conf)
                    leverage = max(1, min(50, leverage))
                    
                    self.open_trades.append({
                        'coin': target_coin,
                        'side': side,
                        'entry_price': target_price,
                        'leverage': leverage,
                        'confidence': combined_conf,
                        'entry_ts': time.time()
                    })
                    logger.info(f'SIGNAL: {side} {target_coin} @ {target_price:.2f} (Conf: {combined_conf:.3f} | Lev: {leverage}x)')
                    self.last_signal_ts = time.time()
                    self.trade_count += 1

    async def handle_book_event(self, book, receipt_timestamp):
        if book.book.bids and book.book.asks:
            bid = float(list(book.book.bids.keys())[0])
            ask = float(list(book.book.asks.keys())[0])
            self.latest_prices[book.symbol] = (bid + ask) / 2

async def main_async():
    logger.info('--- Z9 ORACLE V12.3 BOOTSTRAP ---')
    oracle = Z9OracleHFT()
    fh = FeedHandler()
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    callbacks = {TRADES: oracle.handle_trade_event, L2_BOOK: oracle.handle_book_event}
    
    fh.add_feed(Kraken(symbols=symbols, channels=[TRADES, L2_BOOK], callbacks=callbacks))
    fh.add_feed(Gemini(symbols=symbols, channels=[TRADES, L2_BOOK], callbacks=callbacks))
    
    logger.info('Streaming feeds initialized. Starting loop...')
    loop = asyncio.get_running_loop()
    for feed in fh.feeds:
        feed.start(loop)
    
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info('Shutting down oracle...')
