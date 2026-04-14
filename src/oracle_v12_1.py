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
import ccxt.async_support as ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('Z9Oracle')

def direct_log(msg):
    try:
        with open("/tmp/z9_direct.log", "a") as f:
            f.write(f"{datetime.now()} | {msg}\n")
    except:
        pass

direct_log("--- Z9 ORACLE V12.2 (ETH/SOL PAPER SIM) STARTING ---")

# Sampler Logic
HAS_DWAVE = False
try:
    from dwave.system import LeapHybridSampler
    HAS_DWAVE = True
except ImportError:
    pass

HAS_NEAL = False
try:
    import neal
    HAS_NEAL = True
except ImportError:
    pass

SIMULATION_MODE = True
LEVERAGE_BASE = 15.0
MAX_POSITION_USD = 500
STOP_LOSS_PCT = 0.003
TAKE_PROFIT_PCT = 0.012
Z9_CONFIDENCE_THRESHOLD = 0.89  # Increased for precision
HINDSIGHT_BUFFER_SIZE = 32
OPTIMIZE_EVERY = 100

PROJECT_DIR = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE'
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
TRADE_LOG_PATH = os.path.join(LOG_DIR, 'trade_history.csv')
METRICS_JSON_PATH = os.path.join(LOG_DIR, 'real_time_metrics.json')
DEFAULT_ONNX_PATH = os.path.join(PROJECT_DIR, 'models/skyrmatron_v91_w4.onnx')

class MetricsManager:
    def __init__(self):
        direct_log("Metrics: init")
        self.trades = []
        self.balance = 1000.0
        self.peak_balance = 1000.0
        self.metrics = {
            "accuracy": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "win_loss_ratio": 0.0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "current_balance": 1000.0,
            "win_pct": 0.0
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
        self.balance += pnl_abs
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        trade_data = [timestamp, coin, trade_type, leverage, entry_price, exit_price, pnl_abs, pnl_pct, 'Closed', confidence]
        self.trades.append(trade_data)
        
        with open(TRADE_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(trade_data)
        
        self.update_metrics()
        msg = f"TRADE RECORDED: {trade_type} {coin} | PnL: ${pnl_abs:.2f} ({pnl_pct*100:.2f}%) | Bal: ${self.balance:.2f}"
        logger.info(msg)
        direct_log(msg)

    def update_metrics(self):
        if not self.trades: return
        pnls = [t[6] for t in self.trades]
        win_trades = [p for p in pnls if p > 0]
        loss_trades = [p for p in pnls if p <= 0]
        
        self.metrics["total_trades"] = len(self.trades)
        self.metrics["win_pct"] = (len(win_trades) / len(self.trades)) * 100 if self.trades else 0
        self.metrics["accuracy"] = len(win_trades) / len(self.trades) if self.trades else 0
        self.metrics["total_pnl"] = sum(pnls)
        self.metrics["current_balance"] = self.balance
        
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = abs(np.mean(loss_trades)) if loss_trades else 1e-8
        self.metrics["win_loss_ratio"] = avg_win / avg_loss
        
        if len(pnls) > 1:
            std = np.std(pnls)
            self.metrics["sharpe"] = (np.mean(pnls) / (std + 1e-8)) * np.sqrt(365 * 24 * 60)
        else:
            self.metrics["sharpe"] = 0
            
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.metrics["drawdown"] = drawdown
        self.save_metrics()

    def save_metrics(self):
        with open(METRICS_JSON_PATH, 'w') as f:
            json.dump(self.metrics, f, indent=4)

class Z9HFTOracleV12:
    def __init__(self, onnx_path=DEFAULT_ONNX_PATH):
        direct_log("Oracle: init start")
        if os.path.exists(onnx_path):
            try:
                self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.has_model = True
                direct_log("Oracle: ONNX OK")
            except Exception as e:
                self.has_model = False
                direct_log(f"Oracle: ONNX Load Error: {e}")
        else:
            self.has_model = False
            direct_log("Oracle: ONNX Missing")
            
        self.price_buffer = deque(maxlen=512)
        self.metrics_mgr = MetricsManager()
        
        self.sampler = None
        if HAS_DWAVE:
            try:
                self.sampler = LeapHybridSampler()
                direct_log("Oracle: D-Wave Leap OK")
            except Exception:
                direct_log("Oracle: D-Wave Auth Fail, Fallback to Local")
                
        if self.sampler is None and HAS_NEAL:
            try:
                self.sampler = neal.SimulatedAnnealingSampler()
                direct_log("Oracle: Neal SA Sampler OK")
            except Exception as e:
                direct_log(f"Oracle: Neal Fail: {e}")
            
        self.trade_count = 0
        self.last_trade_ts = 0
        self.hindsight = deque(maxlen=HINDSIGHT_BUFFER_SIZE)
        self.meta_params = {'threshold': Z9_CONFIDENCE_THRESHOLD, 'leverage_base': LEVERAGE_BASE}
        
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'kraken': ccxt.kraken({'enableRateLimit': True}),
            'bitstamp': ccxt.bitstamp({'enableRateLimit': True})
        }
        
        self.latest_prices = {'BTC': 0, 'ETH': 0, 'SOL': 0}
        self.open_trades = []

    async def fetch_prices(self):
        tasks = []
        for name, exchange in self.exchanges.items():
            tasks.append(exchange.fetch_ticker('BTC/USDT' if name == 'binance' else 'BTC/USD'))
            tasks.append(exchange.fetch_ticker('ETH/USDT' if name == 'binance' else 'ETH/USD'))
            tasks.append(exchange.fetch_ticker('SOL/USDT' if name == 'binance' else 'SOL/USD'))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        btc_p, eth_p, sol_p = [], [], []
        for res in results:
            if isinstance(res, dict) and 'symbol' in res:
                if 'BTC' in res['symbol']: btc_p.append(res['last'])
                elif 'ETH' in res['symbol']: eth_p.append(res['last'])
                elif 'SOL' in res['symbol']: sol_p.append(res['last'])
        
        if btc_p: self.latest_prices['BTC'] = np.mean(btc_p)
        if eth_p: self.latest_prices['ETH'] = np.mean(eth_p)
        if sol_p: self.latest_prices['SOL'] = np.mean(sol_p)
        
        return self.latest_prices['BTC']

    def z9_predict(self, btc_window):
        if not self.has_model:
            direction = 1 if np.random.random() > 0.5 else -1
            coherence = np.random.uniform(0.7, 0.99)
            return direction, coherence
        try:
            # Reshape to (1, 512, 1) then tile to (1, 512, 36)
            input_data = np.array(btc_window, dtype=np.float32).reshape(1, 512, 1)
            input_data_full = np.tile(input_data, (1, 1, 36))
            output = self.session.run(['output'], {'input': input_data_full})[0][0]
            direction = int(np.sign(output[0]))
            # coherence calculation from v12 codebase
            coherence = 1.0 - abs(output.mean() % 9) / 9.0
            # Ensure it stays in a reasonable range
            coherence = max(0.1, min(0.99, coherence))
            return direction, coherence
        except Exception as e:
            direct_log(f"Prediction Error: {e}")
            return (1, 0.5)

    def temporal_sense(self, recent_prices):
        prices = np.array(recent_prices)
        vol = np.std(prices[-64:]) / (np.mean(prices[-64:]) + 1e-8) if len(prices) > 64 else 0.01
        
        # convert to numpy array before addition
        diffs = np.diff(prices[-32:])
        divs = np.array(prices[-33:-1]) + 1e-8
        returns = diffs / divs if len(prices) > 33 else np.array([0])
        
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0
        regime = 1.0 if np.std(prices[-9:]) < 0.002 else 0.6
        return {'vol': vol, 'sharpe': sharpe, 'regime': regime, 'sentiment': 0.8}

    def retrocausal_correction(self):
        if len(self.hindsight) < 8: return
        total_error = sum(1 if outcome < 0 else -0.5 for _, outcome in self.hindsight)
        correction_factor = 1.0 - (total_error % 9) * 0.02
        self.meta_params['threshold'] *= correction_factor
        # Bounds for safety
        self.meta_params['threshold'] = max(0.05, min(0.95, self.meta_params['threshold']))
        self.meta_params['leverage_base'] = max(8, min(25, self.meta_params['leverage_base'] * correction_factor))

    def z9_quantum_swarm(self, features):
        if not self.sampler:
            return 1 if np.random.random() > 0.5 else -1
        try:
            Q = {}
            for i in range(9):
                for j in range(i, 9):
                    Q[(i, j)] = float(features[i % len(features)] * features[j % len(features)] * (1.0 if i == j else 0.5))
            
            # Using specific logic for QUBO sampling
            if HAS_DWAVE and isinstance(self.sampler, LeapHybridSampler):
                samples = self.sampler.sample_qubo(Q)
            else:
                samples = self.sampler.sample_qubo(Q, num_reads=9)
                
            best = min(samples.record.energy)
            return 1 if best < 0 else -1
        except Exception:
            return 1

    async def run(self):
        direct_log("Z9-HFT Oracle v12.2 STARTED — Multi-Exchange Paper Trading Mode")
        
        while True:
            try:
                btc_price = await self.fetch_prices()
                if btc_price == 0:
                    await asyncio.sleep(1)
                    continue
                
                self.price_buffer.append(btc_price)
                
                current_time = time.time()
                # Check open positions
                for trade in self.open_trades[:]:
                    coin_price = self.latest_prices[trade['coin']]
                    elapsed = current_time - trade['entry_ts']
                    pnl_pct = (coin_price - trade['entry_price']) / trade['entry_price'] if trade['side'] == 'Long' else (trade['entry_price'] - coin_price) / trade['entry_price']
                    
                    if abs(pnl_pct) >= TAKE_PROFIT_PCT or pnl_pct <= -STOP_LOSS_PCT or elapsed >= 60:
                        pnl_abs = MAX_POSITION_USD * (pnl_pct * trade['leverage'])
                        self.metrics_mgr.record_trade(trade['coin'], trade['side'], trade['leverage'], trade['entry_price'], coin_price, pnl_abs, pnl_pct * trade['leverage'], trade['confidence'])
                        self.open_trades.remove(trade)
                        self.hindsight.append((1 if pnl_pct > 0 else -1, pnl_pct))

                if len(self.price_buffer) < 512:
                    if len(self.price_buffer) % 50 == 0:
                        direct_log(f"Warming up buffer: {len(self.price_buffer)}/512")
                    await asyncio.sleep(0.1)
                    continue
                
                prices_list = list(self.price_buffer)
                direction, z9_conf = self.z9_predict(prices_list)
                sense = self.temporal_sense(prices_list)
                
                # Dynamic leverage base from meta_params
                leverage = int(self.meta_params['leverage_base'] * z9_conf * (1 - sense['vol']) * max(0.1, sense['sharpe']) * 0.8)
                leverage = max(1, min(50, leverage))
                
                swarm_boost = self.z9_quantum_swarm(prices_list[-9:])
                
                combined_conf = z9_conf * (1 + 0.3 * swarm_boost) * sense['regime']
                
                if combined_conf > self.meta_params['threshold'] and time.time() - self.last_trade_ts > 10:
                    coin = 'ETH' if direction > 0 else 'SOL'
                    side = 'Long' if direction > 0 else 'Short'
                    entry_p = self.latest_prices[coin]
                    
                    if entry_p > 0:
                        new_trade = {
                            'coin': coin,
                            'side': side,
                            'entry_price': entry_p,
                            'leverage': leverage,
                            'confidence': combined_conf,
                            'entry_ts': time.time()
                        }
                        self.open_trades.append(new_trade)
                        msg = f"SIGNAL: {side} {coin} @ {entry_p:.2f} | Conf: {combined_conf:.3f} | Lev: {leverage}x"
                        logger.info(msg)
                        direct_log(msg)
                        self.last_trade_ts = time.time()
                        self.trade_count += 1
                        
                        if self.trade_count % OPTIMIZE_EVERY == 0:
                            self.retrocausal_correction()
                            direct_log(f"Retrocausal optimization: New threshold {self.meta_params['threshold']:.3f}")
                
                await asyncio.sleep(0.1)
            except Exception as e:
                direct_log(f"Main Loop Error: {e}")
                import traceback
                direct_log(traceback.format_exc())
                await asyncio.sleep(1)

    async def start(self):
        try:
            await self.run()
        finally:
            for exchange in self.exchanges.values():
                try:
                    await exchange.close()
                except:
                    pass

if __name__ == '__main__':
    oracle = Z9HFTOracleV12()
    try:
        asyncio.run(oracle.start())
    except KeyboardInterrupt:
        pass
