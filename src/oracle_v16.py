import asyncio
import numpy as np
import time
import logging
from collections import deque
import os
import csv
import json
from datetime import datetime
import sys
import psutil
import gc

from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Kraken, Gemini

# ─── Configuration (v16.6 Real-Time IO Fix) ──────────────────────
SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
PROJECT_DIR    = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE'
LOG_DIR        = os.path.join(PROJECT_DIR, 'logs')
TRADE_LOG_PATH = os.path.join(LOG_DIR, 'trade_history.csv')
METRICS_PATH   = os.path.join(LOG_DIR, 'real_time_metrics.json')

CFG = {
    'ML_LR': 0.0072, 'ML_DECAY': 0.9965, 'ML_SCALE': 1.92,
    'RL_LR': 0.0085, 'RL_DECAY': 0.995, 'RL_DISCOUNT': 0.92,
    'KELLY_DECAY': 0.96, 'KELLY_MIN': 20,
    'RR_MULT': {'trend': 7.2, 'range': 2.9, 'volatile': 5.4, 'warmup': 4.0},
    'SL_MULT': 0.90, 'TRAIL_MULT': 0.61, 'TRAIL_ACTIVATE_MULT': 1.9,
    'BE_TRIG_MULT': 2.4, 'PARTIAL_PCT': 0.56, 'BASE_PARTIAL_R': 2.35,
    'PROFIT_PROTECTION_R': 1.5,
    'ACCEL_GATE': {'trend': -0.04, 'range': -0.14, 'volatile': -0.10},
    'CONFIDENCE_BOOST_FACTOR': 0.08,
    'WARMUP': 64, 'THRESH_BASE': 0.59,
    'REGIME_THRESH': {'trend': 0.48, 'range': 0.85, 'volatile': 0.69, 'warmup': 0.92},
    'MIN_ZSCORE': 0.81, 'COOLDOWN_BASE': 5.0, 'TICK_CONFIRM': 3,
    'MAX_POS': 850, 'LEV_BASE': 36, 'MIN_LEV': 16, 'MAX_LEV': 52,
    'MAX_OPEN': 4, 'CB_N': 4, 'CB_WAIT': 600.0,
    'MAX_HOLD': {'trend': 2040.0, 'other': 888.0},
    'REGIME_WR_WINDOW': 40, 'CORR_GUARD_THRESH': 0.65,
    'INIT_BALANCE': 1000.0, 'XE_THRESH': {'BTC-USD': 8.0, 'ETH-USD': 0.40, 'SOL-USD': 0.04},
    'SMOOTHING_ALPHA': 0.15,
    'ISOLATED_CORES': [2, 3]
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('Z9Oracle')

class MetricsManager:
    CSV_COLS = ['timestamp', 'coin', 'side', 'leverage', 'entry_price', 'exit_price', 'pnl_abs', 'pnl_pct_raw', 'pnl_pct_lev', 'hold_s', 'exit_reason', 'confidence', 'regime']
    def __init__(self):
        self.trades = deque(maxlen=2000); self.balance = CFG['INIT_BALANCE']; self.peak = CFG['INIT_BALANCE']
        self.metrics = self._default(); self.regime_wr = {r: deque(maxlen=CFG['REGIME_WR_WINDOW']) for r in ['trend','range','volatile','warmup']}
        self._init_files()
    def _default(self):
        return {'balance': CFG['INIT_BALANCE'], 'peak': CFG['INIT_BALANCE'], 'total_pnl': 0.0, 'total_trades': 0, 'win_pct': 0.0, 'accuracy': 0.0, 'win_loss_ratio': 0.0, 'sharpe': 0.0, 'drawdown': 0.0, 'avg_hold_s': 0.0, 'trades_per_min': 0.0, 'live_prices': {s: 0.0 for s in SYMBOLS}, 'live_signals': {s: {} for s in SYMBOLS}, 'last_update': datetime.now().isoformat(), 'engine': 'v16.6'}
    def _init_files(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(TRADE_LOG_PATH):
            with open(TRADE_LOG_PATH, 'w', newline='') as f: csv.writer(f).writerow(self.CSV_COLS)
        self.save()
    def record(self, coin, side, lev, entry, exit_p, pnl_abs, pnl_pct_raw, pnl_pct_lev, hold_s, reason, conf, regime):
        ts = datetime.now().isoformat(); self.balance += pnl_abs; self.peak = max(self.peak, self.balance)
        row = [ts, coin, side, lev, entry, exit_p, round(pnl_abs, 4), round(pnl_pct_raw, 6), round(pnl_pct_lev, 6), round(hold_s, 2), reason, round(conf, 4), regime]
        self.trades.append(row)
        with open(TRADE_LOG_PATH, 'a', newline='') as f: csv.writer(f).writerow(row)
        self.regime_wr[regime].append(pnl_abs > 0); self._update()
    def get_regime_wr(self, regime):
        wrs = self.regime_wr[regime]; return sum(wrs) / len(wrs) if wrs else 0.5
    def kelly_size(self, regime, atr, streak, drawdown, conf, ml_logit):
        n = len(self.trades)
        if n < CFG['KELLY_MIN']: return min(CFG['MAX_POS'], self.balance * 0.22)
        w_wins, w_loss, total_w, w = 0.0, 0.0, 0.0, 1.0
        for t in list(self.trades)[-80:]:
            pnl = float(t[7]); w_wins += max(0, pnl) * w; w_loss += abs(min(0, pnl)) * w; total_w += w; w *= CFG['KELLY_DECAY']
        W = w_wins / (w_wins + w_loss + 1e-9); avg_w = w_wins / (total_w * W + 1e-9); avg_l = w_loss / (total_w * (1 - W) + 1e-9)
        f = (W - (1 - W) / (avg_w / (avg_l + 1e-9) + 1e-9)) * 0.49 * max(0.0, 1.0 - CFG['KELLY_MIN'] / n)
        f = max(0.05, min(0.46, f)) * {'trend': 1.45, 'range': 0.50, 'volatile': 0.72, 'warmup': 0.58}.get(regime, 1.0) * max(0.64, min(1.45, conf * 2.3))
        if streak > 5: f *= 1.10
        elif streak < -4: f *= 0.78
        if drawdown > 0.10: f *= 0.18
        elif drawdown > 0.05: f *= 0.44
        return min(CFG['MAX_POS'], self.balance * f * max(0.60, min(1.40, 0.00155 / (atr or 0.001))) * max(0.84, min(1.36, 1 + ml_logit * 0.19)))
    def _update(self):
        n = len(self.trades); pnls = [float(t[6]) for t in self.trades]; wins = [p for p in pnls if p > 0]; loss = [p for p in pnls if p <= 0]
        m = self.metrics; m['balance'] = round(self.balance, 4); m['peak'] = round(self.peak, 4); m['total_trades'] = n; m['total_pnl'] = round(sum(pnls), 4); m['win_pct'] = round(len(wins) / n * 100, 2) if n else 0; m['accuracy'] = round(len(wins) / n, 4) if n else 0; m['avg_hold_s'] = round(np.mean([float(t[9]) for t in self.trades]), 2) if n else 0; avg_win = np.mean(wins) if wins else 0; avg_loss = abs(np.mean(loss)) if loss else 1e-8; m['win_loss_ratio'] = round(avg_win / avg_loss, 4)
        if n > 1: m['sharpe'] = round((np.mean(pnls) / (np.std(pnls) + 1e-8)) * np.sqrt(365 * 24 * 60), 4)
        m['drawdown'] = round((self.peak - self.balance) / (self.peak + 1e-9), 6)
    def save(self):
        try:
            self.metrics['last_update'] = datetime.now().isoformat()
            # USE ABSOLUTE PATH FOR ATOMIC SAVE
            tmp_path = METRICS_PATH + '.tmp'
            with open(tmp_path, 'w') as f: json.dump(self.metrics, f, indent=4)
            os.rename(tmp_path, METRICS_PATH)
        except Exception as e: logger.error(f'Metrics save error: {e}')

class Z9OracleHFT:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        try: self.process.cpu_affinity(CFG['ISOLATED_CORES']); logger.info(f'[HFT] CPU Pinning enabled on cores: {CFG["ISOLATED_CORES"]}')
        except Exception as e: logger.warning(f'[HFT] CPU Pinning failed: {e}')
        try: self.process.nice(-15)
        except: pass
        self.mm = MetricsManager(); self.price_buffers = {s: deque(maxlen=512) for s in SYMBOLS}; self.latest_prices = {s: 0.0 for s in SYMBOLS}; self.exchange_mids = {s: {} for s in SYMBOLS}; self.atr = {s: 0.001 for s in SYMBOLS}; self.prev_sigs = {s: None for s in SYMBOLS}; self.open_trades = []; self.last_sig_ts = {s: 0.0 for s in SYMBOLS}; self.cb_loss = {s: 0 for s in SYMBOLS}; self.cb_end = {s: 0.0 for s in SYMBOLS}; self.streak = 0; self.tick_count = 0
        self.ml_weights = {'zscore': 0.48, 'accel': 1.18, 'tc': 0.31, 'bConf': 0.42, 'regimeTrend': 0.71, 'regimeVol': -0.31, 'vol': -0.19, 'bias': -0.09}
        self.rl_weights = {'bias': 0.12, 'regimeTrend': 0.68, 'regimeVol': -0.41, 'conf': 1.25, 'mlLogit': 0.92, 'streak': 0.35, 'drawdown': -1.18}
    def _calc_atr(self, sym):
        buf = list(self.price_buffers[sym]); return max(0.0003, min(0.003, sum(abs(buf[i+1] - buf[i]) / (buf[i] + 1e-9) for i in range(len(buf)-1)) / (len(buf)-1))) if len(buf) > 15 else 0.001
    def _classify_regime(self, atr, zscore): return "volatile" if atr * 100 > 0.165 else ("trend" if abs(zscore) > 1.30 else "range")
    def _tick_confirm(self, sym, dir):
        buf = list(self.price_buffers[sym]); sl = buf[-CFG['TICK_CONFIRM']:]; c = 0
        for i in range(1, len(sl)):
            if (dir > 0 and sl[i] > sl[i-1]) or (dir < 0 and sl[i] < sl[i-1]): c += 1
        return c
    def _regime_thresh_adj(self, regime): return max(0.40, min(0.96, CFG['REGIME_THRESH'].get(regime, CFG['THRESH_BASE']) + max(-0.08, min(0.12, 0.5 * (0.60 - self.mm.get_regime_wr(regime))))))
    def _ml_fuse(self, features):
        logit = self.ml_weights['bias']
        for k, v in features.items(): logit += v * self.ml_weights.get(k, 0)
        return {'dir': 1 if logit > 0 else -1, 'conf': max(0.54, min(0.99, 0.5 + 0.5 * np.tanh(logit * CFG['ML_SCALE']))), 'logit': logit}
    def _rl_scale_pos(self, state):
        score = self.rl_weights['bias']
        for k, v in state.items(): score += v * self.rl_weights.get(k, 0)
        return max(0.55, min(1.85, 0.55 + 1.25 / (1 + np.exp(-score))))
    def _ensemble(self, sym):
        buf = list(self.price_buffers[sym]); n = len(buf)
        if n < 28: return None
        sma = np.mean(buf[-12:]); mma = np.mean(buf[-36:-12]) if n >= 36 else sma; mu = np.mean(buf[-22:]); std = np.std(buf[-22:]) + 1e-9; mom = sma - mma; zscore = mom / std; vol = std / (mu + 1e-9); prev_z = (sma - mma) / (np.std(buf[-30:-22]) + 1e-9) if n >= 30 else zscore; accel = zscore - prev_z; direction = int(np.sign(mom)); tc = self._tick_confirm(sym, direction); m_conf = min(0.99, 0.54 + 0.46 * np.tanh(abs(zscore) * 1.10) + max(0, (tc - 3) * 0.035)); b_dir, b_conf = 0, 0.0
        if sym != 'BTC-USD':
            btc_buf = list(self.price_buffers['BTC-USD'])
            if len(btc_buf) >= 28:
                b_mom = np.mean(btc_buf[-12:]) - np.mean(btc_buf[-36:-12] if len(btc_buf) >= 36 else btc_buf[-12:]); b_dir = int(np.sign(b_mom)); b_conf = min(0.93, (0.54 + 0.46 * np.tanh(abs(b_mom/(np.std(btc_buf[-22:])+1e-9)) * 1.10)) * 0.95)
        regime = self._classify_regime(self.atr[sym], zscore); bull, bear, tw = 0.0, 0.0, 0.0
        if direction != 0: w = 0.64; bull += (m_conf if direction > 0 else 0) * w; bear += (m_conf if direction < 0 else 0) * w; tw += w
        if b_dir != 0: w = 0.36; bull += (b_conf if b_dir > 0 else 0) * w; bear += (b_conf if b_dir < 0 else 0) * w; tw += w
        if tw == 0 or bull == bear: return None
        features = {'zscore': zscore, 'accel': accel, 'tc': tc, 'bConf': b_conf, 'regimeTrend': 1.0 if regime == "trend" else 0.0, 'regimeVol': 1.0 if regime == "volatile" else 0.0, 'vol': vol * -1}
        fused = self._ml_fuse(features); conf = max(bull, bear) / tw; regime_adj = {'trend': 1.18, 'range': 0.79, 'volatile': 0.86}.get(regime, 1.0)
        conf = max(0.0, min(0.99, conf * regime_adj * fused['conf']))
        if len([t for t in list(self.mm.trades)[-10:] if t[12] == regime and float(t[6]) > 0]) >= 2: conf = min(0.99, conf * (1 + CFG['CONFIDENCE_BOOST_FACTOR']))
        return {'dir': fused['dir'], 'conf': conf, 'zscore': zscore, 'vol': vol, 'tc': tc, 'accel': accel, 'bDir': b_dir, 'bConf': b_conf, 'regime': regime, 'mlConf': fused['conf'], 'mlLogit': fused['logit'], 'features': features}
    def _check_exit(self, op, price, now):
        entry = op['entry_price']; live_atr = self.atr[op['sym']]; live_sl = max(0.0012, CFG['SL_MULT'] * live_atr); live_trail = live_sl * CFG['TRAIL_MULT']; pnl = (price - entry) / entry if op['side'] == 'Long' else (entry - price) / entry
        if not op['trailActive'] and pnl >= op['trailActPct']: op['trailActive'] = True
        if not op['be'] and pnl >= op['beTrigPct']: op['trail'] = entry * (1.0005 if op['side'] == 'Long' else 0.9995); op['be'] = True
        if op['trailActive']:
            if op['side'] == 'Long' and price > op['peak']:
                op['peak'] = price; nt = price * (1 - (live_trail * 0.6 if pnl >= CFG['PROFIT_PROTECTION_R'] * op['slPct'] else live_trail)); op['trail'] = max(nt, op['trail'])
            elif op['side'] == 'Short' and price < op['trough']:
                op['trough'] = price; nt = price * (1 + (live_trail * 0.6 if pnl >= CFG['PROFIT_PROTECTION_R'] * op['slPct'] else live_trail)); op['trail'] = min(nt, op['trail'])
        if (op['side'] == 'Long' and price <= op['trail']) or (op['side'] == 'Short' and price >= op['trail']): return pnl, True, 'trail'
        if not op['partialTaken'] and pnl >= op['dynamicPartialR'] * op['slPct']: return pnl, True, 'PARTIAL'
        if pnl >= op['tpPct']: return pnl, True, 'TP'
        if pnl <= -op['slPct']: return pnl, True, 'SL'
        if now - op['ts'] >= op['maxHold']: return pnl, True, 't/o'
        return pnl, False, ''
    def _close(self, op, exit_p, pnl_raw, now, reason):
        if reason == 'PARTIAL':
            pnl_abs = op['pos'] * CFG['PARTIAL_PCT'] * pnl_raw * op['lev']; self.mm.record(op['sym'], op['side'], op['lev'], op['entry_price'], exit_p, pnl_abs, pnl_raw, pnl_raw * op['lev'], (now - op['ts']), 'PARTIAL', op['conf'], op['regime']); op['pos'] *= (1 - CFG['PARTIAL_PCT']); op['partialTaken'] = True; op['trail'] = op['entry_price'] * (1.0010 if op['side'] == 'Long' else 0.9990); op['be'] = True; op['trailActive'] = True; return
        pnl_abs = op['pos'] * pnl_raw * op['lev']; self.mm.record(op['sym'], op['side'], op['lev'], op['entry_price'], exit_p, pnl_abs, pnl_raw, pnl_raw * op['lev'], now - op['ts'], reason, op['conf'], op['regime']); self.open_trades.remove(op); self.streak = self.streak + 1 if pnl_abs > 0 else (self.streak - 1 if self.streak < 0 else -1)
        if pnl_abs < 0:
            self.cb_loss[op['sym']] += 1
            if self.cb_loss[op['sym']] >= CFG['CB_N']: self.cb_end[op['sym']] = now + CFG['CB_WAIT']
        else: self.cb_loss[op['sym']] = 0
    def _open(self, sym, side, price, sig, now):
        regime = sig['regime']; sl_pct = max(0.0012, CFG['SL_MULT'] * self.atr[sym]); dd = (self.mm.peak - self.mm.balance) / (self.mm.peak + 1e-9); rl_state = {'regimeTrend': 1.0 if regime == 'trend' else 0.0, 'regimeVol': 1.0 if regime == 'volatile' else 0.0, 'conf': sig['conf'], 'mlLogit': sig['mlLogit'], 'streak': self.streak, 'drawdown': dd}
        pos = min(CFG['MAX_POS'], self.mm.kelly_size(regime, self.atr[sym], self.streak, dd, sig['conf'], sig['mlLogit']) * self._rl_scale_pos(rl_state)); lev = max(CFG['MIN_LEV'], min(CFG['MAX_LEV'], int(CFG['LEV_BASE'] * sig['conf'] * (1.18 if regime == "trend" else 1.0))))
        self.open_trades.append({'sym': sym, 'side': side, 'entry_price': price, 'lev': lev, 'pos': pos, 'conf': sig['conf'], 'ts': now, 'peak': price, 'trough': price, 'trail': price * (1 - sl_pct if side == 'Long' else 1 + sl_pct), 'trailActive': False, 'be': False, 'partialTaken': False, 'slPct': sl_pct, 'tpPct': sl_pct * CFG['RR_MULT'].get(regime, 4.5), 'trailActPct': sl_pct * CFG['TRAIL_ACTIVATE_MULT'], 'beTrigPct': sl_pct * CFG['BE_TRIG_MULT'], 'dynamicPartialR': CFG['BASE_PARTIAL_R'] * (0.8 + 0.4 * sig['conf']), 'maxHold': CFG['MAX_HOLD']['trend'] if regime == 'trend' else CFG['MAX_HOLD']['other'], 'regime': regime, 'features': sig['features'], 'rl_state': rl_state})
        logger.info(f'[OPEN] {side:5s} {sym:8s} @ {price:.4f} | Conf: {sig["conf"]:.3f} | Lev: {lev}x | Pos: ${pos:.0f}')

    async def trade_callback(self, trade, receipt_timestamp):
        sym = str(trade.symbol); price = float(trade.price); exch = str(trade.exchange); now = time.time(); alpha = CFG['SMOOTHING_ALPHA']
        self.latest_prices[sym] = (alpha * price + (1 - alpha) * self.latest_prices[sym]) if self.latest_prices.get(sym, 0) > 0 else price
        self.price_buffers[sym].append(self.latest_prices[sym]); self.atr[sym] = self._calc_atr(sym)
        self.mm.metrics['live_prices'][sym] = self.latest_prices[sym]
        sig = self._ensemble(sym)
        if sig:
            self.mm.metrics['live_signals'][sym] = {'conf': round(sig['conf'], 4), 'dir': sig['dir'], 'regime': sig['regime'], 'lev': max(CFG['MIN_LEV'], min(CFG['MAX_LEV'], int(CFG['LEV_BASE'] * sig['conf'] * (1.18 if sig['regime'] == "trend" else 1.0))))}
        for op in self.open_trades[:]:
            if op['sym'] == sym:
                pnl, exit, reason = self._check_exit(op, self.latest_prices[sym], now)
                if exit: self._close(op, self.latest_prices[sym], pnl, now, reason)
        if (len(self.price_buffers[sym]) >= CFG['WARMUP'] and now - self.last_sig_ts[sym] >= CFG['COOLDOWN_BASE'] and now >= self.cb_end[sym] and len(self.open_trades) < CFG['MAX_OPEN'] and not any(o['sym'] == sym for o in self.open_trades)):
            if sig and sig['accel'] >= CFG['ACCEL_GATE'].get(sig['regime'], 0):
                prev = self.prev_sigs[sym]; persistent = prev and prev['dir'] == sig['dir'] and prev['conf'] >= 0.45; self.prev_sigs[sym] = sig
                if persistent and sig['conf'] >= self._regime_thresh_adj(sig['regime']) and abs(sig['zscore']) >= CFG['MIN_ZSCORE']:
                    if sym == 'BTC-USD' or not (next((o for o in self.open_trades if o['sym'] == 'BTC-USD'), None) and (1 if next((o for o in self.open_trades if o['sym'] == 'BTC-USD'), None)['side'] == 'Long' else -1) != sig['dir']):
                        self._open(sym, 'Long' if sig['dir'] > 0 else 'Short', self.latest_prices[sym], sig, now); self.last_sig_ts[sym] = now
        self.tick_count += 1
        if self.tick_count % 1000 == 0: logger.info(f'[HEARTBEAT] Mem: {self.process.memory_info().rss / 1024 / 1024:.2f} MB | Ticks: {self.tick_count}'); gc.collect()

async def metrics_save_loop(mm):
    while True: mm.save(); await asyncio.sleep(0.5)

async def run_fh(fh, mm):
    fh.running = True; loop = asyncio.get_running_loop(); asyncio.create_task(metrics_save_loop(mm))
    for feed in fh.feeds: feed.start(loop)
    while fh.running: await asyncio.sleep(1)

def main():
    logger.info('═'*65 + '\n  Z9 ORACLE HFT v16.6 — IO SYNC BUILD\n' + '═'*65)
    oracle = Z9OracleHFT(); fh = FeedHandler(); channels = [TRADES]; callbacks = {TRADES: oracle.trade_callback}
    fh.add_feed(Kraken(symbols=SYMBOLS, channels=channels, callbacks=callbacks)); fh.add_feed(Gemini(symbols=SYMBOLS, channels=channels, callbacks=callbacks))
    try: asyncio.run(run_fh(fh, oracle.mm))
    except KeyboardInterrupt: logger.info('Stopping Z9 Oracle...')
    except Exception as e: logger.error(f'Fatal error: {e}')

if __name__ == '__main__': main()
