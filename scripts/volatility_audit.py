import time
import json
import subprocess
import os
from datetime import datetime

SCREENSHOT_DIR = '/home/badgoysclub/Desktop/GEMINI/screenshots/audit'
LOG_FILE = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/logs/audit_log.json'
METRICS_FILE = '/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/logs/real_time_metrics.json'

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
audit_results = []

print('Starting 5-minute volatility audit (Fixed Script)...')

for i in range(16):
    timestamp = datetime.now().isoformat()
    img_path = f'{SCREENSHOT_DIR}/audit_{i:02d}.png'
    
    # Capture Screenshot
    env = os.environ.copy()
    env['XDG_RUNTIME_DIR'] = '/run/user/1000'
    env['WAYLAND_DISPLAY'] = 'wayland-0'
    env['DISPLAY'] = ':0'
    subprocess.run(['grim', img_path], env=env)
    
    # Read Metrics
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
            prices = metrics.get('live_prices', {})
    except Exception as e:
        prices = {'error': str(e)}
        
    audit_results.append({
        'timestamp': timestamp,
        'index': i,
        'prices': prices,
        'screenshot': img_path
    })
    
    btc = prices.get('BTC-USD', 'N/A')
    eth = prices.get('ETH-USD', 'N/A')
    sol = prices.get('SOL-USD', 'N/A')
    print(f'Audit {i+1}/16: BTC {btc} | ETH {eth} | SOL {sol}')
    
    with open(LOG_FILE, 'w') as f:
        json.dump(audit_results, f, indent=4)
        
    if i < 15:
        time.sleep(20)

print('Audit complete.')
