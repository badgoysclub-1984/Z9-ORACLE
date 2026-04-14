#!/bin/bash

PROJECT_DIR="/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE"
LOGS_DIR="$PROJECT_DIR/logs"
DASHBOARD_DIR="$PROJECT_DIR/dashboard"
VENV_PYTHON="/home/badgoysclub/z9_hft_env/bin/python3"

export DISPLAY=:0
export PYTHONUNBUFFERED=1

mkdir -p "$LOGS_DIR"

# Kill existing processes
pkill -9 -f z9_hft_oracle_v12.py || true
pkill -9 -f "python3 -m http.server 3000" || true
pkill -9 -f "chromium --app=http://localhost:3000" || true
pkill -9 -f "node /usr/bin/serve" || true

echo "--- Starting Z9 ORACLE Build ---"

# Start Oracle
cd "$PROJECT_DIR/src"
nohup $VENV_PYTHON -u z9_hft_oracle_v12.py > "$LOGS_DIR/oracle_output.log" 2>&1 &
echo "Oracle process started."

# Prepare Hard Links
cd "$DASHBOARD_DIR/build"
rm real_time_metrics.json trade_history.csv 2>/dev/null
ln "$LOGS_DIR/real_time_metrics.json" real_time_metrics.json 2>/dev/null || cp "$LOGS_DIR/real_time_metrics.json" real_time_metrics.json
ln "$LOGS_DIR/trade_history.csv" trade_history.csv 2>/dev/null || cp "$LOGS_DIR/trade_history.csv" trade_history.csv

# Start Server (Use serve if available, otherwise python3)
if command -v serve &> /dev/null; then
    nohup serve -s . -l 3000 > "$LOGS_DIR/dashboard.log" 2>&1 &
else
    nohup python3 -m http.server 3000 > "$LOGS_DIR/dashboard.log" 2>&1 &
fi
echo "Dashboard server started on port 3000."

# Launch Browser
sleep 5
(chromium --app=http://localhost:3000 --start-maximized --no-first-run --disable-infobars || chromium-browser --app=http://localhost:3000 --start-maximized --no-first-run) &
echo "Browser launch command issued."

while true; do
    # Health check
    pgrep -f z9_hft_oracle_v12.py > /dev/null || {
        echo "$(date) | Oracle died, restarting..."
        cd "$PROJECT_DIR/src"
        nohup $VENV_PYTHON -u z9_hft_oracle_v12.py >> "$LOGS_DIR/oracle_output.log" 2>&1 &
    }
    sleep 30
done
