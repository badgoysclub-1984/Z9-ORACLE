#!/bin/bash
pkill -f z9_hft_oracle_v12.py || true
sleep 1
rm -f /home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/logs/trade_history.csv /home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/logs/real_time_metrics.json
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
export DISPLAY=:0
cd /home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE
nohup venv/bin/python src/z9_hft_oracle_v12.py </dev/null > logs/oracle_output.log 2>&1 &
