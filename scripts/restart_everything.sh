#!/bin/bash
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
export DISPLAY=:0

echo "Killing existing processes..."
pkill -f oracle_v14.py
pkill -f react-scripts
pkill -f chromium
pkill -f npm

sleep 2

echo "Wiping old logs and metrics..."
cd /home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE
rm -rf logs/*
rm -f dashboard/public/*.json dashboard/public/*.csv
rm -f dashboard/build/*.json dashboard/build/*.csv

mkdir -p logs
touch logs/real_time_metrics.json
touch logs/trade_history.csv

# Symlink to public directory for npm start
ln -sf ../../logs/real_time_metrics.json dashboard/public/real_time_metrics.json
ln -sf ../../logs/trade_history.csv dashboard/public/trade_history.csv
ln -sf ../../logs/real_time_metrics.json dashboard/build/real_time_metrics.json
ln -sf ../../logs/trade_history.csv dashboard/build/trade_history.csv

echo "Starting Z9 Oracle engine..."
nohup venv/bin/python -u src/oracle_v14.py > logs/oracle_output.log 2>&1 &

echo "Starting React dashboard..."
cd dashboard
nohup npm start > ../logs/dashboard.log 2>&1 &

echo "Waiting for dashboard to start and Oracle to get prices..."
sleep 20

echo "Launching Chromium..."
nohup chromium --new-window http://localhost:3000 --user-data-dir=/tmp/z9_test_profile_$(date +%s) --disable-gpu --start-maximized > /dev/null 2>&1 &
