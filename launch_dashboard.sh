#!/bin/bash
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
export DISPLAY=:0

echo '[-] Starting Dashboard Server...'
cd /home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/dashboard
nohup npm start > ../logs/dashboard.log 2>&1 &

echo '[-] Waiting 15s for server cold-start...'
sleep 15

echo '[-] Launching Chromium to Physical Desktop...'
nohup chromium --new-window http://localhost:3000 --start-maximized &
