#!/bin/bash
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
export DISPLAY=:0

cd /home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/dashboard
# Use full path for serve
nohup /usr/bin/serve -s build -l 3000 > ../logs/dashboard.log 2>&1 &

sleep 5
chromium --new-window http://localhost:3000 --start-maximized </dev/null > /dev/null 2>&1 &
