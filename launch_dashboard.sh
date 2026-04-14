#!/bin/bash
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0
export DISPLAY=:0

cd /home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE/dashboard
npm start </dev/null > ../logs/dashboard.log 2>&1 &
sleep 15
chromium --new-window http://localhost:3000 --start-maximized </dev/null > /dev/null 2>&1 &
