#!/bin/bash
PROJECT_DIR="/home/badgoysclub/Desktop/GEMINI/PROJECTS/Z9_ORACLE"
BACKUP_DIR="/home/badgoysclub/Desktop/GEMINI/PROJECTS"
while true; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  tar --exclude='venv' -czf "$BACKUP_DIR/Z9_ORACLE_RESTORE_$TIMESTAMP.tar.gz" -C "$PROJECT_DIR" .
  ls -t "$BACKUP_DIR"/Z9_ORACLE_RESTORE_*.tar.gz | tail -n +11 | xargs -r rm
  sleep 300
done
