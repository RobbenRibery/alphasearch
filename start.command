#!/bin/zsh
# Localhost Search launcher.
# Starts the always-on search service + the Desktop auto-indexer.
# Run it via Terminal (which must have Full Disk Access) so the indexer can read
# your Desktop. Add this file to System Settings > General > Login Items to make
# it start automatically at every login.

cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs

# Avoid duplicate watchers.
pkill -f "watch.py" 2>/dev/null

# Start the service only if it isn't already up.
if ! curl -s -o /dev/null http://localhost:8765/ ; then
  nohup python service.py > logs/service.out.log 2>&1 &
fi

# Start the Desktop auto-indexer.
nohup python watch.py "$HOME/Desktop" > logs/watch.out.log 2>&1 &

echo "✅ Localhost Search is running (service + Desktop auto-indexer)."
echo "   You can close this Terminal window; it keeps running in the background."
sleep 2
