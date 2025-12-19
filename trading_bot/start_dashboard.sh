#!/bin/bash
# Start Trading Bot Dashboard
# This script starts both the Next.js frontend and the Python bot with API server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Trading Bot Dashboard..."

# Check if node_modules exists
if [ ! -d "dashboard/node_modules" ]; then
    echo "📦 Installing dashboard dependencies..."
    (cd dashboard && npm install)
fi

# Start Next.js dashboard in background
echo "🌐 Starting Next.js dashboard on http://localhost:3000"
(cd dashboard && npm run dev) &
DASHBOARD_PID=$!

# Wait for dashboard to start
sleep 3

# Start the trading bot (which includes the API server)
echo "🤖 Starting trading bot with API server on http://localhost:5000"
python3 "$SCRIPT_DIR/bot_advanced.py"

# Cleanup on exit
trap "kill $DASHBOARD_PID 2>/dev/null" EXIT
