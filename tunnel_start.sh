#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 1. Clean up any existing background instances
pkill -f "src/app.py" 2>/dev/null || true
killall cloudflared 2>/dev/null || true

# 2. Trap SIGINT (Ctrl+C) and SIGTERM to kill both background jobs on exit
cleanup() {
    echo -e "\n[*] Shutting down application and tunnel..."
    kill "$APP_PID" 2>/dev/null || true
    kill "$TUNNEL_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 3. Start the application in the background
echo "[*] Starting Air Export Pricing Engine on port 5050..."
./run.sh serve > /tmp/pricing_app.log 2>&1 &
APP_PID=$!

# Wait for local app to become ready
sleep 2
while ! curl -s http://127.0.0.1:5050 > /dev/null; do
    echo "    Waiting for app to respond on 127.0.0.1:5050..."
    sleep 1
done
echo "[✓] Local app is running on http://127.0.0.1:5050"

# 4. Start cloudflared tunnel
echo "[*] Connecting Cloudflare Tunnel..."
rm -f /tmp/cloudflared.log

# Force http2 protocol for reliable DNS and connection stability
cloudflared tunnel --protocol http2 --url http://127.0.0.1:5050 > /tmp/cloudflared.log 2>&1 &
TUNNEL_PID=$!

# Wait for the actual HTTPS trycloudflare URL to appear in the log
echo "    Establishing tunnel with Cloudflare edge..."
TUNNEL_URL=""
for i in {1..20}; do
    TUNNEL_URL=$(grep -Eo 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -n 1 || true)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    echo "[!] Could not automatically detect URL. Check /tmp/cloudflared.log:"
    cat /tmp/cloudflared.log
    exit 1
fi

echo "===================================================================="
echo "  🚀 TUNNEL IS LIVE!"
echo "  Public URL: $TUNNEL_URL"
echo "  Local App:  http://127.0.0.1:5050"
echo "===================================================================="
echo "NOTE: Each time you restart the tunnel, a NEW unique URL is generated."
echo "Press [Ctrl + C] to stop both the application and the tunnel."

# Keep script running to maintain the processes
wait "$TUNNEL_PID"
