#!/bin/bash
# Detect the trycloudflare.com URL from luci-tunnel and notify via Telegram.
# Runs as ExecStartPost in luci-tunnel.service.

CHAT_ID=8757958279

# Wait for tunnel to establish and URL to appear in journal
sleep 8

URL=$(journalctl --user -u luci-tunnel --since "30 seconds ago" --no-pager 2>/dev/null \
    | grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com' | tail -1)

# Fallback: read from /proc if journal didn't have it yet
if [ -z "$URL" ]; then
    PID=$(systemctl --user show luci-tunnel -p MainPID --value 2>/dev/null)
    if [ -n "$PID" ] && [ "$PID" != "0" ]; then
        URL=$(strings /proc/$PID/fd/1 2>/dev/null \
            | grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com' | tail -1)
    fi
fi

source /home/equansa00/beast/workspace/.env

if [ -n "$URL" ]; then
    curl -s "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d "chat_id=${CHAT_ID}" \
        -d "text=🌐 LUCI web UI is available at: ${URL}" > /dev/null
    echo "Sent URL to Telegram: $URL"
else
    curl -s "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d "chat_id=${CHAT_ID}" \
        -d "text=⚠️ LUCI tunnel started but could not detect URL. Check laptop." > /dev/null
    echo "Could not detect tunnel URL"
fi
