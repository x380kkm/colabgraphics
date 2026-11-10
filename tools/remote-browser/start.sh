#!/usr/bin/env bash
# Start a virtual display, a window manager, an ordinary Chromium on it, and the remote view.
#
#   PROFILE=/root/browser-profile REMOTE_VIEW_CODE=... tools/remote-browser/start.sh https://example.com
#
# The browser is started the way a person would start it: no automation flags and no
# debugging port. PROFILE holds cookies and logins, so it must live outside the repository.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
SIZE="${SIZE:-1280x800}"
PORT="${PORT:-8765}"
URL="${1:-about:blank}"
: "${PROFILE:?set PROFILE to a directory outside the repository}"
: "${REMOTE_VIEW_CODE:?set REMOTE_VIEW_CODE to the access code for the remote view}"
CHROME="${CHROME:-$(node -e "console.log(require('playwright').chromium.executablePath())")}"

RUN=/tmp/remote-browser
FB="$RUN/fb-$DISPLAY_NUM"
mkdir -p "$RUN" "$FB" "$PROFILE"
chmod 700 "$PROFILE"
export DISPLAY=":$DISPLAY_NUM"

running() { pgrep -f "$1" >/dev/null 2>&1; }

if ! running "Xvfb :$DISPLAY_NUM "; then
  setsid nohup Xvfb ":$DISPLAY_NUM" -screen 0 "${SIZE}x24" -fbdir "$FB" -nolisten tcp >"$RUN/xvfb.log" 2>&1 &
  for _ in $(seq 50); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.1; done
fi
if ! running "openbox"; then
  setsid nohup openbox >"$RUN/openbox.log" 2>&1 &
fi
sleep 0.5

if ! running "user-data-dir=$PROFILE"; then
  setsid nohup "$CHROME" --user-data-dir="$PROFILE" --no-sandbox --password-store=basic \
    --no-first-run --no-default-browser-check --disable-dev-shm-usage \
    --window-position=0,0 --window-size="${SIZE/x/,}" --start-maximized \
    ${CHROME_FLAGS:-} "$URL" >"$RUN/chrome.log" 2>&1 &
fi

if ! running "remote_view.py --display :$DISPLAY_NUM "; then
  setsid nohup python3 "$HERE/remote_view.py" --display ":$DISPLAY_NUM" --fb "$FB/Xvfb_screen0" --port "$PORT" \
    >"$RUN/remote_view.log" 2>&1 &
fi
sleep 1.5
echo "display :$DISPLAY_NUM (${SIZE}), remote view on port $PORT"
