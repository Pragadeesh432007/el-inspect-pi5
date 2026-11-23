#!/bin/bash
# Makes EL Inspect start by itself every time the Raspberry Pi powers on.
#   Install:    bash install_autostart.sh
#   Remove:     bash install_autostart.sh remove
# Run it on the Pi, inside ~/pi_bundle, as the normal user (it asks for sudo itself).
set -e
NAME=el-inspect
APPDIR="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/venv/bin/python"
UNIT=/etc/systemd/system/$NAME.service

if [ "$1" = "remove" ]; then
  sudo systemctl disable --now $NAME 2>/dev/null || true
  sudo rm -f "$UNIT"
  sudo systemctl daemon-reload
  echo "Autostart removed."
  exit 0
fi

for f in pvel_app.py pvel_int8.onnx thresholds_int8.json classes.json; do
  [ -f "$APPDIR/$f" ] || { echo "Missing $f in $APPDIR"; exit 1; }
done
[ -x "$PY" ] || { echo "Python venv not found at $PY (create it first, see the manual, chapter 11)"; exit 1; }

sudo tee "$UNIT" > /dev/null <<EOF
[Unit]
Description=EL Inspect - solar cell inspection + on-device Llama advisor on Raspberry Pi 5
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APPDIR
ExecStart=$PY pvel_app.py --app --serve 8000 --smooth 3
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Desktop icon (only if the Pi has a desktop): opens the dashboard like an app window
if [ -d "$HOME/Desktop" ]; then
cat > "$HOME/Desktop/EL-Inspect.desktop" <<EOF2
[Desktop Entry]
Type=Application
Name=EL Inspect
Comment=Solar cell inspection dashboard
Exec=sh -c 'chromium --app=http://localhost:8000 || chromium-browser --app=http://localhost:8000 || xdg-open http://localhost:8000'
Terminal=false
Categories=Utility;
EOF2
chmod +x "$HOME/Desktop/EL-Inspect.desktop"
echo "Desktop icon created: EL Inspect"
fi

sudo systemctl daemon-reload
sudo systemctl enable --now $NAME
sleep 3
systemctl --no-pager --lines=5 status $NAME || true
echo
echo "Done. EL Inspect now starts automatically at boot."
echo "Dashboard: http://$(hostname).local:8000   (or http://$(hostname -I | awk '{print $1}'):8000)"
echo "Live log:  journalctl -u $NAME -f"
echo "Stop it before running pvel_app.py by hand (the camera can only be used by one program):  sudo systemctl stop $NAME"
