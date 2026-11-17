#!/usr/bin/env bash
# Install whisper-dictate for the current user (Arch / CachyOS first, other distros best effort).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/whisper-dictate"
VENV="$DATA_DIR/venv"
BIN_DIR="$HOME/.local/bin"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }

# 1. System packages ---------------------------------------------------------
if command -v pacman >/dev/null; then
  say "Installing system packages with pacman (sudo)"
  sudo pacman -S --needed --noconfirm python portaudio ydotool wl-clipboard xdotool xclip libnotify
else
  warn "Not an Arch-based system. Install these yourself: python3 (3.11+), python3-venv, portaudio, ydotool, wl-clipboard, xdotool, xclip, libnotify (notify-send)."
fi

# 2. Permissions for the global hotkey and for typing ------------------------
if ! id -nG "$USER" | grep -qw input; then
  say "Adding $USER to the 'input' group (needed to see the hotkey and to type)"
  sudo usermod -aG input "$USER"
  NEED_RELOGIN=1
fi
if [ ! -f /etc/udev/rules.d/60-whisper-dictate-uinput.rules ]; then
  say "Installing udev rule so the input group can use /dev/uinput"
  sudo install -m 644 "$REPO_DIR/packaging/60-whisper-dictate-uinput.rules" /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger --name-match=uinput || true
fi
# udevadm trigger doesn't always re-apply permissions to an existing node,
# so set them directly for this boot (the rule covers future boots).
if [ -e /dev/uinput ] && [ "$(stat -c %G /dev/uinput)" != input ]; then
  sudo modprobe uinput 2>/dev/null || true
  sudo chgrp input /dev/uinput && sudo chmod 660 /dev/uinput
fi

# 3. Python environment ------------------------------------------------------
say "Creating Python environment in $VENV"
mkdir -p "$DATA_DIR" "$BIN_DIR" "$UNIT_DIR"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$REPO_DIR[faster-whisper]"

LD_EXTRA=""
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  say "NVIDIA GPU found: installing CUDA libraries for faster-whisper"
  "$VENV/bin/pip" install --quiet nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
  LD_EXTRA=$("$VENV/bin/python" -c 'import os, nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(os.path.dirname(a.__file__) + ":" + os.path.dirname(b.__file__))')
else
  if command -v whisper-cli >/dev/null; then
    say "whisper.cpp found: AMD/Intel GPU acceleration will be used"
  else
    warn "No NVIDIA GPU. Speech will run on the CPU, which is fine for small.en."
    warn "For AMD or Intel Arc GPU speed, install whisper.cpp built with Vulkan (e.g. AUR whisper.cpp-vulkan),"
    warn "then run: whisper-dictate download-model"
  fi
fi

# 4. Launcher ----------------------------------------------------------------
cat > "$BIN_DIR/whisper-dictate" <<LAUNCHER
#!/usr/bin/env bash
${LD_EXTRA:+export LD_LIBRARY_PATH="$LD_EXTRA\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"}
exec "$VENV/bin/whisper-dictate" "\$@"
LAUNCHER
chmod +x "$BIN_DIR/whisper-dictate"

# 5. Config and services -----------------------------------------------------
"$BIN_DIR/whisper-dictate" init-config >/dev/null
install -m 644 "$REPO_DIR/packaging/whisper-dictate.service" "$UNIT_DIR/"
systemctl --user daemon-reload
if systemctl --user list-unit-files ydotool.service >/dev/null 2>&1; then
  systemctl --user reset-failed ydotool.service 2>/dev/null || true
  systemctl --user enable ydotool.service || warn "could not enable ydotool.service"
fi
systemctl --user enable whisper-dictate.service

say "Installed."
echo "  Config:   ${XDG_CONFIG_HOME:-$HOME/.config}/whisper-dictate/config.toml"
echo "  Check:    whisper-dictate doctor"
if [ "${NEED_RELOGIN:-0}" = 1 ]; then
  warn "Reboot (or log out and back in) so the 'input' group applies; dictation then starts automatically."
else
  systemctl --user restart ydotool.service 2>/dev/null || true
  systemctl --user restart whisper-dictate.service
  echo "  Running:  press Ctrl+Alt+Space, speak, press it again."
fi
