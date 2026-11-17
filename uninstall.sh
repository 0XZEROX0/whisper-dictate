#!/usr/bin/env bash
set -euo pipefail
systemctl --user disable --now whisper-dictate.service 2>/dev/null || true
rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/whisper-dictate.service" "$HOME/.local/bin/whisper-dictate"
rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/whisper-dictate/venv"
rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/applications/whisper-dictate-settings.desktop"
rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps/whisper-dictate.svg"
systemctl --user daemon-reload
echo "Removed. Your config and downloaded models were kept:"
echo "  ${XDG_CONFIG_HOME:-$HOME/.config}/whisper-dictate"
echo "  ${XDG_DATA_HOME:-$HOME/.local/share}/whisper-dictate/models"
echo "Optional: sudo rm /etc/udev/rules.d/60-whisper-dictate-uinput.rules"
