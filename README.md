# whisper-dictate

System-wide voice typing for Linux. Press a shortcut in any app, speak, and
your words are typed into whatever window has focus. Speech recognition runs
**locally** with OpenAI's Whisper model, so nothing leaves your machine.

- Works on **Wayland and X11**: KDE Plasma, GNOME, Hyprland, Sway and others.
- **Hold-to-talk** or **toggle** (press to start, press again to stop).
- Uses your GPU when it can: **NVIDIA** through CUDA (faster-whisper),
  **AMD and Intel Arc** through Vulkan (whisper.cpp). Falls back to the CPU.
- English to start (the model can do other languages later).

## Install (CachyOS / Arch)

```bash
git clone https://github.com/0XZEROX0/whisper-dictate
cd whisper-dictate
./install.sh
```

The installer:

1. Installs `portaudio ydotool wl-clipboard xdotool xclip libnotify` with pacman.
2. Adds you to the `input` group and adds a udev rule for `/dev/uinput`.
   This is what lets the app see the shortcut and type for you on Wayland.
3. Creates a Python environment in `~/.local/share/whisper-dictate/venv`,
   with CUDA libraries if it finds an NVIDIA GPU.
4. Installs and enables a systemd user service that starts with your desktop.

If it added you to the `input` group, **log out and back in once**.
Then run `whisper-dictate doctor` to check everything.

The first start downloads the Whisper model (about 480 MB for `small.en`).

### AMD or Intel GPU

faster-whisper only uses NVIDIA GPUs. On AMD or Intel Arc, install whisper.cpp
built with Vulkan (for example the AUR package `whisper.cpp-vulkan`), then:

```bash
whisper-dictate download-model          # fetches ggml-small.en.bin
systemctl --user restart whisper-dictate
```

With `engine.name = "auto"` the app uses whisper.cpp whenever `whisper-cli`
is installed and no NVIDIA GPU is present.

## Use

Default shortcut: **Ctrl+Alt+Space**.

- **toggle** mode (default): press once, speak, press again.
- **hold** mode: hold the keys while you speak, let go to type.

A notification shows when it is listening and when it is done.

Other commands:

```
whisper-dictate toggle | start | stop | cancel | status
whisper-dictate doctor      # check your setup
whisper-dictate init-config # write the config file with comments
```

You can also bind `whisper-dictate toggle` to a key in your desktop's own
shortcut settings (KDE: System Settings > Shortcuts > Add Command) and set
`hotkey.combo = ""` to switch off the built-in listener.

## Configure

`~/.config/whisper-dictate/config.toml` (create it with `whisper-dictate init-config`):

```toml
[hotkey]
combo = "ctrl+alt+space"   # or "KEY_RIGHTCTRL", "super+h", "KEY_F9" ...
mode = "toggle"            # or "hold"

[engine]
name = "auto"              # "faster-whisper" or "whisper-cpp"
model = "small.en"         # tiny.en, base.en, small.en, medium.en, large-v3-turbo
device = "auto"            # faster-whisper: "cuda" or "cpu"

[output]
typer = "auto"             # ydotool, wtype, xdotool, clipboard
trailing_space = true
```

Restart after editing: `systemctl --user restart whisper-dictate`.

**Picking a model:** `small.en` is a good default on any recent CPU.
With a GPU, try `medium.en` or `large-v3-turbo` for better accuracy.
On a slow laptop use `base.en`.

## How it works

| Part | Wayland | X11 |
|---|---|---|
| Shortcut | reads the keyboard through evdev (`/dev/input`) | same |
| Typing | `ydotool` (KDE, GNOME), `wtype` (Hyprland, Sway) | `xdotool` |
| Fallback | copy to clipboard and press Ctrl+V | same |

Wayland does not let normal apps grab global shortcuts or type into other
windows, so the app talks to the kernel's input layer instead. That works on
every desktop and is why the `input` group is needed.

## Troubleshooting

- **Nothing happens on the shortcut:** run `whisper-dictate doctor`. Usually
  you haven't logged out since joining the `input` group.
- **Text doesn't appear:** make sure ydotool's daemon runs:
  `systemctl --user enable --now ydotool`.
- **Hyprland / Sway don't start the service:** add
  `exec-once = systemctl --user start whisper-dictate` to your config.
- **Logs:** `journalctl --user -u whisper-dictate -f`

## Develop

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,faster-whisper]'
pytest
whisper-dictate -v daemon
```

## License

MIT
