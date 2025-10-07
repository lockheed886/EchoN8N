# Python 3.10 Virtual Environment Setup

## Problem
The `TTS` (Coqui TTS) package doesn't support Python 3.13. It requires Python 3.9-3.11.

## Solution
I've set up a Python 3.10 virtual environment using pyenv.

## What's Been Done

✅ **Installed pyenv** - Python version manager at `~/.pyenv`
✅ **Installed Python 3.10.16** - via pyenv
✅ **Created venv310** - virtual environment in `/home/wolf/voice2/venv310`
✅ **Created setup script** - `setup_venv.sh` for easy reinstallation

## Quick Start

### Complete the Installation

The TTS installation was interrupted. To complete it, run:

```bash
cd /home/wolf/voice2
./setup_venv.sh
```

**OR** manually:

```bash
cd /home/wolf/voice2
source venv310/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Expected Download Sizes
- **torch**: ~888 MB
- **CUDA libraries**: ~600+ MB  
- **Other dependencies**: ~200 MB
- **Total**: ~1.5-2 GB

Installation typically takes 15-30 minutes depending on internet speed.

## Daily Usage

### Activate the Environment

```bash
cd /home/wolf/voice2
source venv310/bin/activate
```

### Run Your Streamlit App

```bash
streamlit run streamlit_app.py
```

### Deactivate When Done

```bash
deactivate
```

## Verify Installation

After installation completes, verify with:

```bash
source venv310/bin/activate
python -c "import TTS; print('TTS version:', getattr(TTS, '__version__', 'unknown'))"
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Pyenv Configuration (Optional)

To make pyenv available in all terminal sessions, add to `~/.zshrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
```

Then reload: `source ~/.zshrc`

## Files in Your Project

```
/home/wolf/voice2/
├── venv310/              # Python 3.10 virtual environment (NEW)
├── v/                    # Old Python 3.13 venv (can be deleted)
├── setup_venv.sh         # Setup automation script (NEW)
├── SETUP_INSTRUCTIONS.md # This file (NEW)
├── requirements.txt      # Python dependencies
└── streamlit_app.py      # Your Streamlit application
```

## Troubleshooting

### Issue: "pyenv: command not found"
**Solution**: Run the setup script which exports pyenv to PATH, or add it to your shell:
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
```

### Issue: Installation fails with network errors
**Solution**: The downloads are large. Ensure stable internet and retry:
```bash
source venv310/bin/activate
pip install --no-cache-dir -r requirements.txt
```

### Issue: GPU/CUDA errors when running TTS
**Solution**: TTS will automatically use CPU if CUDA isn't available. For CPU-only torch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Python Version Compatibility

| Package | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
|---------|-----------|-------------|-------------|-------------|-------------|
| TTS     | ✅        | ✅          | ✅          | ❌          | ❌          |
| torch   | ✅        | ✅          | ✅          | ✅          | ⚠️ Limited  |
| streamlit | ✅      | ✅          | ✅          | ✅          | ✅          |

**Chosen**: Python 3.10 (stable, well-supported by all packages)
