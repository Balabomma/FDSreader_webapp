#!/usr/bin/env bash
# FDSReader WebApp — One-line installer for Linux / macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/Balabomma/FDSreader_webapp/master/install.sh | bash
set -e

REPO="https://github.com/Balabomma/FDSreader_webapp.git"
INSTALL_DIR="FDSreader_webapp"

echo "========================================="
echo "  FDSReader WebApp — Installer"
echo "========================================="

# Check for Python 3
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: Python 3 is required but not found."
    echo "Install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

PY_VER=$($PY --version 2>&1)
echo "Found: $PY_VER"

# Check for git
if ! command -v git &>/dev/null; then
    echo "ERROR: git is required but not found."
    exit 1
fi

# Clone or update
if [ -d "$INSTALL_DIR" ]; then
    echo "Directory '$INSTALL_DIR' already exists. Pulling latest..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Create virtual environment
echo "Creating virtual environment..."
$PY -m venv venv

# Activate and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "========================================="
echo "  Installation complete!"
echo "========================================="
echo ""
echo "To run the app:"
echo "  cd $INSTALL_DIR"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open http://localhost:5000 in your browser."
