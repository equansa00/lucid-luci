#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="$SCRIPT_DIR/luci_code.py"

# Make entry executable
chmod +x "$ENTRY"

# Install to ~/.local/bin (no sudo needed)
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"
ln -sf "$ENTRY" "$INSTALL_DIR/luci-code"
echo "✅ luci-code installed to $INSTALL_DIR"

# Also try system-wide if sudo is available non-interactively
if sudo -n true 2>/dev/null; then
    sudo ln -sf "$ENTRY" /usr/local/bin/luci-code
    echo "✅ luci-code also installed to /usr/local/bin"
fi

# Verify
which luci-code
luci-code --help
