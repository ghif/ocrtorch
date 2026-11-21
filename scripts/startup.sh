#!/bin/bash
set -e

DISK_DEV="/dev/disk/by-id/google-data-disk"
MOUNT_DIR="/mnt/data"
TARGET_USER="mghifary"
USER_HOME="/home/$TARGET_USER"

# ==========================================
# 1. FORMAT DISK (First Run Only)
# ==========================================
# Format the disk only if it does not already contain a filesystem
if [ -b "$DISK_DEV" ] && [ -z "$(blkid "$DISK_DEV")" ]; then
    mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$DISK_DEV"
fi

# ==========================================
# 2. MOUNT PERSISTENT DISK
# ==========================================
mkdir -p "$MOUNT_DIR"
if ! mountpoint -q "$MOUNT_DIR"; then
    mount -o discard,defaults "$DISK_DEV" "$MOUNT_DIR"
fi
chmod a+w "$MOUNT_DIR"

# ==========================================
# 3. FSTAB REGISTRATION
# ==========================================
# Add to /etc/fstab for clean reboots (with nofail so the VM still boots if detached)
if ! grep -q "$DISK_DEV" /etc/fstab; then
    echo "$DISK_DEV $MOUNT_DIR ext4 discard,defaults,nofail 0 2" >> /etc/fstab
fi

# ==========================================
# 4. SYSTEM DEPENDENCIES
# ==========================================
# Neovim AppImage requires libfuse2 on modern Ubuntu/Debian instances
if ! dpkg -s libfuse2 &>/dev/null; then
    apt-get update -y && apt-get install -y libfuse2
fi

# ==========================================
# 5. GLOBAL BINARIES (/usr/local/bin)
# ==========================================

# Node.js runtime & package managers (required by Pi)
ln -sf "$MOUNT_DIR/apps/nodejs/bin/node" /usr/local/bin/node
ln -sf "$MOUNT_DIR/apps/nodejs/bin/npm"  /usr/local/bin/npm
ln -sf "$MOUNT_DIR/apps/nodejs/bin/npx"  /usr/local/bin/npx

# Neovim
chmod +x "$MOUNT_DIR/apps/nvim/nvim-linux-x86_64.appimage" 2>/dev/null || true
ln -sf "$MOUNT_DIR/apps/nvim/nvim-linux-x86_64.appimage" /usr/local/bin/nvim

# Herdr
chmod +x "$MOUNT_DIR/apps/herdr/bin/herdr" 2>/dev/null || true
ln -sf "$MOUNT_DIR/apps/herdr/bin/herdr" /usr/local/bin/herdr

# Pi Agent
chmod +x "$MOUNT_DIR/apps/pi/agent/bin/pi" 2>/dev/null || true
ln -sf "$MOUNT_DIR/apps/pi/agent/bin/pi" /usr/local/bin/pi

# GitHub CLI (gh)
chmod +x "$MOUNT_DIR/apps/gh/bin/gh" 2>/dev/null || true
ln -sf "$MOUNT_DIR/apps/gh/bin/gh" /usr/local/bin/gh

# ==========================================
# 6. USER STATE, DOTFILES & CONFIGS
# ==========================================
if [ -d "$USER_HOME" ]; then
    # Base user directories
    mkdir -p "$USER_HOME/.config" "$USER_HOME/.local/share" "$USER_HOME/.local/state"

    # Pi session & workspace state
    mkdir -p "$MOUNT_DIR/apps/pi"
    ln -sfn "$MOUNT_DIR/apps/pi" "$USER_HOME/.pi"
    chown -h "$TARGET_USER:$TARGET_USER" "$USER_HOME/.pi"

    # GitHub CLI config & credentials
    mkdir -p "$MOUNT_DIR/dotfiles/config/gh"
    ln -sfn "$MOUNT_DIR/dotfiles/config/gh" "$USER_HOME/.config/gh"

    # Neovim: Configuration, Installed Plugins/Data, and State/History
    mkdir -p "$MOUNT_DIR/dotfiles/nvim/config"
    mkdir -p "$MOUNT_DIR/dotfiles/nvim/data"
    mkdir -p "$MOUNT_DIR/dotfiles/nvim/state"

    ln -sfn "$MOUNT_DIR/dotfiles/nvim/config" "$USER_HOME/.config/nvim"
    ln -sfn "$MOUNT_DIR/dotfiles/nvim/data"   "$USER_HOME/.local/share/nvim"
    ln -sfn "$MOUNT_DIR/dotfiles/nvim/state"  "$USER_HOME/.local/state/nvim"

    # Ensure ownership belongs to user across config and local directories
    chown -R "$TARGET_USER:$TARGET_USER" "$USER_HOME/.config" "$USER_HOME/.local"
fi

# (Optional) TPU libraries if needed:
# pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html