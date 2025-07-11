#!/bin/bash

echo "=== ML Development Environment Setup with User Creation ==="

# Configuration
USERNAME=${USERNAME:-wb2x}
USER_UID=${USER_UID:-1000}
USER_GID=${USER_GID:-1000}

# Step 1: System setup (as root)
echo "1. Setting up system locale and packages..."
export LC_ALL=C.UTF-8 LANG=C.UTF-8
apt update
apt install -y locales language-pack-ko language-pack-ko-base fonts-nanum fonts-nanum-coding fonts-nanum-extra sudo
locale-gen en_US.UTF-8 ko_KR.UTF-8
update-locale

# Step 2: Update conda (as root)
echo "2. Updating conda..."
conda update -y conda

# Step 3: Create user
echo "3. Creating user '$USERNAME'..."
groupadd -g $USER_GID $USERNAME 2>/dev/null || true
useradd -u $USER_UID -g $USER_GID -m -s /bin/bash $USERNAME 2>/dev/null || true
usermod -aG sudo $USERNAME

# Step 4: Configure user environment
echo "4. Configuring user environment..."
cat > /home/$USERNAME/.bashrc << 'EOF'
# Conda initialization
export PATH="/opt/conda/bin:$PATH"

# Locale configuration
export LANG=en_US.UTF-8
export LANGUAGE=en_US:ko_KR

# ML development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias jlab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
EOF

# Step 5: Set up ML directories
mkdir -p /home/$USERNAME/{projects,data,notebooks,models}
chown -R $USERNAME:$USERNAME /home/$USERNAME

# Step 6: Configure conda for user
# Make conda accessible to user (but keep root ownership for security)
chmod -R 755 /opt/conda

echo "=== Setup Complete ==="
echo ""
echo "✅ Switch to ML user with:"
echo "   su - $USERNAME"
echo ""
echo "✅ Or start a new shell as user:"
echo "   docker exec -it --user $USERNAME <container_name> bash"