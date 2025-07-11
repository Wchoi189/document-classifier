#!/bin/bash

# setup-dev-user.sh - Create dev group and setup user development privileges
# Usage: ./setup-dev-user.sh [username] [project_path] [options]
# Download and make executable
# curl -o setup-dev-user.sh [URL_TO_SCRIPT]
# chmod +x setup-dev-user.sh

# # Or create locally
# nano setup-dev-user.sh
# # (paste the script content)
# chmod +x setup-dev-user.sh

# # Use it
# sudo ./setup-dev-user.sh wb2x /root/document-classifier



set -e  # Exit on any error

# Default values
DEFAULT_GROUP="dev"
PASSWORDLESS_SUDO=false
MOVE_TO_HOME=false
ADD_DOCKER=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] USERNAME [PROJECT_PATH]

Setup development environment for a user with proper group permissions.

ARGUMENTS:
    USERNAME        Username to setup (required)
    PROJECT_PATH    Path to project directory (optional)

OPTIONS:
    -g, --group GROUP       Group name (default: dev)
    -p, --passwordless      Enable passwordless sudo for the group
    -m, --move-to-home      Move project to user's home directory
    -d, --docker            Add user to docker group
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0 wb2x
    $0 wb2x /root/document-classifier
    $0 -p -d wb2x /root/document-classifier
    $0 --move-to-home --docker wb2x /opt/myproject

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--group)
            DEFAULT_GROUP="$2"
            shift 2
            ;;
        -p|--passwordless)
            PASSWORDLESS_SUDO=true
            shift
            ;;
        -m|--move-to-home)
            MOVE_TO_HOME=true
            shift
            ;;
        -d|--docker)
            ADD_DOCKER=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$USERNAME" ]]; then
                USERNAME="$1"
            elif [[ -z "$PROJECT_PATH" ]]; then
                PROJECT_PATH="$1"
            else
                print_error "Too many arguments"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if username is provided
if [[ -z "$USERNAME" ]]; then
    print_error "Username is required"
    show_help
    exit 1
fi

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    print_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if user exists
if ! id "$USERNAME" &>/dev/null; then
    print_error "User '$USERNAME' does not exist"
    exit 1
fi

print_header "Development Environment Setup for $USERNAME"

# Create dev group
print_status "Creating group '$DEFAULT_GROUP'..."
if getent group "$DEFAULT_GROUP" > /dev/null 2>&1; then
    print_warning "Group '$DEFAULT_GROUP' already exists"
else
    groupadd "$DEFAULT_GROUP"
    print_status "Group '$DEFAULT_GROUP' created successfully"
fi

# Add user to dev group
print_status "Adding user '$USERNAME' to group '$DEFAULT_GROUP'..."
usermod -a -G "$DEFAULT_GROUP" "$USERNAME"

# Add user to sudo group
print_status "Adding user '$USERNAME' to sudo group..."
usermod -a -G sudo "$USERNAME"

# Add user to other useful groups
print_status "Adding user to development groups (adm, dialout, plugdev)..."
usermod -a -G adm,dialout,plugdev "$USERNAME"

# Add to docker group if requested
if [[ "$ADD_DOCKER" == true ]]; then
    print_status "Adding user '$USERNAME' to docker group..."
    if getent group docker > /dev/null 2>&1; then
        usermod -a -G docker "$USERNAME"
    else
        print_warning "Docker group does not exist, skipping..."
    fi
fi

# Setup sudo privileges for dev group
print_status "Setting up sudo privileges for '$DEFAULT_GROUP' group..."
SUDOERS_FILE="/etc/sudoers.d/$DEFAULT_GROUP"

if [[ "$PASSWORDLESS_SUDO" == true ]]; then
    echo "%$DEFAULT_GROUP ALL=(ALL:ALL) NOPASSWD:ALL" > "$SUDOERS_FILE"
    print_status "Passwordless sudo enabled for '$DEFAULT_GROUP' group"
else
    echo "%$DEFAULT_GROUP ALL=(ALL:ALL) ALL" > "$SUDOERS_FILE"
    print_status "Sudo access enabled for '$DEFAULT_GROUP' group"
fi

# Set proper permissions on sudoers file
chmod 440 "$SUDOERS_FILE"

# Handle project directory if provided
if [[ -n "$PROJECT_PATH" ]]; then
    if [[ -d "$PROJECT_PATH" ]]; then
        print_header "Setting up project directory: $PROJECT_PATH"
        
        if [[ "$MOVE_TO_HOME" == true ]]; then
            USER_HOME=$(eval echo "~$USERNAME")
            PROJECT_NAME=$(basename "$PROJECT_PATH")
            NEW_PATH="$USER_HOME/$PROJECT_NAME"
            
            print_status "Moving project to user home directory..."
            if [[ -d "$NEW_PATH" ]]; then
                print_warning "Directory '$NEW_PATH' already exists, skipping move..."
            else
                mv "$PROJECT_PATH" "$NEW_PATH"
                PROJECT_PATH="$NEW_PATH"
                print_status "Project moved to: $PROJECT_PATH"
            fi
        fi
        
        # Set ownership and permissions
        print_status "Setting ownership and permissions..."
        chown -R "$USERNAME:$DEFAULT_GROUP" "$PROJECT_PATH"
        chmod -R g+rwx "$PROJECT_PATH"
        chmod g+s "$PROJECT_PATH"  # Set group sticky bit
        
        print_status "Project directory setup complete"
    else
        print_warning "Project path '$PROJECT_PATH' does not exist, skipping..."
    fi
fi

# Create a simple verification script
VERIFY_SCRIPT="/tmp/verify-dev-setup-$USERNAME.sh"
cat > "$VERIFY_SCRIPT" << EOF
#!/bin/bash
echo "=== Development Setup Verification for $USERNAME ==="
echo "User groups: \$(groups $USERNAME)"
echo "User ID info: \$(id $USERNAME)"
echo ""
echo "Sudo access test:"
sudo -l -U $USERNAME 2>/dev/null || echo "No sudo access configured"
echo ""
if [[ -n "$PROJECT_PATH" && -d "$PROJECT_PATH" ]]; then
    echo "Project directory permissions:"
    ls -la "$PROJECT_PATH"
fi
echo ""
echo "=== Setup Complete ==="
echo "Note: User must log out and back in for group changes to take effect"
EOF

chmod +x "$VERIFY_SCRIPT"

print_header "Setup Summary"
echo "✓ User '$USERNAME' added to group '$DEFAULT_GROUP'"
echo "✓ User added to sudo group"
echo "✓ Development groups configured"
if [[ "$ADD_DOCKER" == true ]]; then
    echo "✓ Docker group access configured"
fi
if [[ "$PASSWORDLESS_SUDO" == true ]]; then
    echo "✓ Passwordless sudo enabled"
else
    echo "✓ Sudo access enabled (password required)"
fi
if [[ -n "$PROJECT_PATH" && -d "$PROJECT_PATH" ]]; then
    echo "✓ Project directory configured: $PROJECT_PATH"
fi

print_header "Next Steps"
echo "1. User '$USERNAME' should log out and back in for group changes to take effect"
echo "2. Run verification script: $VERIFY_SCRIPT"
if [[ -n "$PROJECT_PATH" ]]; then
    echo "3. Navigate to project: cd $PROJECT_PATH"
fi
echo "4. Test sudo access: sudo -v"

print_status "Development environment setup completed successfully!"

# Run verification if verbose mode
if [[ "$VERBOSE" == true ]]; then
    print_header "Verification Results"
    bash "$VERIFY_SCRIPT"
fi
