#!/bin/bash
# Script to deploy code to GAUL
# Usage: ./deploy_to_gaul.sh <username>
# Example: ./deploy_to_gaul.sh jdoe

if [ -z "$1" ]; then
    echo "Usage: $0 <gaul_username>"
    exit 1
fi

USER=$1
HOST="compute.gaul.csd.uwo.ca"
REMOTE_DIR="~/ai-law-agents"

echo "🚀 Deploying to $USER@$HOST..."

# Create remote directory if it doesn't exist
ssh "$USER@$HOST" "mkdir -p $REMOTE_DIR"

# Sync files (using rsync if available, else scp)
# We exclude .git, .venv, and __pycache__ to save time/bandwidth
echo "📦 Transferring files..."

# Using rsync is preferred for speed and exclusion
if command -v rsync &> /dev/null; then
    rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
        ./ "$USER@$HOST:$REMOTE_DIR/"
else
    # Fallback to SCP (less efficient but universal)
    echo "⚠️ rsync not found, using scp (might be slower)..."
    scp -r bailiff configs scripts setup.py pyproject.toml README.md "$USER@$HOST:$REMOTE_DIR/"
fi

echo "✅ Deployment Complete!"
echo "   Next steps:"
echo "   1. ssh $USER@$HOST"
echo "   2. cd $REMOTE_DIR"
echo "   3. ./scripts/setup_gaul.sh"
