#!/bin/bash
# Setup script for GAUL environment
# Usage: ./setup_gaul.sh

set -e  # Exit on error

echo "🚀 Starting GAUL Environment Setup..."

# 1. Load Python module (if required on GAUL, otherwise assumes python3 is available)
# module load python/3.10  # Uncomment if GAUL uses modules

# 2. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    if python3 -m venv .venv; then
        echo "✅ Venv created successfully."
    else
        echo "⚠️ Standard venv creation failed (likely missing ensurepip). Retrying without pip..."
        # Fallback for systems with broken python3-venv (common on Debian/Ubuntu)
        python3 -m venv .venv --without-pip
        
        # Manually install pip
        source .venv/bin/activate
        echo "⬇️ Downloading get-pip.py..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py
        rm get-pip.py
        deactivate
    fi
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate Venv
source .venv/bin/activate

# 4. Load optional Groq API keys from .env
if [ -f ".env" ]; then
    echo "Loading Groq API keys from .env..."
    set -a
    source .env
    set +a
else
    echo "No .env found. To use Groq multi-key, create .env with:"
    echo "  GROQ_API_KEYS='[\"key1\",\"key2\",\"key3\"]'"
    echo "Optional per-key limits:"
    echo "  GROQ_API_KEY_CONCURRENCY='{\"key1\":2,\"key2\":2}'"
fi

# 5. Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# 6. Install Dependencies
echo "📥 Installing dependencies..."
# Install core package with analysis and agent extras
pip install -e ".[analysis,agent]"

# Install local backend dependencies (PyTorch + Transformers)
# Note: GAUL likely has CUDA 11.x or 12.x. This installs a compatible version.
echo "📥 Installing PyTorch and Transformers for GPU support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes

# 7. Verify Install
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import bailiff; print(f'Bailiff: {bailiff.__file__}')"
if [ -n "$GROQ_API_KEYS" ] || [ -n "$GROQ_API_KEY" ]; then
    echo "Checking GroqKeyPool configuration..."
    python - <<'PY'
from bailiff.agents.groq_pool import GroqKeyPool

pool = GroqKeyPool.from_env()
print("GroqKeyPool initialized with", len(pool.summary()), "keys")
print(pool.summary())
PY
fi

echo "✅ Setup Complete! You can now run experiments."
echo "   To activate: source .venv/bin/activate"
