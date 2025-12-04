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
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate Venv
source .venv/bin/activate

# 4. Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# 5. Install Dependencies
echo "📥 Installing dependencies..."
# Install core package with analysis and agent extras
pip install -e ".[analysis,agent]"

# Install local backend dependencies (PyTorch + Transformers)
# Note: GAUL likely has CUDA 11.x or 12.x. This installs a compatible version.
echo "📥 Installing PyTorch and Transformers for GPU support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes

# 6. Verify Install
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import bailiff; print(f'Bailiff: {bailiff.__file__}')"

echo "✅ Setup Complete! You can now run experiments."
echo "   To activate: source .venv/bin/activate"
