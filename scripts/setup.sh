#!/bin/bash
# Quick setup script for students

set -e

echo "=================================================="
echo "Distributed GAN Training - Quick Setup"
echo "=================================================="

# Check if config exists
if [ ! -f "config/config.yaml" ]; then
    echo ""
    echo "Step 1: Creating config file..."
    cp config/config.yaml.template config/config.yaml
    echo "✓ Config file created at config/config.yaml"
    echo ""
    echo "⚠️  IMPORTANT: Edit config/config.yaml with database credentials"
    echo "   provided by your instructor before continuing."
    echo ""
    read -p "Press Enter when you've updated the config file..."
fi

# Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check for dataset
if [ ! -d "data/celeba" ] && [ ! -d "data/celeba_torchvision" ]; then
    echo ""
    echo "Step 3: Dataset not found."
    echo "Options:"
    echo "  1. Download automatically using torchvision (recommended)"
    echo "  2. Download manually from Kaggle"
    echo "  3. Skip for now"
    echo ""
    read -p "Enter choice (1/2/3): " choice
    
    case $choice in
        1)
            python scripts/download_celeba.py
            ;;
        2)
            echo ""
            echo "Manual download instructions:"
            echo "1. Go to: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
            echo "2. Download and extract to ./data/celeba/"
            ;;
        3)
            echo "Skipping dataset download. Download it before starting the worker."
            ;;
    esac
else
    echo ""
    echo "Step 3: ✓ Dataset found"
fi

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To start your worker:"
echo "  cd src"
echo "  python worker.py"
echo ""
echo "Your GPU will then be part of the distributed training cluster!"
echo ""
