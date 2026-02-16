#!/bin/bash
# Comparison helper script - run a quick test of both training methods

echo "================================================================"
echo "DCGAN Training Methods Comparison"
echo "================================================================"
echo ""
echo "This script helps you compare distributed vs local training."
echo ""
echo "Choose an option:"
echo "  1. Test local training (1 epoch, quick)"
echo "  2. Test distributed worker (requires DB setup)"
echo "  3. Run both for comparison"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "Starting local training test (1 epoch)..."
        echo "This will train on your GPU without database."
        echo ""
        cd src
        python train_local.py --epochs 1 --batch-size 64 --sample-interval 1
        echo ""
        echo "Check outputs_local/samples/ for generated images!"
        ;;
    2)
        echo ""
        echo "Starting distributed worker test..."
        echo "Make sure main.py is running first!"
        echo ""
        cd src
        python worker.py
        ;;
    3)
        echo ""
        echo "NOTE: For distributed, you need to start main.py separately."
        echo "This will only run local training as a demonstration."
        echo ""
        echo "Starting local training (5 epochs)..."
        cd src
        python train_local.py --epochs 5 --batch-size 64 --sample-interval 1
        echo ""
        echo "Local training complete!"
        echo "Results in: outputs_local/samples/"
        echo ""
        echo "To compare with distributed:"
        echo "1. Setup database connection in config/config.yaml"
        echo "2. Run: python worker.py"
        echo "3. Compare outputs/ vs outputs_local/"
        ;;
    *)
        echo "Invalid choice. Exiting."
        ;;
esac
