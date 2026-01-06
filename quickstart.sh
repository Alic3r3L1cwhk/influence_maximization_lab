#!/bin/bash

# Quick Start Script for Influence Maximization Lab
# This script runs a small demo experiment

echo "======================================================================"
echo "           Influence Maximization Lab - Quick Start Demo             "
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "[Step 1/3] Training parameters on a small network (200 nodes)..."
echo "This will take about 2-3 minutes..."
echo ""

python experiments/train_params.py \
    --network-type ba \
    --num-nodes 200 \
    --ba-m 3 \
    --num-cascades 500 \
    --embedding-dim 64 \
    --epochs 30 \
    --device cpu \
    --seed 42 \
    --output-dir outputs/quickstart

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "[Step 2/3] Running influence maximization with learned parameters..."
echo "This will take about 1-2 minutes..."
echo ""

python experiments/run_influence_max.py \
    --network-path outputs/quickstart/network.edgelist \
    --model-path outputs/quickstart/param_learner.pth \
    --embeddings-path outputs/quickstart/embeddings.txt \
    --algorithm lazy_greedy \
    --k 5 \
    --num-simulations 500 \
    --compare-params \
    --num-runs 3 \
    --device cpu \
    --seed 42 \
    --output-dir outputs/quickstart/im_results

if [ $? -ne 0 ]; then
    echo "Error: Influence maximization failed"
    exit 1
fi

echo ""
echo "[Step 3/3] Running quick comparison of algorithms..."
echo "This will take about 2-3 minutes..."
echo ""

python experiments/compare_methods.py \
    --num-nodes 200 \
    --k-values 5 10 \
    --algorithms lazy_greedy tim_plus \
    --train-params \
    --num-runs 2 \
    --num-cascades 300 \
    --epochs 20 \
    --device cpu \
    --seed 42 \
    --output-dir outputs/quickstart/comparison

if [ $? -ne 0 ]; then
    echo "Error: Comparison experiment failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "                    Quick Start Demo Complete!                        "
echo "======================================================================"
echo ""
echo "Results are saved in outputs/quickstart/"
echo ""
echo "You can view:"
echo "  - Training curves: outputs/quickstart/training_history.png"
echo "  - IM results: outputs/quickstart/im_results/im_results.json"
echo "  - Comparison: outputs/quickstart/comparison/comparison_results.csv"
echo ""
echo "For larger experiments, check the README.md for detailed instructions."
echo ""
