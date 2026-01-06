"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import os


sns.set_style("whitegrid")


def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC plot
    axes[1].plot(history['train_auc'], label='Train AUC', linewidth=2)
    if 'val_auc' in history and len(history['val_auc']) > 0:
        axes[1].plot(history['val_auc'], label='Val AUC', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('Training and Validation AUC', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_influence_comparison(results: Dict[str, List[float]],
                             save_path: Optional[str] = None):
    """
    Plot influence comparison between methods.

    Args:
        results: Dictionary mapping method names to influence values
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results.keys())
    means = [np.mean(results[m]) for m in methods]
    stds = [np.std(results[m]) for m in methods]

    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  alpha=0.7, edgecolor='black', linewidth=1.5)

    # Color bars
    colors = sns.color_palette("husl", len(methods))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Expected Influence', fontsize=12)
    ax.set_title('Influence Comparison Across Methods', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Influence comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_marginal_gains(marginal_gains: List[float],
                       save_path: Optional[str] = None):
    """
    Plot marginal gains during greedy selection.

    Args:
        marginal_gains: List of marginal gains
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k = len(marginal_gains)
    ax.plot(range(1, k + 1), marginal_gains, marker='o',
           linewidth=2, markersize=6)

    ax.set_xlabel('Iteration (Seed Number)', fontsize=12)
    ax.set_ylabel('Marginal Gain', fontsize=12)
    ax.set_title('Marginal Gains During Greedy Selection', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Marginal gains plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_runtime_comparison(runtimes: Dict[str, float],
                           save_path: Optional[str] = None):
    """
    Plot runtime comparison between methods.

    Args:
        runtimes: Dictionary mapping method names to runtime in seconds
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(runtimes.keys())
    times = [runtimes[m] for m in methods]

    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, times, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Color bars
    colors = sns.color_palette("husl", len(methods))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison Across Methods', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time:.2f}s',
               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Runtime comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_experiments(all_results: List[Dict[str, List[float]]],
                             labels: List[str],
                             save_path: Optional[str] = None):
    """
    Plot results from multiple experiments.

    Args:
        all_results: List of result dictionaries
        labels: List of experiment labels
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get all methods from first result
    methods = list(all_results[0].keys())
    x = np.arange(len(methods))
    width = 0.8 / len(all_results)

    for i, (results, label) in enumerate(zip(all_results, labels)):
        means = [np.mean(results[m]) for m in methods]
        stds = [np.std(results[m]) for m in methods]

        offset = (i - len(all_results)/2) * width + width/2
        ax.bar(x + offset, means, width, label=label,
              yerr=stds, capsize=3, alpha=0.7)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Expected Influence', fontsize=12)
    ax.set_title('Comparison Across Multiple Experiments', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-experiment plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
