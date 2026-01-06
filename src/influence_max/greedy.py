"""
Greedy algorithm for influence maximization.
Based on marginal gain estimation using Monte Carlo simulation.
"""

import networkx as nx
import numpy as np
from typing import List, Set, Optional
from tqdm import tqdm
import time


class GreedyIM:
    """Greedy algorithm for influence maximization."""

    def __init__(self, G: nx.DiGraph, simulator,
                 seed: Optional[int] = None):
        """
        Initialize the greedy algorithm.

        Args:
            G: Directed graph
            simulator: DiffusionSimulator instance
            seed: Random seed
        """
        self.G = G
        self.simulator = simulator
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def select_seeds(self, k: int, num_simulations: int = 1000,
                    verbose: bool = True) -> tuple:
        """
        Select k seed nodes using greedy algorithm.

        Args:
            k: Number of seeds to select
            num_simulations: Number of MC simulations for influence estimation
            verbose: Whether to show progress

        Returns:
            Tuple of (selected seeds, marginal gains, total time)
        """
        selected_seeds = []
        marginal_gains = []
        start_time = time.time()

        nodes = list(self.G.nodes())
        iterator = tqdm(range(k), desc="Selecting seeds") if verbose else range(k)

        for iteration in iterator:
            best_node = None
            best_gain = -1

            # Current influence
            if len(selected_seeds) > 0:
                current_influence = self.simulator.estimate_influence(
                    selected_seeds, num_simulations)
            else:
                current_influence = 0

            # Try each candidate node
            candidates = [n for n in nodes if n not in selected_seeds]

            for node in candidates:
                # Marginal gain of adding this node
                new_seeds = selected_seeds + [node]
                new_influence = self.simulator.estimate_influence(
                    new_seeds, num_simulations)
                gain = new_influence - current_influence

                if gain > best_gain:
                    best_gain = gain
                    best_node = node

            if best_node is not None:
                selected_seeds.append(best_node)
                marginal_gains.append(best_gain)

                if verbose:
                    iterator.set_postfix({
                        'node': best_node,
                        'gain': f'{best_gain:.2f}'
                    })

        total_time = time.time() - start_time

        return selected_seeds, marginal_gains, total_time


class LazyGreedyIM:
    """
    Lazy-forward greedy algorithm (CELF optimization).
    Significantly faster than vanilla greedy.
    """

    def __init__(self, G: nx.DiGraph, simulator,
                 seed: Optional[int] = None):
        """
        Initialize the lazy greedy algorithm.

        Args:
            G: Directed graph
            simulator: DiffusionSimulator instance
            seed: Random seed
        """
        self.G = G
        self.simulator = simulator
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def select_seeds(self, k: int, num_simulations: int = 1000,
                    verbose: bool = True) -> tuple:
        """
        Select k seed nodes using lazy greedy (CELF) algorithm.

        Args:
            k: Number of seeds to select
            num_simulations: Number of MC simulations
            verbose: Whether to show progress

        Returns:
            Tuple of (selected seeds, marginal gains, total time)
        """
        selected_seeds = []
        marginal_gains = []
        start_time = time.time()

        nodes = list(self.G.nodes())

        # Initialize marginal gains heap (node, gain, iteration_last_updated)
        gain_heap = []
        current_influence = 0

        # First iteration: compute all marginal gains
        if verbose:
            print("Computing initial marginal gains...")

        for node in tqdm(nodes, disable=not verbose):
            influence = self.simulator.estimate_influence([node], num_simulations)
            gain_heap.append([node, influence, 0])

        # Sort by gain (descending)
        gain_heap.sort(key=lambda x: x[1], reverse=True)

        iterator = tqdm(range(k), desc="Selecting seeds") if verbose else range(k)

        for iteration in iterator:
            # Find node with true maximum marginal gain
            while True:
                # Get top node
                current = gain_heap[0]
                node, gain, last_update = current

                # If this node was updated in the previous iteration, it's the true max
                if last_update == iteration:
                    break

                # Recompute marginal gain
                new_seeds = selected_seeds + [node]
                new_influence = self.simulator.estimate_influence(
                    new_seeds, num_simulations)
                new_gain = new_influence - current_influence

                # Update
                current[1] = new_gain
                current[2] = iteration

                # Re-sort
                gain_heap.sort(key=lambda x: x[1], reverse=True)

            # Select the best node
            best_node, best_gain, _ = gain_heap.pop(0)
            selected_seeds.append(best_node)
            marginal_gains.append(best_gain)

            # Update current influence
            current_influence += best_gain

            if verbose:
                iterator.set_postfix({
                    'node': best_node,
                    'gain': f'{best_gain:.2f}'
                })

        total_time = time.time() - start_time

        return selected_seeds, marginal_gains, total_time
