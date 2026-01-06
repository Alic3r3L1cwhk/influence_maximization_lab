"""
Linear Threshold (LT) diffusion model simulation.
"""

import networkx as nx
import numpy as np
from typing import List, Set, Optional, Dict
import random
from multiprocessing import Pool


class LTModel:
    """Linear Threshold diffusion model."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None):
        """
        Initialize the LT model.

        Args:
            G: Directed graph with 'threshold' on nodes and 'weight' on edges
            seed: Random seed
        """
        self.G = G
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def simulate_single(self, initial_nodes: List[int]) -> Set[int]:
        """
        Simulate a single LT diffusion process.

        Args:
            initial_nodes: List of initial seed nodes

        Returns:
            Set of all activated nodes
        """
        activated = set(initial_nodes)
        new_active = set(initial_nodes)

        # Track accumulated influence for each node
        node_influence = {node: 0.0 for node in self.G.nodes()}

        # Initialize influence from seed nodes
        for u in initial_nodes:
            for v in self.G.neighbors(u):
                if v not in activated:
                    weight = self.G[u][v].get('weight', 1.0 / max(self.G.in_degree(v), 1))
                    node_influence[v] += weight

        while new_active:
            next_active = set()

            for u in new_active:
                for v in self.G.neighbors(u):
                    if v not in activated:
                        weight = self.G[u][v].get('weight', 1.0 / max(self.G.in_degree(v), 1))
                        node_influence[v] += weight

                        threshold = self.G.nodes[v].get('threshold', 0.5)
                        if node_influence[v] >= threshold:
                            next_active.add(v)

            activated.update(next_active)
            new_active = next_active

        return activated

    def simulate_mc(self, initial_nodes: List[int], num_simulations: int = 1000) -> float:
        """
        Estimate expected influence using Monte Carlo simulation.

        Note: LT is deterministic given the thresholds, but thresholds can be randomized.

        Args:
            initial_nodes: List of initial seed nodes
            num_simulations: Number of simulations

        Returns:
            Average number of activated nodes
        """
        total_activated = 0

        for _ in range(num_simulations):
            activated = self.simulate_single(initial_nodes)
            total_activated += len(activated)

        return total_activated / num_simulations

    def simulate_batch(self, seed_sets: List[List[int]],
                      num_simulations: int = 1000) -> List[float]:
        """
        Simulate multiple seed sets in batch.

        Args:
            seed_sets: List of seed node lists
            num_simulations: Number of simulations per seed set

        Returns:
            List of expected influences
        """
        influences = []
        for seeds in seed_sets:
            influence = self.simulate_mc(seeds, num_simulations)
            influences.append(influence)

        return influences


def _simulate_single_lt(args):
    """Helper function for parallel simulation."""
    G, initial_nodes, seed_offset = args

    if seed_offset is not None:
        random.seed(seed_offset)
        np.random.seed(seed_offset)

    activated = set(initial_nodes)
    new_active = set(initial_nodes)

    # Track accumulated influence
    node_influence = {node: 0.0 for node in G.nodes()}

    for u in initial_nodes:
        for v in G.neighbors(u):
            if v not in activated:
                weight = G[u][v].get('weight', 1.0 / max(G.in_degree(v), 1))
                node_influence[v] += weight

    while new_active:
        next_active = set()

        for u in new_active:
            for v in G.neighbors(u):
                if v not in activated:
                    weight = G[u][v].get('weight', 1.0 / max(G.in_degree(v), 1))
                    node_influence[v] += weight

                    threshold = G.nodes[v].get('threshold', 0.5)
                    if node_influence[v] >= threshold:
                        next_active.add(v)

        activated.update(next_active)
        new_active = next_active

    return len(activated)


class LTModelParallel:
    """LT model with parallel Monte Carlo simulation."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None, num_workers: int = 4):
        """
        Initialize the parallel LT model.

        Args:
            G: Directed graph
            seed: Random seed
            num_workers: Number of parallel workers
        """
        self.G = G
        self.seed = seed
        self.num_workers = num_workers

    def simulate_mc_parallel(self, initial_nodes: List[int],
                           num_simulations: int = 1000) -> float:
        """
        Estimate expected influence using parallel Monte Carlo simulation.

        Args:
            initial_nodes: List of initial seed nodes
            num_simulations: Number of simulations

        Returns:
            Average number of activated nodes
        """
        # Create arguments for each simulation
        if self.seed is not None:
            seeds = [self.seed + i for i in range(num_simulations)]
        else:
            seeds = [None] * num_simulations

        args_list = [(self.G, initial_nodes, s) for s in seeds]

        # Parallel simulation
        with Pool(self.num_workers) as pool:
            results = pool.map(_simulate_single_lt, args_list)

        return sum(results) / len(results)

    def simulate_batch_parallel(self, seed_sets: List[List[int]],
                               num_simulations: int = 1000) -> List[float]:
        """
        Simulate multiple seed sets in parallel.

        Args:
            seed_sets: List of seed node lists
            num_simulations: Number of simulations per seed set

        Returns:
            List of expected influences
        """
        influences = []
        for seeds in seed_sets:
            influence = self.simulate_mc_parallel(seeds, num_simulations)
            influences.append(influence)

        return influences
