"""
TIM and TIM+ algorithms for influence maximization.
Based on Two-phase Influence Maximization (Tang et al., 2014).
Uses reverse reachable (RR) sets for efficient seed selection.
"""

import networkx as nx
import numpy as np
from typing import List, Set, Optional
import random
import math
import time
from tqdm import tqdm
from collections import defaultdict


class TIM:
    """Two-phase Influence Maximization (TIM) algorithm."""

    def __init__(self, G: nx.DiGraph, model: str = 'ic',
                 seed: Optional[int] = None):
        """
        Initialize TIM algorithm.

        Args:
            G: Directed graph
            model: Diffusion model ('ic' or 'lt')
            seed: Random seed
        """
        self.G = G
        self.model = model
        self.seed = seed
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_rr_set_ic(self) -> Set[int]:
        """
        Generate a random reverse reachable (RR) set for IC model.

        Returns:
            Set of nodes in the RR set
        """
        # Select random target node
        v = random.choice(list(self.G.nodes()))
        rr_set = {v}

        # Reverse BFS with probabilistic edge activation
        queue = [v]
        visited = {v}

        while queue:
            current = queue.pop(0)

            # Check all in-neighbors
            for u in self.G.predecessors(current):
                if u not in visited:
                    # Edge is activated with probability
                    prob = self.G[u][current].get('prob', 0.1)
                    if random.random() < prob:
                        rr_set.add(u)
                        queue.append(u)
                        visited.add(u)

        return rr_set

    def generate_rr_set_lt(self) -> Set[int]:
        """
        Generate a random RR set for LT model.

        Returns:
            Set of nodes in the RR set
        """
        # Select random target node
        v = random.choice(list(self.G.nodes()))
        rr_set = {v}

        # For LT, we need to select one in-neighbor based on edge weights
        queue = [v]
        visited = {v}

        while queue:
            current = queue.pop(0)

            in_neighbors = list(self.G.predecessors(current))
            if len(in_neighbors) == 0:
                continue

            # Get weights
            weights = []
            for u in in_neighbors:
                weight = self.G[u][current].get('weight',
                                               1.0 / max(self.G.in_degree(current), 1))
                weights.append(weight)

            # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()

                # Select one in-neighbor
                selected = np.random.choice(in_neighbors, p=weights)

                if selected not in visited:
                    rr_set.add(selected)
                    queue.append(selected)
                    visited.add(selected)

        return rr_set

    def generate_rr_sets(self, theta: int) -> List[Set[int]]:
        """
        Generate theta RR sets.

        Args:
            theta: Number of RR sets to generate

        Returns:
            List of RR sets
        """
        rr_sets = []

        for _ in range(theta):
            if self.model == 'ic':
                rr_set = self.generate_rr_set_ic()
            elif self.model == 'lt':
                rr_set = self.generate_rr_set_lt()
            else:
                raise ValueError(f"Unknown model: {self.model}")

            rr_sets.append(rr_set)

        return rr_sets

    def node_selection(self, rr_sets: List[Set[int]], k: int) -> List[int]:
        """
        Select k nodes that cover the most RR sets.

        Args:
            rr_sets: List of RR sets
            k: Number of seeds to select

        Returns:
            List of selected seed nodes
        """
        selected_seeds = []
        uncovered_sets = list(range(len(rr_sets)))

        # Build node to RR sets mapping
        node_to_rr = defaultdict(set)
        for idx, rr_set in enumerate(rr_sets):
            for node in rr_set:
                node_to_rr[node].add(idx)

        for _ in range(k):
            if len(uncovered_sets) == 0:
                break

            # Find node that covers most uncovered RR sets
            best_node = None
            best_coverage = 0

            for node in self.G.nodes():
                coverage = len(node_to_rr[node].intersection(uncovered_sets))
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_node = node

            if best_node is None:
                break

            selected_seeds.append(best_node)

            # Remove covered sets
            covered = node_to_rr[best_node].intersection(uncovered_sets)
            for idx in covered:
                uncovered_sets.remove(idx)

            # Update node_to_rr
            for node in node_to_rr:
                node_to_rr[node] -= covered

        return selected_seeds

    def select_seeds(self, k: int, epsilon: float = 0.2,
                    verbose: bool = True) -> tuple:
        """
        Select k seeds using TIM algorithm.

        Args:
            k: Number of seeds
            epsilon: Error tolerance (smaller = more RR sets = slower but more accurate)
            verbose: Whether to show progress

        Returns:
            Tuple of (selected seeds, estimated influence, total time)
        """
        start_time = time.time()

        # Calculate theta (number of RR sets needed)
        # Based on theoretical analysis in the paper
        lambda_param = (8 + 2 * epsilon) * self.n * (
            math.log(self.n) + math.log(2) + math.log(epsilon**-1)
        ) / (epsilon ** 2)

        theta = int(lambda_param / k)

        if verbose:
            print(f"Generating {theta} RR sets...")

        # Generate RR sets
        rr_sets = []
        iterator = tqdm(range(theta), disable=not verbose)
        for _ in iterator:
            if self.model == 'ic':
                rr_set = self.generate_rr_set_ic()
            else:
                rr_set = self.generate_rr_set_lt()
            rr_sets.append(rr_set)

        if verbose:
            print(f"Selecting {k} seeds...")

        # Select seeds
        selected_seeds = self.node_selection(rr_sets, k)

        # Estimate influence
        covered_sets = 0
        for rr_set in rr_sets:
            if any(seed in rr_set for seed in selected_seeds):
                covered_sets += 1

        estimated_influence = (covered_sets / theta) * self.n

        total_time = time.time() - start_time

        return selected_seeds, estimated_influence, total_time


class TIMPlus:
    """
    TIM+ algorithm with improved RR set generation.
    More efficient than basic TIM.
    """

    def __init__(self, G: nx.DiGraph, model: str = 'ic',
                 seed: Optional[int] = None):
        """
        Initialize TIM+ algorithm.

        Args:
            G: Directed graph
            model: Diffusion model
            seed: Random seed
        """
        self.G = G
        self.model = model
        self.seed = seed
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Precompute node selection probabilities for efficiency
        self.node_list = list(G.nodes())

    def generate_rr_set_ic(self) -> Set[int]:
        """Generate RR set for IC model (same as TIM)."""
        v = random.choice(self.node_list)
        rr_set = {v}

        queue = [v]
        visited = {v}

        while queue:
            current = queue.pop(0)

            for u in self.G.predecessors(current):
                if u not in visited:
                    prob = self.G[u][current].get('prob', 0.1)
                    if random.random() < prob:
                        rr_set.add(u)
                        queue.append(u)
                        visited.add(u)

        return rr_set

    def generate_rr_set_lt(self) -> Set[int]:
        """Generate RR set for LT model (same as TIM)."""
        v = random.choice(self.node_list)
        rr_set = {v}

        queue = [v]
        visited = {v}

        while queue:
            current = queue.pop(0)

            in_neighbors = list(self.G.predecessors(current))
            if len(in_neighbors) == 0:
                continue

            weights = []
            for u in in_neighbors:
                weight = self.G[u][current].get('weight',
                                               1.0 / max(self.G.in_degree(current), 1))
                weights.append(weight)

            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                selected = np.random.choice(in_neighbors, p=weights)

                if selected not in visited:
                    rr_set.add(selected)
                    queue.append(selected)
                    visited.add(selected)

        return rr_set

    def node_selection(self, rr_sets: List[Set[int]], k: int) -> List[int]:
        """Node selection (same as TIM)."""
        selected_seeds = []
        uncovered_sets = set(range(len(rr_sets)))

        node_to_rr = defaultdict(set)
        for idx, rr_set in enumerate(rr_sets):
            for node in rr_set:
                node_to_rr[node].add(idx)

        for _ in range(k):
            if len(uncovered_sets) == 0:
                break

            best_node = None
            best_coverage = 0

            for node in self.G.nodes():
                coverage = len(node_to_rr[node].intersection(uncovered_sets))
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_node = node

            if best_node is None:
                break

            selected_seeds.append(best_node)

            covered = node_to_rr[best_node].intersection(uncovered_sets)
            uncovered_sets -= covered

            for node in node_to_rr:
                node_to_rr[node] -= covered

        return selected_seeds

    def select_seeds(self, k: int, epsilon: float = 0.2,
                    verbose: bool = True) -> tuple:
        """
        Select k seeds using TIM+ algorithm.

        Args:
            k: Number of seeds
            epsilon: Error tolerance
            verbose: Whether to show progress

        Returns:
            Tuple of (selected seeds, estimated influence, total time)
        """
        start_time = time.time()

        # Improved theta calculation (TIM+)
        lambda_param = (2 + 2/3 * epsilon) * self.n * (
            math.log(self.n) + math.log(2)
        ) / (epsilon ** 2)

        theta = int(lambda_param / k)

        if verbose:
            print(f"Generating {theta} RR sets (TIM+)...")

        # Generate RR sets
        rr_sets = []
        iterator = tqdm(range(theta), disable=not verbose)
        for _ in iterator:
            if self.model == 'ic':
                rr_set = self.generate_rr_set_ic()
            else:
                rr_set = self.generate_rr_set_lt()
            rr_sets.append(rr_set)

        if verbose:
            print(f"Selecting {k} seeds...")

        # Select seeds
        selected_seeds = self.node_selection(rr_sets, k)

        # Estimate influence
        covered_sets = sum(1 for rr_set in rr_sets
                          if any(seed in rr_set for seed in selected_seeds))

        estimated_influence = (covered_sets / theta) * self.n

        total_time = time.time() - start_time

        return selected_seeds, estimated_influence, total_time
