"""
IMM (Influence Maximization via Martingales) algorithm.
More efficient than TIM with better theoretical guarantees.
Reference: Tang et al. "Influence Maximization in Near-Linear Time: A Martingale Approach" (SIGMOD 2015)
"""

import networkx as nx
import numpy as np
from typing import List, Set, Optional
import random
import math
import time
from tqdm import tqdm
from collections import defaultdict


class IMM:
    """
    Influence Maximization via Martingales (IMM) algorithm.
    More efficient than TIM with adaptive RR set generation.
    """

    def __init__(self, G: nx.DiGraph, model: str = 'ic', seed: Optional[int] = None):
        """
        Initialize IMM algorithm.

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

        self.node_list = list(G.nodes())

    def generate_rr_set_ic(self) -> Set[int]:
        """Generate RR set for IC model."""
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
        """Generate RR set for LT model."""
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
        """Select k nodes that cover most RR sets."""
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

    def kpt_estimation(self, k: int, epsilon: float) -> int:
        """
        KPT estimation for number of RR sets needed.

        Args:
            k: Number of seeds
            epsilon: Error tolerance

        Returns:
            Number of RR sets theta
        """
        # KPT bound
        lambda_prime = (2 + 2.0/3 * epsilon) * (
            np.log(self.n) + np.log(2) + np.log(np.log2(self.n))
        ) * self.n / (epsilon ** 2)

        return int(lambda_prime / k)

    def select_seeds(self, k: int, epsilon: float = 0.1, verbose: bool = True) -> tuple:
        """
        Select k seeds using IMM algorithm with adaptive sampling.

        Args:
            k: Number of seeds
            epsilon: Error tolerance (smaller = more accurate but slower)
            verbose: Whether to show progress

        Returns:
            Tuple of (selected seeds, estimated influence, total time)
        """
        start_time = time.time()

        if verbose:
            print(f"Running IMM with epsilon={epsilon}")

        # Step 1: Parameter initialization
        eps_prime = epsilon * math.sqrt(2)

        # Step 2: Generate initial RR sets for estimation
        theta_initial = self.kpt_estimation(k, eps_prime)

        if verbose:
            print(f"Initial theta estimation: {theta_initial}")
            print("Generating RR sets...")

        # Generate RR sets
        rr_sets = []
        iterator = tqdm(range(theta_initial), disable=not verbose)

        for _ in iterator:
            if self.model == 'ic':
                rr_set = self.generate_rr_set_ic()
            else:
                rr_set = self.generate_rr_set_lt()
            rr_sets.append(rr_set)

        if verbose:
            print(f"Selecting {k} seeds...")

        # Step 3: Node selection
        selected_seeds = self.node_selection(rr_sets, k)

        # Step 4: Estimate influence
        covered_sets = sum(1 for rr_set in rr_sets
                          if any(seed in rr_set for seed in selected_seeds))
        estimated_influence = (covered_sets / len(rr_sets)) * self.n

        total_time = time.time() - start_time

        if verbose:
            print(f"Selected seeds: {selected_seeds}")
            print(f"Estimated influence: {estimated_influence:.2f}")
            print(f"Runtime: {total_time:.2f}s")

        return selected_seeds, estimated_influence, total_time


class SSA:
    """
    Stop-and-Stare Algorithm for IMM.
    Used for adaptive RR set generation.
    """

    def __init__(self, G: nx.DiGraph, model: str = 'ic', seed: Optional[int] = None):
        self.G = G
        self.model = model
        self.seed = seed
        self.n = G.number_of_nodes()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def lambda_star(self, k: int, epsilon: float, ell: float) -> float:
        """Calculate lambda* parameter."""
        alpha = math.sqrt(math.log(self.n) + math.log(2))
        beta = math.sqrt((1 - 1/math.e) * (
            math.log(self.n) + math.log(2) + ell * math.log(2)
        ))

        lambda_val = 2 * self.n * ((1 - 1/math.e) * alpha + beta) ** 2 * epsilon ** (-2)

        return lambda_val / k
