"""
Heuristic-based influence maximization algorithms.
Fast baseline methods based on centrality measures.
"""

import networkx as nx
import numpy as np
from typing import List, Optional
import time


class DegreeHeuristic:
    """Select nodes with highest degree."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None):
        self.G = G
        self.seed = seed

    def select_seeds(self, k: int) -> tuple:
        """
        Select k nodes with highest out-degree.

        Args:
            k: Number of seeds

        Returns:
            Tuple of (selected seeds, runtime)
        """
        start_time = time.time()

        # Calculate out-degree for all nodes
        degrees = dict(self.G.out_degree())

        # Sort by degree (descending)
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        # Select top k
        selected_seeds = [node for node, _ in sorted_nodes[:k]]

        runtime = time.time() - start_time

        return selected_seeds, runtime


class PageRankHeuristic:
    """Select nodes with highest PageRank."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None):
        self.G = G
        self.seed = seed

    def select_seeds(self, k: int, alpha: float = 0.85) -> tuple:
        """
        Select k nodes with highest PageRank.

        Args:
            k: Number of seeds
            alpha: Damping parameter for PageRank

        Returns:
            Tuple of (selected seeds, runtime)
        """
        start_time = time.time()

        # Calculate PageRank
        pagerank = nx.pagerank(self.G, alpha=alpha)

        # Sort by PageRank (descending)
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

        # Select top k
        selected_seeds = [node for node, _ in sorted_nodes[:k]]

        runtime = time.time() - start_time

        return selected_seeds, runtime


class BetweennessHeuristic:
    """Select nodes with highest betweenness centrality."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None):
        self.G = G
        self.seed = seed

    def select_seeds(self, k: int) -> tuple:
        """
        Select k nodes with highest betweenness centrality.

        Args:
            k: Number of seeds

        Returns:
            Tuple of (selected seeds, runtime)
        """
        start_time = time.time()

        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(self.G)

        # Sort by betweenness (descending)
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

        # Select top k
        selected_seeds = [node for node, _ in sorted_nodes[:k]]

        runtime = time.time() - start_time

        return selected_seeds, runtime


class ClosenessCentralityHeuristic:
    """Select nodes with highest closeness centrality."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None):
        self.G = G
        self.seed = seed

    def select_seeds(self, k: int) -> tuple:
        """
        Select k nodes with highest closeness centrality.

        Args:
            k: Number of seeds

        Returns:
            Tuple of (selected seeds, runtime)
        """
        start_time = time.time()

        # Calculate closeness centrality
        try:
            closeness = nx.closeness_centrality(self.G)
        except:
            # If graph is not strongly connected, use weakly connected
            closeness = {}
            for node in self.G.nodes():
                try:
                    closeness[node] = nx.closeness_centrality(self.G, node)
                except:
                    closeness[node] = 0

        # Sort by closeness (descending)
        sorted_nodes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)

        # Select top k
        selected_seeds = [node for node, _ in sorted_nodes[:k]]

        runtime = time.time() - start_time

        return selected_seeds, runtime


class KShellHeuristic:
    """Select nodes with highest k-shell (coreness)."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None):
        self.G = G
        self.seed = seed

    def select_seeds(self, k: int) -> tuple:
        """
        Select k nodes with highest k-shell value.

        Args:
            k: Number of seeds

        Returns:
            Tuple of (selected seeds, runtime)
        """
        start_time = time.time()

        # Convert to undirected for k-core
        G_undirected = self.G.to_undirected()

        # Calculate k-core (shell)
        core_numbers = nx.core_number(G_undirected)

        # Sort by core number (descending)
        sorted_nodes = sorted(core_numbers.items(), key=lambda x: x[1], reverse=True)

        # Select top k
        selected_seeds = [node for node, _ in sorted_nodes[:k]]

        runtime = time.time() - start_time

        return selected_seeds, runtime


class RandomHeuristic:
    """Random node selection baseline."""

    def __init__(self, G: nx.DiGraph, seed: Optional[int] = None):
        self.G = G
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def select_seeds(self, k: int) -> tuple:
        """
        Randomly select k nodes.

        Args:
            k: Number of seeds

        Returns:
            Tuple of (selected seeds, runtime)
        """
        start_time = time.time()

        nodes = list(self.G.nodes())
        selected_seeds = list(np.random.choice(nodes, size=min(k, len(nodes)),
                                              replace=False))

        runtime = time.time() - start_time

        return selected_seeds, runtime
