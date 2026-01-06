"""
Unified simulator interface for different diffusion models.
"""

import networkx as nx
from typing import List, Optional
from .ic_model import ICModel, ICModelParallel
from .lt_model import LTModel, LTModelParallel


class DiffusionSimulator:
    """Unified interface for diffusion simulation."""

    def __init__(self, G: nx.DiGraph, model: str = 'ic',
                 seed: Optional[int] = None,
                 parallel: bool = False,
                 num_workers: int = 4):
        """
        Initialize the diffusion simulator.

        Args:
            G: Directed graph
            model: Diffusion model ('ic' or 'lt')
            seed: Random seed
            parallel: Whether to use parallel simulation
            num_workers: Number of workers for parallel simulation
        """
        self.G = G
        self.model_type = model
        self.seed = seed

        if model == 'ic':
            if parallel:
                self.model = ICModelParallel(G, seed=seed, num_workers=num_workers)
            else:
                self.model = ICModel(G, seed=seed)
        elif model == 'lt':
            if parallel:
                self.model = LTModelParallel(G, seed=seed, num_workers=num_workers)
            else:
                self.model = LTModel(G, seed=seed)
        else:
            raise ValueError(f"Unknown diffusion model: {model}")

    def simulate_single(self, initial_nodes: List[int]):
        """Simulate a single diffusion process."""
        if hasattr(self.model, 'simulate_single'):
            return self.model.simulate_single(initial_nodes)
        else:
            raise NotImplementedError("Parallel models don't support single simulation")

    def estimate_influence(self, initial_nodes: List[int],
                         num_simulations: int = 1000) -> float:
        """
        Estimate expected influence.

        Args:
            initial_nodes: List of seed nodes
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Expected number of activated nodes
        """
        if hasattr(self.model, 'simulate_mc_parallel'):
            return self.model.simulate_mc_parallel(initial_nodes, num_simulations)
        else:
            return self.model.simulate_mc(initial_nodes, num_simulations)

    def estimate_batch(self, seed_sets: List[List[int]],
                      num_simulations: int = 1000) -> List[float]:
        """
        Estimate influence for multiple seed sets.

        Args:
            seed_sets: List of seed node lists
            num_simulations: Number of simulations per seed set

        Returns:
            List of expected influences
        """
        if hasattr(self.model, 'simulate_batch_parallel'):
            return self.model.simulate_batch_parallel(seed_sets, num_simulations)
        else:
            return self.model.simulate_batch(seed_sets, num_simulations)
