"""
Graph embedding module using Node2Vec and DeepWalk.
Provides node embeddings as features for parameter learning.
"""

import networkx as nx
import numpy as np
from typing import Optional, Dict
from gensim.models import Word2Vec
import random


class RandomWalker:
    """Generate random walks on graphs."""

    def __init__(self, G: nx.Graph, seed: Optional[int] = None):
        """
        Initialize the random walker.

        Args:
            G: NetworkX graph
            seed: Random seed
        """
        self.G = G
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def deepwalk_walk(self, start_node: int, walk_length: int) -> list:
        """
        Generate a random walk (DeepWalk style - uniform random).

        Args:
            start_node: Starting node
            walk_length: Length of the walk

        Returns:
            List of nodes in the walk
        """
        walk = [start_node]

        for _ in range(walk_length - 1):
            current = walk[-1]
            neighbors = list(self.G.neighbors(current))
            if len(neighbors) > 0:
                walk.append(random.choice(neighbors))
            else:
                break

        return walk

    def node2vec_walk(self, start_node: int, walk_length: int,
                     p: float = 1.0, q: float = 1.0) -> list:
        """
        Generate a biased random walk (Node2Vec style).

        Args:
            start_node: Starting node
            walk_length: Length of the walk
            p: Return parameter
            q: In-out parameter

        Returns:
            List of nodes in the walk
        """
        walk = [start_node]

        for _ in range(walk_length - 1):
            current = walk[-1]
            neighbors = list(self.G.neighbors(current))

            if len(neighbors) == 0:
                break

            if len(walk) == 1:
                # First step: uniform random
                walk.append(random.choice(neighbors))
            else:
                prev = walk[-2]
                probs = []

                for neighbor in neighbors:
                    if neighbor == prev:
                        # Return to previous node
                        probs.append(1.0 / p)
                    elif self.G.has_edge(prev, neighbor):
                        # Stay close
                        probs.append(1.0)
                    else:
                        # Move away
                        probs.append(1.0 / q)

                # Normalize probabilities
                probs = np.array(probs)
                probs = probs / probs.sum()

                walk.append(np.random.choice(neighbors, p=probs))

        return walk

    def generate_walks(self, num_walks: int, walk_length: int,
                      strategy: str = 'deepwalk',
                      p: float = 1.0, q: float = 1.0) -> list:
        """
        Generate random walks for all nodes.

        Args:
            num_walks: Number of walks per node
            walk_length: Length of each walk
            strategy: 'deepwalk' or 'node2vec'
            p: Return parameter (for node2vec)
            q: In-out parameter (for node2vec)

        Returns:
            List of walks (each walk is a list of nodes)
        """
        walks = []
        nodes = list(self.G.nodes())

        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if strategy == 'deepwalk':
                    walk = self.deepwalk_walk(node, walk_length)
                elif strategy == 'node2vec':
                    walk = self.node2vec_walk(node, walk_length, p, q)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                walks.append([str(n) for n in walk])  # Convert to strings for Word2Vec

        return walks


class GraphEmbedding:
    """Generate graph embeddings using DeepWalk or Node2Vec."""

    def __init__(self, G: nx.Graph, embedding_dim: int = 128,
                 seed: Optional[int] = None):
        """
        Initialize the graph embedding generator.

        Args:
            G: NetworkX graph
            embedding_dim: Dimension of embeddings
            seed: Random seed
        """
        self.G = G
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.model = None
        self.embeddings = None

    def train_deepwalk(self, num_walks: int = 10, walk_length: int = 80,
                      window_size: int = 10, epochs: int = 5,
                      workers: int = 4) -> Dict[int, np.ndarray]:
        """
        Train DeepWalk embeddings.

        Args:
            num_walks: Number of walks per node
            walk_length: Length of each walk
            window_size: Window size for Word2Vec
            epochs: Number of training epochs
            workers: Number of parallel workers

        Returns:
            Dictionary mapping node ID to embedding vector
        """
        walker = RandomWalker(self.G, seed=self.seed)
        walks = walker.generate_walks(num_walks, walk_length, strategy='deepwalk')

        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=window_size,
            min_count=0,
            sg=1,  # Skip-gram
            workers=workers,
            epochs=epochs,
            seed=self.seed
        )

        # Extract embeddings
        self.embeddings = {}
        for node in self.G.nodes():
            node_str = str(node)
            if node_str in self.model.wv:
                self.embeddings[node] = self.model.wv[node_str]
            else:
                # Handle missing nodes with zero vector
                self.embeddings[node] = np.zeros(self.embedding_dim)

        return self.embeddings

    def train_node2vec(self, num_walks: int = 10, walk_length: int = 80,
                      p: float = 1.0, q: float = 1.0,
                      window_size: int = 10, epochs: int = 5,
                      workers: int = 4) -> Dict[int, np.ndarray]:
        """
        Train Node2Vec embeddings.

        Args:
            num_walks: Number of walks per node
            walk_length: Length of each walk
            p: Return parameter
            q: In-out parameter
            window_size: Window size for Word2Vec
            epochs: Number of training epochs
            workers: Number of parallel workers

        Returns:
            Dictionary mapping node ID to embedding vector
        """
        walker = RandomWalker(self.G, seed=self.seed)
        walks = walker.generate_walks(num_walks, walk_length,
                                     strategy='node2vec', p=p, q=q)

        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=window_size,
            min_count=0,
            sg=1,  # Skip-gram
            workers=workers,
            epochs=epochs,
            seed=self.seed
        )

        # Extract embeddings
        self.embeddings = {}
        for node in self.G.nodes():
            node_str = str(node)
            if node_str in self.model.wv:
                self.embeddings[node] = self.model.wv[node_str]
            else:
                # Handle missing nodes with zero vector
                self.embeddings[node] = np.zeros(self.embedding_dim)

        return self.embeddings

    def get_edge_features(self, edge: tuple) -> np.ndarray:
        """
        Get feature vector for an edge by concatenating source and target embeddings.

        Args:
            edge: Tuple of (source, target)

        Returns:
            Concatenated embedding vector
        """
        if self.embeddings is None:
            raise ValueError("Must train embeddings first")

        u, v = edge
        u_emb = self.embeddings.get(u, np.zeros(self.embedding_dim))
        v_emb = self.embeddings.get(v, np.zeros(self.embedding_dim))

        # Concatenate embeddings
        return np.concatenate([u_emb, v_emb])

    def save_embeddings(self, filepath: str):
        """Save embeddings to file."""
        if self.embeddings is None:
            raise ValueError("No embeddings to save")

        with open(filepath, 'w') as f:
            f.write(f"{len(self.embeddings)} {self.embedding_dim}\n")
            for node, emb in self.embeddings.items():
                emb_str = ' '.join(map(str, emb))
                f.write(f"{node} {emb_str}\n")

    def load_embeddings(self, filepath: str) -> Dict[int, np.ndarray]:
        """Load embeddings from file."""
        self.embeddings = {}

        with open(filepath, 'r') as f:
            first_line = f.readline().strip().split()
            num_nodes, dim = int(first_line[0]), int(first_line[1])
            self.embedding_dim = dim

            for line in f:
                parts = line.strip().split()
                node = int(parts[0])
                emb = np.array([float(x) for x in parts[1:]])
                self.embeddings[node] = emb

        return self.embeddings
