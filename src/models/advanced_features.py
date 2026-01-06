"""
Advanced feature engineering for edges and nodes.
Includes structural features beyond embeddings.
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class StructuralFeatures:
    """Extract structural features from graphs."""

    def __init__(self, G: nx.DiGraph):
        """
        Initialize structural feature extractor.

        Args:
            G: NetworkX DiGraph
        """
        self.G = G
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()

        # Cache for computed features
        self._degree_features = None
        self._centrality_features = None
        self._local_features = None

    def compute_degree_features(self) -> Dict[int, Dict[str, float]]:
        """
        Compute degree-based features for each node.

        Returns:
            Dictionary mapping node to feature dict
        """
        if self._degree_features is not None:
            return self._degree_features

        features = {}

        for node in self.G.nodes():
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
            total_deg = in_deg + out_deg

            features[node] = {
                'in_degree': in_deg,
                'out_degree': out_deg,
                'total_degree': total_deg,
                'degree_ratio': out_deg / max(in_deg, 1),
                'in_degree_norm': in_deg / self.n,
                'out_degree_norm': out_deg / self.n
            }

        self._degree_features = features
        return features

    def compute_centrality_features(self) -> Dict[int, Dict[str, float]]:
        """
        Compute centrality features for each node.

        Returns:
            Dictionary mapping node to feature dict
        """
        if self._centrality_features is not None:
            return self._centrality_features

        print("Computing centrality features...")

        # PageRank
        pagerank = nx.pagerank(self.G)

        # Betweenness (approximate for large graphs)
        if self.n > 1000:
            # Use approximate betweenness for large graphs
            k = min(100, self.n)
            betweenness = nx.betweenness_centrality(self.G, k=k)
        else:
            betweenness = nx.betweenness_centrality(self.G)

        # Closeness (can be expensive)
        try:
            closeness = nx.closeness_centrality(self.G)
        except:
            closeness = {node: 0 for node in self.G.nodes()}

        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(self.G, max_iter=100)
        except:
            eigenvector = {node: 0 for node in self.G.nodes()}

        # Combine features
        features = {}
        for node in self.G.nodes():
            features[node] = {
                'pagerank': pagerank.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'closeness': closeness.get(node, 0),
                'eigenvector': eigenvector.get(node, 0)
            }

        self._centrality_features = features
        return features

    def compute_local_features(self) -> Dict[int, Dict[str, float]]:
        """
        Compute local neighborhood features.

        Returns:
            Dictionary mapping node to feature dict
        """
        if self._local_features is not None:
            return self._local_features

        features = {}

        # Convert to undirected for clustering coefficient
        G_undirected = self.G.to_undirected()
        clustering = nx.clustering(G_undirected)

        # Core number
        core_number = nx.core_number(G_undirected)

        for node in self.G.nodes():
            # 2-hop neighborhood size
            neighbors = set(self.G.successors(node))
            two_hop_neighbors = set()
            for neighbor in neighbors:
                two_hop_neighbors.update(self.G.successors(neighbor))
            two_hop_neighbors -= neighbors
            two_hop_neighbors.discard(node)

            features[node] = {
                'clustering_coefficient': clustering.get(node, 0),
                'core_number': core_number.get(node, 0),
                'num_neighbors': len(neighbors),
                'num_2hop_neighbors': len(two_hop_neighbors)
            }

        self._local_features = features
        return features

    def get_all_node_features(self, node: int) -> np.ndarray:
        """
        Get all structural features for a node.

        Args:
            node: Node ID

        Returns:
            Feature vector
        """
        degree_feat = self.compute_degree_features()[node]
        centrality_feat = self.compute_centrality_features()[node]
        local_feat = self.compute_local_features()[node]

        # Combine all features
        features = []
        features.extend(degree_feat.values())
        features.extend(centrality_feat.values())
        features.extend(local_feat.values())

        return np.array(features, dtype=np.float32)

    def get_edge_features(self, edge: Tuple[int, int]) -> np.ndarray:
        """
        Get structural features for an edge.

        Args:
            edge: Tuple of (source, target)

        Returns:
            Feature vector
        """
        u, v = edge

        # Get node features
        u_features = self.get_all_node_features(u)
        v_features = self.get_all_node_features(v)

        # Edge-specific features
        edge_features = []

        # Common neighbors
        u_neighbors = set(self.G.successors(u))
        v_neighbors = set(self.G.successors(v))
        common = len(u_neighbors.intersection(v_neighbors))
        edge_features.append(common)

        # Jaccard coefficient
        union = len(u_neighbors.union(v_neighbors))
        jaccard = common / max(union, 1)
        edge_features.append(jaccard)

        # Adamic-Adar index
        aa_index = 0
        for w in u_neighbors.intersection(v_neighbors):
            aa_index += 1.0 / max(np.log(self.G.out_degree(w) + 1), 1)
        edge_features.append(aa_index)

        # Preferential attachment
        pref_attach = self.G.out_degree(u) * self.G.out_degree(v)
        edge_features.append(pref_attach)

        # Combine: source features + target features + edge features
        combined = np.concatenate([
            u_features,
            v_features,
            np.array(edge_features, dtype=np.float32)
        ])

        return combined

    def get_feature_names(self) -> List[str]:
        """Get feature names in order."""
        names = []

        # Degree features
        names.extend(['in_degree', 'out_degree', 'total_degree',
                     'degree_ratio', 'in_degree_norm', 'out_degree_norm'])

        # Centrality features
        names.extend(['pagerank', 'betweenness', 'closeness', 'eigenvector'])

        # Local features
        names.extend(['clustering_coefficient', 'core_number',
                     'num_neighbors', 'num_2hop_neighbors'])

        return names


class CombinedFeatures:
    """Combine structural features with embeddings."""

    def __init__(self, G: nx.DiGraph, embeddings: Dict[int, np.ndarray] = None):
        """
        Initialize combined feature extractor.

        Args:
            G: NetworkX DiGraph
            embeddings: Optional pre-computed embeddings
        """
        self.G = G
        self.embeddings = embeddings
        self.structural = StructuralFeatures(G)

    def get_node_features(self, node: int,
                         use_structural: bool = True,
                         use_embedding: bool = True) -> np.ndarray:
        """
        Get combined features for a node.

        Args:
            node: Node ID
            use_structural: Whether to include structural features
            use_embedding: Whether to include embeddings

        Returns:
            Combined feature vector
        """
        features = []

        if use_structural:
            struct_feat = self.structural.get_all_node_features(node)
            features.append(struct_feat)

        if use_embedding and self.embeddings is not None:
            emb = self.embeddings.get(node, np.zeros(128))
            features.append(emb)

        if len(features) == 0:
            raise ValueError("At least one feature type must be enabled")

        return np.concatenate(features)

    def get_edge_features(self, edge: Tuple[int, int],
                         use_structural: bool = True,
                         use_embedding: bool = True,
                         edge_operator: str = 'concat') -> np.ndarray:
        """
        Get combined features for an edge.

        Args:
            edge: Tuple of (source, target)
            use_structural: Whether to include structural features
            use_embedding: Whether to include embeddings
            edge_operator: How to combine source/target features
                          'concat', 'hadamard', 'average', 'l1', 'l2'

        Returns:
            Combined feature vector
        """
        u, v = edge

        if use_embedding and self.embeddings is not None:
            u_emb = self.embeddings.get(u, np.zeros(128))
            v_emb = self.embeddings.get(v, np.zeros(128))

            # Combine embeddings
            if edge_operator == 'concat':
                emb_feat = np.concatenate([u_emb, v_emb])
            elif edge_operator == 'hadamard':
                emb_feat = u_emb * v_emb
            elif edge_operator == 'average':
                emb_feat = (u_emb + v_emb) / 2
            elif edge_operator == 'l1':
                emb_feat = np.abs(u_emb - v_emb)
            elif edge_operator == 'l2':
                emb_feat = (u_emb - v_emb) ** 2
            else:
                raise ValueError(f"Unknown edge operator: {edge_operator}")

        if use_structural:
            struct_feat = self.structural.get_edge_features(edge)

            if use_embedding and self.embeddings is not None:
                return np.concatenate([struct_feat, emb_feat])
            else:
                return struct_feat
        else:
            if use_embedding and self.embeddings is not None:
                return emb_feat
            else:
                raise ValueError("At least one feature type must be enabled")
