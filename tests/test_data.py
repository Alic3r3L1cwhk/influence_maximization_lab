"""
Unit tests for data generation modules.
"""

import unittest
import networkx as nx
import numpy as np
from src.data import NetworkGenerator, CascadeGenerator, DataSplitter


class TestNetworkGenerator(unittest.TestCase):
    """Test network generation."""

    def setUp(self):
        self.gen = NetworkGenerator(seed=42)

    def test_generate_ba(self):
        """Test BA network generation."""
        G = self.gen.generate_ba(n=100, m=3)

        self.assertEqual(G.number_of_nodes(), 100)
        self.assertGreater(G.number_of_edges(), 0)
        self.assertIsInstance(G, nx.DiGraph)

    def test_generate_er(self):
        """Test ER network generation."""
        G = self.gen.generate_er(n=100, p=0.05)

        self.assertEqual(G.number_of_nodes(), 100)
        self.assertGreater(G.number_of_edges(), 0)
        self.assertIsInstance(G, nx.DiGraph)

    def test_generate_ws(self):
        """Test WS network generation."""
        G = self.gen.generate_ws(n=100, k=4, p=0.1)

        self.assertEqual(G.number_of_nodes(), 100)
        self.assertGreater(G.number_of_edges(), 0)
        self.assertIsInstance(G, nx.DiGraph)

    def test_assign_ic_probabilities(self):
        """Test IC probability assignment."""
        G = self.gen.generate_ba(n=50, m=2)
        G = self.gen.assign_ic_probabilities(G, prob_range=(0.01, 0.1))

        # Check all edges have prob attribute
        for u, v in G.edges():
            self.assertIn('prob', G[u][v])
            prob = G[u][v]['prob']
            self.assertGreaterEqual(prob, 0.01)
            self.assertLessEqual(prob, 0.1)

    def test_reproducibility(self):
        """Test random seed reproducibility."""
        gen1 = NetworkGenerator(seed=42)
        gen2 = NetworkGenerator(seed=42)

        G1 = gen1.generate_ba(n=50, m=2)
        G2 = gen2.generate_ba(n=50, m=2)

        # Should have same number of edges (due to seed)
        self.assertEqual(G1.number_of_edges(), G2.number_of_edges())


class TestCascadeGenerator(unittest.TestCase):
    """Test cascade generation."""

    def setUp(self):
        gen = NetworkGenerator(seed=42)
        self.G = gen.generate_ba(n=50, m=2)
        self.G = gen.assign_ic_probabilities(self.G)
        self.cascade_gen = CascadeGenerator(self.G, seed=42)

    def test_generate_cascades(self):
        """Test cascade generation."""
        cascades = self.cascade_gen.generate_cascades('ic', num_cascades=10)

        self.assertEqual(len(cascades), 10)

        for cascade in cascades:
            self.assertIn('initial_nodes', cascade)
            self.assertIn('activated_nodes', cascade)
            self.assertIn('edges', cascade)
            self.assertIn('cascade_size', cascade)

            # Initial nodes should be in activated nodes
            for node in cascade['initial_nodes']:
                self.assertIn(node, cascade['activated_nodes'])

    def test_cascades_to_training_data(self):
        """Test conversion to training data."""
        cascades = self.cascade_gen.generate_cascades('ic', num_cascades=20)
        edges, labels = self.cascade_gen.cascades_to_training_data(cascades)

        self.assertEqual(len(edges), len(labels))
        self.assertGreater(len(edges), 0)

        # Labels should be between 0 and 1
        for label in labels:
            self.assertGreaterEqual(label, 0)
            self.assertLessEqual(label, 1)


class TestDataSplitter(unittest.TestCase):
    """Test data splitting."""

    def setUp(self):
        self.splitter = DataSplitter(seed=42)

    def test_split_edges(self):
        """Test edge data splitting."""
        # Create dummy data
        edges = [(i, i+1) for i in range(100)]
        labels = [0.5] * 100

        train_e, train_l, val_e, val_l, test_e, test_l = \
            self.splitter.split_edges(edges, labels, 0.7, 0.15, 0.15)

        # Check sizes
        self.assertEqual(len(train_e), 70)
        self.assertEqual(len(val_e), 15)
        self.assertEqual(len(test_e), 15)

        # Check labels match
        self.assertEqual(len(train_e), len(train_l))
        self.assertEqual(len(val_e), len(val_l))
        self.assertEqual(len(test_e), len(test_l))

    def test_reproducibility(self):
        """Test splitting reproducibility."""
        edges = [(i, i+1) for i in range(100)]
        labels = [0.5] * 100

        splitter1 = DataSplitter(seed=42)
        splitter2 = DataSplitter(seed=42)

        result1 = splitter1.split_edges(edges, labels)
        result2 = splitter2.split_edges(edges, labels)

        # Train sets should be identical
        self.assertEqual(result1[0], result2[0])


if __name__ == '__main__':
    unittest.main()
