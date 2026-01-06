"""
Network visualization and cascade animation tools.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np
from typing import List, Set, Optional, Dict
import os


class NetworkVisualizer:
    """Visualize networks and influence propagation."""

    def __init__(self, G: nx.DiGraph, figsize: tuple = (12, 10)):
        """
        Initialize network visualizer.

        Args:
            G: NetworkX DiGraph
            figsize: Figure size
        """
        self.G = G
        self.figsize = figsize
        self.pos = None

    def compute_layout(self, layout: str = 'spring', **kwargs):
        """
        Compute node positions using specified layout.

        Args:
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
            **kwargs: Additional arguments for layout algorithm
        """
        if layout == 'spring':
            self.pos = nx.spring_layout(self.G, **kwargs)
        elif layout == 'circular':
            self.pos = nx.circular_layout(self.G, **kwargs)
        elif layout == 'kamada_kawai':
            self.pos = nx.kamada_kawai_layout(self.G, **kwargs)
        elif layout == 'spectral':
            self.pos = nx.spectral_layout(self.G, **kwargs)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def plot_network(self, highlighted_nodes: Optional[List[int]] = None,
                    node_colors: Optional[Dict[int, str]] = None,
                    save_path: Optional[str] = None,
                    show_labels: bool = False,
                    title: str = "Network Visualization"):
        """
        Plot the network.

        Args:
            highlighted_nodes: Nodes to highlight (e.g., seed nodes)
            node_colors: Dictionary mapping node to color
            save_path: Path to save figure
            show_labels: Whether to show node labels
            title: Plot title
        """
        if self.pos is None:
            self.compute_layout()

        fig, ax = plt.subplots(figsize=self.figsize)

        # Determine node colors
        if node_colors is not None:
            colors = [node_colors.get(node, 'lightblue') for node in self.G.nodes()]
        elif highlighted_nodes is not None:
            colors = ['red' if node in highlighted_nodes else 'lightblue'
                     for node in self.G.nodes()]
        else:
            colors = 'lightblue'

        # Determine node sizes based on degree
        node_sizes = [50 + 10 * self.G.out_degree(node) for node in self.G.nodes()]

        # Draw network
        nx.draw_networkx_nodes(self.G, self.pos, node_color=colors,
                              node_size=node_sizes, alpha=0.7, ax=ax)

        nx.draw_networkx_edges(self.G, self.pos, edge_color='gray',
                              alpha=0.3, arrows=True,
                              arrowsize=10, ax=ax,
                              connectionstyle='arc3,rad=0.1')

        if show_labels:
            nx.draw_networkx_labels(self.G, self.pos, font_size=8, ax=ax)

        # Add legend for highlighted nodes
        if highlighted_nodes is not None:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Seed Nodes',
                      markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Other Nodes',
                      markerfacecolor='lightblue', markersize=10)
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to {save_path}")

        plt.show()
        plt.close()

    def plot_degree_distribution(self, save_path: Optional[str] = None):
        """
        Plot degree distribution.

        Args:
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # In-degree distribution
        in_degrees = [d for n, d in self.G.in_degree()]
        axes[0].hist(in_degrees, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('In-Degree', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('In-Degree Distribution', fontsize=14)
        axes[0].set_yscale('log')

        # Out-degree distribution
        out_degrees = [d for n, d in self.G.out_degree()]
        axes[1].hist(out_degrees, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xlabel('Out-Degree', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Out-Degree Distribution', fontsize=14)
        axes[1].set_yscale('log')

        # Cumulative degree distribution (log-log)
        total_degrees = sorted([d for n, d in self.G.degree()], reverse=True)
        ccdf = np.arange(1, len(total_degrees) + 1) / len(total_degrees)
        axes[2].loglog(total_degrees, ccdf, 'o', alpha=0.5)
        axes[2].set_xlabel('Degree (log)', fontsize=12)
        axes[2].set_ylabel('P(Degree >= x) (log)', fontsize=12)
        axes[2].set_title('Cumulative Degree Distribution', fontsize=14)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Degree distribution plot saved to {save_path}")

        plt.show()
        plt.close()


class CascadeAnimator:
    """Animate cascade propagation process."""

    def __init__(self, G: nx.DiGraph, pos: Optional[Dict] = None):
        """
        Initialize cascade animator.

        Args:
            G: NetworkX DiGraph
            pos: Node positions (if None, will compute)
        """
        self.G = G
        self.pos = pos if pos is not None else nx.spring_layout(G)

    def animate_cascade(self, cascade_edges: List[tuple],
                       initial_nodes: List[int],
                       save_path: Optional[str] = None,
                       interval: int = 500,
                       title: str = "Cascade Propagation"):
        """
        Create animation of cascade propagation.

        Args:
            cascade_edges: List of edges in order of activation
            initial_nodes: Initial seed nodes
            save_path: Path to save animation (as GIF)
            interval: Delay between frames in milliseconds
            title: Animation title
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Track activated nodes over time
        activated_by_time = [set(initial_nodes)]
        current_activated = set(initial_nodes)

        # Group edges by time step
        for u, v in cascade_edges:
            if v not in current_activated:
                current_activated.add(v)
                activated_by_time.append(current_activated.copy())

        def update(frame):
            ax.clear()

            # Get activated nodes at this frame
            activated = activated_by_time[min(frame, len(activated_by_time) - 1)]

            # Color nodes
            node_colors = []
            for node in self.G.nodes():
                if node in initial_nodes:
                    node_colors.append('red')  # Seed nodes
                elif node in activated:
                    node_colors.append('orange')  # Activated nodes
                else:
                    node_colors.append('lightgray')  # Inactive nodes

            # Node sizes
            node_sizes = [100 if node in activated else 50
                         for node in self.G.nodes()]

            # Draw nodes
            nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors,
                                  node_size=node_sizes, alpha=0.8, ax=ax)

            # Draw edges
            nx.draw_networkx_edges(self.G, self.pos, edge_color='lightgray',
                                  alpha=0.2, arrows=True, ax=ax)

            # Highlight activation edges up to current frame
            active_edges = cascade_edges[:min(frame, len(cascade_edges))]
            if active_edges:
                nx.draw_networkx_edges(self.G, self.pos, edgelist=active_edges,
                                      edge_color='red', width=2, alpha=0.7,
                                      arrows=True, ax=ax)

            ax.set_title(f"{title} - Step {frame}/{len(activated_by_time)-1}\n"
                        f"Activated: {len(activated)} nodes",
                        fontsize=14, fontweight='bold')
            ax.axis('off')

        # Create animation
        anim = animation.FuncAnimation(fig, update,
                                      frames=len(activated_by_time),
                                      interval=interval,
                                      repeat=True)

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000/interval)
                print(f"Animation saved to {save_path}")
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000/interval)
                print(f"Animation saved to {save_path}")
            else:
                print("Warning: save_path should end with .gif or .mp4")

        plt.show()
        plt.close()

        return anim

    def create_cascade_snapshots(self, cascade_edges: List[tuple],
                                 initial_nodes: List[int],
                                 num_snapshots: int = 5,
                                 save_dir: Optional[str] = None):
        """
        Create snapshots of cascade at different time steps.

        Args:
            cascade_edges: List of edges in order of activation
            initial_nodes: Initial seed nodes
            num_snapshots: Number of snapshots to create
            save_dir: Directory to save snapshots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Build activation timeline
        activated_by_step = {0: set(initial_nodes)}
        current_activated = set(initial_nodes)

        for i, (u, v) in enumerate(cascade_edges, 1):
            if v not in current_activated:
                current_activated.add(v)
                activated_by_step[i] = current_activated.copy()

        # Select evenly spaced snapshots
        total_steps = len(activated_by_step)
        snapshot_steps = np.linspace(0, total_steps - 1, num_snapshots, dtype=int)

        fig, axes = plt.subplots(1, num_snapshots, figsize=(5 * num_snapshots, 5))
        if num_snapshots == 1:
            axes = [axes]

        for idx, step in enumerate(snapshot_steps):
            ax = axes[idx]

            activated = activated_by_step[step]

            # Color nodes
            node_colors = []
            for node in self.G.nodes():
                if node in initial_nodes:
                    node_colors.append('red')
                elif node in activated:
                    node_colors.append('orange')
                else:
                    node_colors.append('lightgray')

            # Draw
            nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors,
                                  node_size=100, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(self.G, self.pos, edge_color='lightgray',
                                  alpha=0.2, arrows=True, ax=ax)

            ax.set_title(f"Step {step}\n{len(activated)} nodes",
                        fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'cascade_snapshots.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cascade snapshots saved to {save_path}")

        plt.show()
        plt.close()
