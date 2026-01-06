"""
Run influence maximization experiments.
Compare different algorithms using true vs learned parameters.
"""

import argparse
import os
import sys
import numpy as np
import networkx as nx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import NetworkGenerator
from src.models import GraphEmbedding, ParameterLearner
from src.diffusion import DiffusionSimulator
from src.influence_max import GreedyIM, LazyGreedyIM, TIM, TIMPlus
from src.utils import (save_json, plot_influence_comparison,
                      plot_marginal_gains, plot_runtime_comparison)


def main():
    parser = argparse.ArgumentParser(description='Run influence maximization experiments')

    # Network parameters
    parser.add_argument('--network-path', type=str, default=None,
                       help='Path to network edgelist (if not provided, will generate)')
    parser.add_argument('--network-type', type=str, default='ba',
                       choices=['er', 'ba', 'ws'],
                       help='Type of network to generate (if network-path not provided)')
    parser.add_argument('--num-nodes', type=int, default=500,
                       help='Number of nodes')
    parser.add_argument('--ba-m', type=int, default=3,
                       help='BA model parameter')

    # Model parameters
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (for learned parameters)')
    parser.add_argument('--embeddings-path', type=str, default=None,
                       help='Path to embeddings file')
    parser.add_argument('--diffusion-model', type=str, default='ic',
                       choices=['ic', 'lt'],
                       help='Diffusion model')

    # IM parameters
    parser.add_argument('--algorithm', type=str, default='lazy_greedy',
                       choices=['greedy', 'lazy_greedy', 'tim', 'tim_plus'],
                       help='Influence maximization algorithm')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of seeds to select')
    parser.add_argument('--num-simulations', type=int, default=1000,
                       help='Number of MC simulations for influence estimation')
    parser.add_argument('--tim-epsilon', type=float, default=0.2,
                       help='Epsilon parameter for TIM/TIM+')

    # Comparison settings
    parser.add_argument('--compare-params', action='store_true',
                       help='Compare true vs learned parameters')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of experimental runs for averaging')

    # General parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs/im_results',
                       help='Output directory')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel simulation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for parallel simulation')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("INFLUENCE MAXIMIZATION EXPERIMENT")
    print("=" * 80)

    # Load or generate network
    print("\n[1/4] Loading/generating network...")
    if args.network_path:
        gen = NetworkGenerator(seed=args.seed)
        G_true = gen.load_from_edgelist(args.network_path)
        print(f"Loaded network from {args.network_path}")
    else:
        gen = NetworkGenerator(seed=args.seed)
        if args.network_type == 'ba':
            G_true = gen.generate_ba(args.num_nodes, args.ba_m)
        else:
            raise NotImplementedError(f"Generation for {args.network_type} not implemented in this script")

        # Assign true probabilities
        if args.diffusion_model == 'ic':
            G_true = gen.assign_ic_probabilities(G_true)

    print(f"Network: {G_true.number_of_nodes()} nodes, {G_true.number_of_edges()} edges")

    # Prepare graphs for comparison
    graphs_to_compare = {'true_params': G_true}

    # Load learned parameters if available
    if args.compare_params and args.model_path:
        print("\n[2/4] Loading learned parameters...")

        # Load embeddings
        embedding_gen = GraphEmbedding(G_true, seed=args.seed)
        if args.embeddings_path:
            embedding_gen.load_embeddings(args.embeddings_path)
            print(f"Loaded embeddings from {args.embeddings_path}")
        else:
            print("Generating embeddings...")
            embedding_gen.train_node2vec()

        # Load model
        input_dim = embedding_gen.embedding_dim * 2
        learner = ParameterLearner(input_dim=input_dim, device=args.device)
        learner.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")

        # Create graph with learned probabilities
        G_learned = G_true.copy()

        # Predict probabilities for all edges
        all_edges = list(G_learned.edges())
        edge_features = np.array([embedding_gen.get_edge_features(e) for e in all_edges])
        learned_probs = learner.predict(edge_features)

        # Assign learned probabilities
        for edge, prob in zip(all_edges, learned_probs):
            G_learned[edge[0]][edge[1]]['prob'] = float(prob)

        graphs_to_compare['learned_params'] = G_learned
        print("Graph with learned parameters created")
    else:
        print("\n[2/4] Skipping learned parameters (not requested or model not provided)")

    # Run influence maximization
    print(f"\n[3/4] Running {args.algorithm} algorithm...")
    results = {}

    for param_type, G in graphs_to_compare.items():
        print(f"\n--- Using {param_type} ---")

        all_influences = []
        all_seeds = []
        all_times = []

        for run in range(args.num_runs):
            print(f"\nRun {run + 1}/{args.num_runs}")

            # Create simulator
            sim = DiffusionSimulator(G, model=args.diffusion_model,
                                    seed=args.seed + run,
                                    parallel=args.parallel,
                                    num_workers=args.num_workers)

            # Run algorithm
            if args.algorithm == 'greedy':
                algo = GreedyIM(G, sim, seed=args.seed + run)
                seeds, gains, runtime = algo.select_seeds(args.k, args.num_simulations)
            elif args.algorithm == 'lazy_greedy':
                algo = LazyGreedyIM(G, sim, seed=args.seed + run)
                seeds, gains, runtime = algo.select_seeds(args.k, args.num_simulations)
            elif args.algorithm == 'tim':
                algo = TIM(G, model=args.diffusion_model, seed=args.seed + run)
                seeds, estimated_inf, runtime = algo.select_seeds(args.k, args.tim_epsilon)
                gains = None  # TIM doesn't provide marginal gains
            elif args.algorithm == 'tim_plus':
                algo = TIMPlus(G, model=args.diffusion_model, seed=args.seed + run)
                seeds, estimated_inf, runtime = algo.select_seeds(args.k, args.tim_epsilon)
                gains = None

            # Estimate actual influence
            actual_influence = sim.estimate_influence(seeds, args.num_simulations)

            all_influences.append(actual_influence)
            all_seeds.append(seeds)
            all_times.append(runtime)

            print(f"Selected seeds: {seeds}")
            print(f"Influence: {actual_influence:.2f}")
            print(f"Runtime: {runtime:.2f}s")

        results[param_type] = {
            'influences': all_influences,
            'seeds': all_seeds,
            'times': all_times,
            'mean_influence': float(np.mean(all_influences)),
            'std_influence': float(np.std(all_influences)),
            'mean_time': float(np.mean(all_times))
        }

    # Save results
    print(f"\n[4/4] Saving results...")

    results_summary = {
        'config': vars(args),
        'results': results
    }

    results_path = os.path.join(args.output_dir, 'im_results.json')
    save_json(results_summary, results_path)

    # Plot comparisons
    if args.compare_params and len(graphs_to_compare) > 1:
        plot_data = {k: v['influences'] for k, v in results.items()}

        comp_plot_path = os.path.join(args.output_dir, 'influence_comparison.png')
        plot_influence_comparison(plot_data, save_path=comp_plot_path)

        runtime_data = {k: v['mean_time'] for k, v in results.items()}
        runtime_plot_path = os.path.join(args.output_dir, 'runtime_comparison.png')
        plot_runtime_comparison(runtime_data, save_path=runtime_plot_path)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"\nResults Summary:")
    for param_type, res in results.items():
        print(f"\n{param_type}:")
        print(f"  Mean Influence: {res['mean_influence']:.2f} ± {res['std_influence']:.2f}")
        print(f"  Mean Runtime: {res['mean_time']:.2f}s")

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
