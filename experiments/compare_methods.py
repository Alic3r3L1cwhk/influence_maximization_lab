"""
Comprehensive comparison of influence maximization methods.
Compares multiple algorithms with both true and learned parameters.
"""

import argparse
import os
import sys
import numpy as np
import networkx as nx
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import NetworkGenerator, CascadeGenerator
from src.models import GraphEmbedding, ParameterLearner
from src.diffusion import DiffusionSimulator
from src.influence_max import GreedyIM, LazyGreedyIM, TIM, TIMPlus
from src.utils import (save_json, save_csv, plot_influence_comparison,
                      plot_runtime_comparison, plot_multiple_experiments)


def run_algorithm(algo_name, G, diffusion_model, k, num_simulations, tim_epsilon, seed):
    """Run a single influence maximization algorithm."""

    sim = DiffusionSimulator(G, model=diffusion_model, seed=seed)

    start_time = time.time()

    if algo_name == 'greedy':
        algo = GreedyIM(G, sim, seed=seed)
        seeds, _, _ = algo.select_seeds(k, num_simulations, verbose=False)
    elif algo_name == 'lazy_greedy':
        algo = LazyGreedyIM(G, sim, seed=seed)
        seeds, _, _ = algo.select_seeds(k, num_simulations, verbose=False)
    elif algo_name == 'tim':
        algo = TIM(G, model=diffusion_model, seed=seed)
        seeds, _, _ = algo.select_seeds(k, tim_epsilon, verbose=False)
    elif algo_name == 'tim_plus':
        algo = TIMPlus(G, model=diffusion_model, seed=seed)
        seeds, _, _ = algo.select_seeds(k, tim_epsilon, verbose=False)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    runtime = time.time() - start_time

    # Estimate influence
    influence = sim.estimate_influence(seeds, num_simulations)

    return seeds, influence, runtime


def main():
    parser = argparse.ArgumentParser(description='Comprehensive IM comparison')

    # Network parameters
    parser.add_argument('--network-type', type=str, default='ba')
    parser.add_argument('--num-nodes', type=int, default=500)
    parser.add_argument('--ba-m', type=int, default=3)

    # Experiment parameters
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 20, 30],
                       help='Different k values to test')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['lazy_greedy', 'tim', 'tim_plus'],
                       choices=['greedy', 'lazy_greedy', 'tim', 'tim_plus'],
                       help='Algorithms to compare')
    parser.add_argument('--diffusion-model', type=str, default='ic')
    parser.add_argument('--num-simulations', type=int, default=1000)
    parser.add_argument('--tim-epsilon', type=float, default=0.2)
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs per configuration')

    # Training parameters (for learned params comparison)
    parser.add_argument('--train-params', action='store_true',
                       help='Train parameters from cascades')
    parser.add_argument('--num-cascades', type=int, default=1000)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)

    # General
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE INFLUENCE MAXIMIZATION COMPARISON")
    print("=" * 80)

    # Generate network
    print("\n[1/5] Generating network...")
    gen = NetworkGenerator(seed=args.seed)
    G_true = gen.generate_ba(args.num_nodes, args.ba_m)
    G_true = gen.assign_ic_probabilities(G_true)
    print(f"Network: {G_true.number_of_nodes()} nodes, {G_true.number_of_edges()} edges")

    # Save network
    nx.write_edgelist(G_true, os.path.join(args.output_dir, 'network.edgelist'),
                     data=['prob'])

    graphs = {'True Parameters': G_true}

    # Train learned parameters if requested
    if args.train_params:
        print("\n[2/5] Training learned parameters...")

        # Generate cascades
        print("Generating cascades...")
        cascade_gen = CascadeGenerator(G_true, seed=args.seed)
        cascades = cascade_gen.generate_cascades(args.diffusion_model, args.num_cascades)
        edges, labels = cascade_gen.cascades_to_training_data(cascades)

        # Generate embeddings
        print("Generating embeddings...")
        embedding_gen = GraphEmbedding(G_true, embedding_dim=args.embedding_dim,
                                      seed=args.seed)
        embeddings = embedding_gen.train_node2vec(num_walks=10, walk_length=80, verbose=False)

        # Prepare features
        features = np.array([embedding_gen.get_edge_features(e) for e in edges])
        labels = np.array(labels)

        # Train model
        print("Training model...")
        learner = ParameterLearner(
            input_dim=features.shape[1],
            hidden_dims=[256, 128, 64],
            device=args.device,
            seed=args.seed
        )

        learner.fit(features, labels, epochs=args.epochs, verbose=False)

        # Create learned graph
        G_learned = G_true.copy()
        all_edges = list(G_learned.edges())
        edge_features = np.array([embedding_gen.get_edge_features(e) for e in all_edges])
        learned_probs = learner.predict(edge_features)

        for edge, prob in zip(all_edges, learned_probs):
            G_learned[edge[0]][edge[1]]['prob'] = float(prob)

        graphs['Learned Parameters'] = G_learned
        print("Learned parameters ready")

        # Save learned graph
        nx.write_edgelist(G_learned,
                         os.path.join(args.output_dir, 'network_learned.edgelist'),
                         data=['prob'])
    else:
        print("\n[2/5] Skipping parameter learning")

    # Run experiments
    print(f"\n[3/5] Running experiments...")
    print(f"Algorithms: {args.algorithms}")
    print(f"K values: {args.k_values}")
    print(f"Runs per config: {args.num_runs}")

    all_results = []

    for k in args.k_values:
        print(f"\n{'='*60}")
        print(f"K = {k}")
        print(f"{'='*60}")

        for param_name, G in graphs.items():
            print(f"\n--- {param_name} ---")

            for algo_name in args.algorithms:
                print(f"\n  Algorithm: {algo_name}")

                influences = []
                runtimes = []

                for run in range(args.num_runs):
                    seeds, influence, runtime = run_algorithm(
                        algo_name, G, args.diffusion_model, k,
                        args.num_simulations, args.tim_epsilon,
                        args.seed + run
                    )

                    influences.append(influence)
                    runtimes.append(runtime)

                    print(f"    Run {run+1}: Influence={influence:.2f}, Time={runtime:.2f}s")

                result_entry = {
                    'k': k,
                    'param_type': param_name,
                    'algorithm': algo_name,
                    'mean_influence': np.mean(influences),
                    'std_influence': np.std(influences),
                    'mean_runtime': np.mean(runtimes),
                    'std_runtime': np.std(runtimes)
                }

                all_results.append(result_entry)

                print(f"    Average: Influence={result_entry['mean_influence']:.2f}±"
                      f"{result_entry['std_influence']:.2f}, "
                      f"Time={result_entry['mean_runtime']:.2f}s")

    # Save results
    print(f"\n[4/5] Saving results...")

    # Save as CSV
    csv_path = os.path.join(args.output_dir, 'comparison_results.csv')
    save_csv(all_results, csv_path)

    # Save as JSON
    results_summary = {
        'config': vars(args),
        'network_stats': {
            'nodes': G_true.number_of_nodes(),
            'edges': G_true.number_of_edges()
        },
        'results': all_results
    }

    json_path = os.path.join(args.output_dir, 'comparison_results.json')
    save_json(results_summary, json_path)

    # Generate plots
    print(f"\n[5/5] Generating plots...")

    for k in args.k_values:
        # Filter results for this k
        k_results = [r for r in all_results if r['k'] == k]

        # Group by param type and algorithm
        plot_data = {}
        runtime_data = {}

        for r in k_results:
            key = f"{r['param_type']}-{r['algorithm']}"
            plot_data[key] = [r['mean_influence']]  # Single value, but plot expects list
            runtime_data[key] = r['mean_runtime']

        # Plot influence comparison
        if len(plot_data) > 0:
            influence_plot_path = os.path.join(args.output_dir,
                                              f'influence_comparison_k{k}.png')
            # Convert single values to lists for plotting
            plot_data_list = {k: [v[0]] for k, v in plot_data.items()}
            plot_influence_comparison(plot_data_list, save_path=influence_plot_path)

            # Plot runtime comparison
            runtime_plot_path = os.path.join(args.output_dir,
                                            f'runtime_comparison_k{k}.png')
            plot_runtime_comparison(runtime_data, save_path=runtime_plot_path)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to {args.output_dir}")
    print(f"- comparison_results.csv: Detailed results in CSV format")
    print(f"- comparison_results.json: Complete results with configuration")
    print(f"- influence_comparison_k*.png: Influence comparison plots")
    print(f"- runtime_comparison_k*.png: Runtime comparison plots")


if __name__ == '__main__':
    main()
