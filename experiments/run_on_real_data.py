"""
Run experiments on real-world datasets.
Download, preprocess, and run influence maximization on real social networks.
"""

import argparse
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import DatasetLoader
from src.diffusion import DiffusionSimulator
from src.influence_max import LazyGreedyIM, TIM, IMM
from src.influence_max.heuristics import (DegreeHeuristic, PageRankHeuristic,
                                         BetweennessHeuristic)
from src.utils import save_json, save_csv, plot_influence_comparison, plot_runtime_comparison


def main():
    parser = argparse.ArgumentParser(description='Run IM on real datasets')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='wiki-vote',
                       choices=['wiki-vote', 'email-enron', 'facebook', 'gplus'],
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str, default='data/real_networks',
                       help='Data directory')
    parser.add_argument('--prob-method', type=str, default='wc',
                       choices=['const', 'wc', 'trivalency'],
                       help='Method to assign IC probabilities')
    parser.add_argument('--const-prob', type=float, default=0.1,
                       help='Constant probability (if using const method)')

    # IM parameters
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['degree', 'pagerank', 'lazy_greedy', 'tim'],
                       help='Algorithms to run')
    parser.add_argument('--k', type=int, default=50,
                       help='Number of seeds')
    parser.add_argument('--num-simulations', type=int, default=1000,
                       help='Number of MC simulations')
    parser.add_argument('--tim-epsilon', type=float, default=0.5,
                       help='Epsilon for TIM/IMM')

    # General
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs/real_datasets',
                       help='Output directory')
    parser.add_argument('--max-nodes', type=int, default=10000,
                       help='Maximum number of nodes (for large graphs)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"INFLUENCE MAXIMIZATION ON REAL DATASET: {args.dataset.upper()}")
    print("=" * 80)

    # Load dataset
    print(f"\n[1/4] Loading dataset: {args.dataset}")
    loader = DatasetLoader(data_dir=args.data_dir)

    # Show available datasets
    print("\nAvailable datasets:")
    loader.list_datasets()

    # Load graph
    G = loader.load_dataset(args.dataset, download=True)

    # Preprocess
    print(f"\n[2/4] Preprocessing graph...")
    G = loader.preprocess_graph(G, largest_cc=True, relabel_nodes=True)

    # Check if graph is too large
    if G.number_of_nodes() > args.max_nodes:
        print(f"\nWARNING: Graph has {G.number_of_nodes()} nodes, which exceeds max_nodes={args.max_nodes}")
        print("Consider using a smaller dataset or increasing --max-nodes")
        print("For demo purposes, we'll sample a subgraph...")

        # Sample a subgraph
        nodes = list(G.nodes())[:args.max_nodes]
        G = G.subgraph(nodes).copy()
        G = loader.preprocess_graph(G, largest_cc=True, relabel_nodes=True)
        print(f"Sampled graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Add IC probabilities
    G = loader.add_ic_probabilities(G, method=args.prob_method,
                                    const_prob=args.const_prob)

    # Save preprocessed graph
    graph_path = os.path.join(args.output_dir, f'{args.dataset}_preprocessed.edgelist')
    loader.save_graph(G, graph_path)

    print(f"\nFinal graph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Avg degree: {np.mean([d for n, d in G.degree()]):.2f}")

    # Run algorithms
    print(f"\n[3/4] Running influence maximization algorithms...")
    print(f"k = {args.k}, simulations = {args.num_simulations}")

    results = []
    sim = DiffusionSimulator(G, model='ic', seed=args.seed)

    for algo_name in args.algorithms:
        print(f"\n--- {algo_name.upper()} ---")

        start_time = time.time()

        try:
            if algo_name == 'degree':
                algo = DegreeHeuristic(G, seed=args.seed)
                seeds, runtime = algo.select_seeds(args.k)

            elif algo_name == 'pagerank':
                algo = PageRankHeuristic(G, seed=args.seed)
                seeds, runtime = algo.select_seeds(args.k)

            elif algo_name == 'betweenness':
                algo = BetweennessHeuristic(G, seed=args.seed)
                seeds, runtime = algo.select_seeds(args.k)

            elif algo_name == 'lazy_greedy':
                algo = LazyGreedyIM(G, sim, seed=args.seed)
                seeds, _, runtime = algo.select_seeds(args.k, args.num_simulations,
                                                     verbose=False)

            elif algo_name == 'tim':
                algo = TIM(G, model='ic', seed=args.seed)
                seeds, _, runtime = algo.select_seeds(args.k, args.tim_epsilon,
                                                     verbose=False)

            elif algo_name == 'imm':
                algo = IMM(G, model='ic', seed=args.seed)
                seeds, _, runtime = algo.select_seeds(args.k, args.tim_epsilon,
                                                     verbose=False)

            else:
                print(f"Unknown algorithm: {algo_name}, skipping...")
                continue

            # Estimate influence
            print(f"Estimating influence...")
            influence = sim.estimate_influence(seeds, args.num_simulations)

            result = {
                'algorithm': algo_name,
                'k': args.k,
                'seeds': seeds,
                'influence': influence,
                'runtime': runtime,
                'influence_per_seed': influence / args.k
            }

            results.append(result)

            print(f"Seeds: {seeds[:10]}..." if len(seeds) > 10 else f"Seeds: {seeds}")
            print(f"Influence: {influence:.2f}")
            print(f"Runtime: {runtime:.2f}s")

        except Exception as e:
            print(f"ERROR running {algo_name}: {e}")
            continue

    # Save results
    print(f"\n[4/4] Saving results...")

    # Summary
    summary = {
        'dataset': args.dataset,
        'graph_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
        },
        'config': vars(args),
        'results': results
    }

    json_path = os.path.join(args.output_dir, f'{args.dataset}_results.json')
    save_json(summary, json_path)

    # CSV for easier analysis
    csv_results = []
    for r in results:
        csv_results.append({
            'dataset': args.dataset,
            'algorithm': r['algorithm'],
            'k': r['k'],
            'influence': r['influence'],
            'runtime': r['runtime'],
            'influence_per_seed': r['influence_per_seed']
        })

    csv_path = os.path.join(args.output_dir, f'{args.dataset}_results.csv')
    save_csv(csv_results, csv_path)

    # Plots
    if len(results) > 0:
        # Influence comparison
        plot_data = {r['algorithm']: [r['influence']] for r in results}
        influence_plot = os.path.join(args.output_dir,
                                     f'{args.dataset}_influence_comparison.png')
        plot_influence_comparison(plot_data, save_path=influence_plot)

        # Runtime comparison
        runtime_data = {r['algorithm']: r['runtime'] for r in results}
        runtime_plot = os.path.join(args.output_dir,
                                   f'{args.dataset}_runtime_comparison.png')
        plot_runtime_comparison(runtime_data, save_path=runtime_plot)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"\nResults summary:")
    for r in results:
        print(f"\n{r['algorithm'].upper()}:")
        print(f"  Influence: {r['influence']:.2f}")
        print(f"  Runtime: {r['runtime']:.2f}s")
        print(f"  Influence/Seed: {r['influence_per_seed']:.2f}")

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
