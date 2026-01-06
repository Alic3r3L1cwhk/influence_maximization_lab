"""
Train diffusion parameters from cascade data.
This script generates a network, creates cascades, learns edge probabilities using PyTorch.
"""

import argparse
import os
import sys
import numpy as np
import warnings
import networkx as nx

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import NetworkGenerator, CascadeGenerator, DataSplitter
from src.models import GraphEmbedding, ParameterLearner
from src.utils import plot_training_history, save_json, save_experiment_results


def main():
    parser = argparse.ArgumentParser(description='Train diffusion parameters from cascades')

    # Network parameters
    parser.add_argument('--network-type', type=str, default='ba',
                       choices=['er', 'ba', 'ws'],
                       help='Type of network to generate')
    parser.add_argument('--num-nodes', type=int, default=500,
                       help='Number of nodes in the network')
    parser.add_argument('--ba-m', type=int, default=3,
                       help='BA model: number of edges to attach')
    parser.add_argument('--er-p', type=float, default=0.01,
                       help='ER model: edge probability')
    parser.add_argument('--ws-k', type=int, default=4,
                       help='WS model: number of nearest neighbors')
    parser.add_argument('--ws-p', type=float, default=0.1,
                       help='WS model: rewiring probability')

    # Cascade parameters
    parser.add_argument('--diffusion-model', type=str, default='ic',
                       choices=['ic', 'lt'],
                       help='Diffusion model for cascade generation')
    parser.add_argument('--num-cascades', type=int, default=1000,
                       help='Number of cascades to generate')
    parser.add_argument('--prob-range', type=float, nargs=2, default=[0.01, 0.1],
                       help='Range for IC edge probabilities')

    # Embedding parameters
    parser.add_argument('--embedding-method', type=str, default='node2vec',
                       choices=['node2vec', 'deepwalk'],
                       help='Graph embedding method')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--num-walks', type=int, default=10,
                       help='Number of random walks per node')
    parser.add_argument('--walk-length', type=int, default=80,
                       help='Length of each random walk')
    parser.add_argument('--node2vec-p', type=float, default=1.0,
                       help='Node2Vec return parameter')
    parser.add_argument('--node2vec-q', type=float, default=1.0,
                       help='Node2Vec in-out parameter')

    # Learning parameters
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                       help='Hidden layer dimensions for MLP')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')

    # Data split
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')

    # General parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs/trained_models',
                       help='Output directory')
    parser.add_argument('--save-graph', action='store_true',
                       help='Save the generated graph')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PARAMETER LEARNING FROM CASCADE DATA")
    print("=" * 80)

    # Step 1: Generate network
    print(f"\n[1/6] Generating {args.network_type.upper()} network with {args.num_nodes} nodes...")
    gen = NetworkGenerator(seed=args.seed)

    if args.network_type == 'ba':
        G = gen.generate_ba(args.num_nodes, args.ba_m)
    elif args.network_type == 'er':
        G = gen.generate_er(args.num_nodes, args.er_p)
    elif args.network_type == 'ws':
        G = gen.generate_ws(args.num_nodes, args.ws_k, args.ws_p)

    print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Assign true probabilities
    if args.diffusion_model == 'ic':
        G = gen.assign_ic_probabilities(G, prob_range=tuple(args.prob_range))
    else:
        G = gen.assign_lt_thresholds(G)

    # Save graph if requested
    if args.save_graph:
        graph_path = os.path.join(args.output_dir, 'network.edgelist')
        nx.write_edgelist(G, graph_path, data=['prob'] if args.diffusion_model == 'ic' else ['weight'])
        print(f"Graph saved to {graph_path}")

    # Step 2: Generate cascades
    print(f"\n[2/6] Generating {args.num_cascades} cascades using {args.diffusion_model.upper()} model...")
    cascade_gen = CascadeGenerator(G, seed=args.seed)
    cascades = cascade_gen.generate_cascades(args.diffusion_model, args.num_cascades)

    avg_cascade_size = np.mean([c['cascade_size'] for c in cascades])
    print(f"Generated {len(cascades)} cascades, average size: {avg_cascade_size:.2f}")

    # Step 3: Convert cascades to training data
    print(f"\n[3/6] Converting cascades to edge training data...")
    edges, labels = cascade_gen.cascades_to_training_data(cascades)
    print(f"Created {len(edges)} training examples")
    print(f"Positive rate: {np.mean(labels):.3f}")

    # Split data
    splitter = DataSplitter(seed=args.seed)
    train_edges, train_labels, val_edges, val_labels, test_edges, test_labels = \
        splitter.split_edges(edges, labels, args.train_ratio, args.val_ratio, args.test_ratio)

    print(f"Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")

    # Step 4: Generate embeddings
    print(f"\n[4/6] Generating {args.embedding_method} embeddings (dim={args.embedding_dim})...")
    embedding_gen = GraphEmbedding(G, embedding_dim=args.embedding_dim, seed=args.seed)

    if args.embedding_method == 'node2vec':
        embeddings = embedding_gen.train_node2vec(
            num_walks=args.num_walks,
            walk_length=args.walk_length,
            p=args.node2vec_p,
            q=args.node2vec_q
        )
    else:
        embeddings = embedding_gen.train_deepwalk(
            num_walks=args.num_walks,
            walk_length=args.walk_length
        )

    # Save embeddings
    emb_path = os.path.join(args.output_dir, 'embeddings.txt')
    embedding_gen.save_embeddings(emb_path)

    # Convert edges to features
    print("Converting edges to feature vectors...")
    train_features = np.array([embedding_gen.get_edge_features(e) for e in train_edges])
    val_features = np.array([embedding_gen.get_edge_features(e) for e in val_edges])
    test_features = np.array([embedding_gen.get_edge_features(e) for e in test_edges])
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    # Step 5: Train model
    print(f"\n[5/6] Training MLP model on {args.device}...")
    input_dim = train_features.shape[1]

    learner = ParameterLearner(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed
    )

    history = learner.fit(
        train_features, train_labels,
        val_features, val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping,
        verbose=True
    )

    # Step 6: Evaluate
    print(f"\n[6/6] Evaluating on test set...")
    test_predictions = learner.predict(test_features)

    from sklearn.metrics import roc_auc_score, accuracy_score
    binary_test_labels = (np.array(test_labels) > 0).astype(int)

    if binary_test_labels.max() == binary_test_labels.min():
        warnings.warn("Test set has only one class; AUC is undefined. Returning NaN.")
        test_auc = float('nan')
    else:
        test_auc = roc_auc_score(binary_test_labels, test_predictions)

    test_acc = accuracy_score(binary_test_labels, (test_predictions > 0.5).astype(int))

    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    model_path = os.path.join(args.output_dir, 'param_learner.pth')
    learner.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Save training history plot
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)

    # Save experiment configuration and results
    results = {
        'config': vars(args),
        'network_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        },
        'cascade_stats': {
            'num_cascades': len(cascades),
            'avg_cascade_size': float(avg_cascade_size)
        },
        'training_stats': {
            'train_size': len(train_edges),
            'val_size': len(val_edges),
            'test_size': len(test_edges),
            'final_train_loss': history['train_loss'][-1],
            'final_train_auc': history['train_auc'][-1],
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'final_val_auc': history['val_auc'][-1] if history['val_auc'] else None,
            'test_auc': float(test_auc),
            'test_accuracy': float(test_acc)
        }
    }

    results_path = os.path.join(args.output_dir, 'training_results.json')
    save_json(results, results_path)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"All results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
