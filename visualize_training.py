"""
Visualization tools for Genetic Algorithm training progress.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def plot_training_history(filename: str = "genetic_agent.pkl", save_plot: bool = True):
    """
    Plot the training history from a saved agent.
    
    Args:
        filename: File containing the saved agent
        save_plot: Whether to save the plot to a file
    """
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return
    
    # Load agent data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    history = data['history']
    best_chromosome = data['best_chromosome']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Genetic Algorithm Training Progress', fontsize=16, fontweight='bold')
    
    generations = range(1, len(history['best_fitness']) + 1)
    
    # Plot 1: Best and Average Fitness
    ax1 = axes[0, 0]
    ax1.plot(generations, history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(generations, history['avg_fitness'], 'r--', linewidth=2, label='Average Fitness')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title('Fitness Over Generations', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitness Improvement
    ax2 = axes[0, 1]
    improvements = [0] + [history['best_fitness'][i] - history['best_fitness'][i-1] 
                          for i in range(1, len(history['best_fitness']))]
    ax2.bar(generations, improvements, color='green', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Fitness Improvement', fontsize=12)
    ax2.set_title('Generation-to-Generation Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Gene Evolution
    ax3 = axes[1, 0]
    gene_names = ['Aggregate Height', 'Holes', 'Bumpiness', 'Completed Lines']
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (name, color) in enumerate(zip(gene_names, colors)):
        gene_values = [genes[i] for genes in history['best_genes']]
        ax3.plot(generations, gene_values, color=color, linewidth=2, 
                label=name, marker='o', markersize=3)
    
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Gene Weight', fontsize=12)
    ax3.set_title('Gene Weight Evolution', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 4: Final Gene Weights
    ax4 = axes[1, 1]
    final_genes = best_chromosome.genes
    bars = ax4.barh(gene_names, final_genes, color=colors, alpha=0.7)
    ax4.set_xlabel('Weight Value', fontsize=12)
    ax4.set_title('Best Chromosome Gene Weights', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_genes):
        width = bar.get_width()
        label_x_pos = width + 0.02 if width > 0 else width - 0.02
        ax4.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                ha='left' if width > 0 else 'right', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        plot_filename = filename.replace('.pkl', '_training_plot.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Generations: {len(history['best_fitness'])}")
    print(f"Initial Best Fitness: {history['best_fitness'][0]:.2f}")
    print(f"Final Best Fitness: {history['best_fitness'][-1]:.2f}")
    print(f"Total Improvement: {history['best_fitness'][-1] - history['best_fitness'][0]:.2f}")
    print(f"Best Fitness Ever: {max(history['best_fitness']):.2f}")
    print(f"\nFinal Average Fitness: {history['avg_fitness'][-1]:.2f}")
    print(f"\nBest Chromosome Genes:")
    for name, value in zip(gene_names, final_genes):
        print(f"  {name:20s}: {value:7.3f}")
    print("="*60)


def compare_agents(filenames: list, labels: list = None):
    """
    Compare multiple trained agents.
    
    Args:
        filenames: List of agent files to compare
        labels: Custom labels for each agent
    """
    if labels is None:
        labels = [f"Agent {i+1}" for i in range(len(filenames))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Agent Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(filenames)))
    
    # Plot 1: Best Fitness Comparison
    ax1 = axes[0]
    for filename, label, color in zip(filenames, labels, colors):
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        history = data['history']
        generations = range(1, len(history['best_fitness']) + 1)
        ax1.plot(generations, history['best_fitness'], linewidth=2, 
                label=label, color=color)
    
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Best Fitness Over Generations', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gene Weights Comparison
    ax2 = axes[1]
    gene_names = ['Height', 'Holes', 'Bumpy', 'Lines']
    x = np.arange(len(gene_names))
    width = 0.8 / len(filenames)
    
    for i, (filename, label, color) in enumerate(zip(filenames, labels, colors)):
        if not os.path.exists(filename):
            continue
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        genes = data['best_chromosome'].genes
        offset = (i - len(filenames)/2) * width + width/2
        ax2.bar(x + offset, genes, width, label=label, color=color, alpha=0.7)
    
    ax2.set_xlabel('Gene', fontsize=12)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.set_title('Final Gene Weights Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gene_names)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Genetic Algorithm training')
    parser.add_argument('--file', type=str, default='genetic_agent.pkl',
                        help='Agent file to visualize')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='Multiple agent files to compare')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plot to file')
    
    args = parser.parse_args()
    
    if args.compare:
        print("Comparing multiple agents...")
        compare_agents(args.compare)
    else:
        print(f"Visualizing training history from {args.file}...")
        plot_training_history(args.file, save_plot=not args.no_save)
