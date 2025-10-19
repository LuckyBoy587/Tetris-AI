"""
Quick example demonstrating the Genetic Algorithm agent usage.
This script trains a small agent and then plays a game.
"""

from genetic_agent import GeneticAgent, TetrisChromosome
from tetris_env import TetrisEnv


def quick_demo():
    """Run a quick demo of the genetic algorithm."""
    
    print("="*60)
    print("GENETIC ALGORITHM TETRIS AGENT - QUICK DEMO")
    print("="*60)
    
    # Create a small agent for quick demonstration
    print("\n1. Creating Genetic Algorithm agent...")
    agent = GeneticAgent(
        population_size=10,      # Small population for quick demo
        mutation_rate=0.15,
        mutation_strength=0.3,
        crossover_rate=0.8,
        elitism_count=1,
        tournament_size=3
    )
    
    print(f"   Population size: {agent.population_size}")
    print(f"   Mutation rate: {agent.mutation_rate}")
    print(f"   Crossover rate: {agent.crossover_rate}")
    
    # Train for a few generations
    print("\n2. Training for 5 generations...")
    print("   (This may take a few minutes)")
    agent.evolve(generations=5, games_per_eval=2, verbose=True)
    
    # Show results
    print("\n3. Training Complete!")
    print(f"   Best fitness: {agent.best_chromosome.fitness:.2f}")
    print(f"   Best genes: {agent.best_chromosome.genes}")
    
    # Save the agent
    print("\n4. Saving agent to 'demo_agent.pkl'...")
    agent.save('demo_agent.pkl')
    
    # Play a demonstration game
    print("\n5. Playing a demonstration game...")
    print("   Close the Pygame window to end the game early.")
    info = agent.play_game(render=True, max_pieces=200)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"Final Score: {info.get('score', 0)}")
    print(f"Lines Cleared: {info.get('lines_cleared', 0)}")
    print(f"Pieces Placed: {info.get('total_pieces', 0)}")
    print("\nTo continue training, run:")
    print("  python genetic_agent.py --mode continue --file demo_agent.pkl --generations 15")
    print("\nTo visualize training:")
    print("  python visualize_training.py --file demo_agent.pkl")


def test_manual_chromosome():
    """Test a manually created chromosome with specific weights."""
    
    print("="*60)
    print("TESTING MANUAL CHROMOSOME")
    print("="*60)
    
    # Create a chromosome with hand-tuned weights
    # These are reasonable starting values based on Tetris strategy
    manual_genes = np.array([
        -0.5,   # aggregate_height (penalty)
        -0.7,   # holes (strong penalty)
        -0.3,   # bumpiness (moderate penalty)
        0.8     # completed_lines (reward)
    ])
    
    chromosome = TetrisChromosome(manual_genes)
    
    print(f"\nManual Chromosome: {chromosome}")
    
    # Create agent and set the chromosome
    agent = GeneticAgent()
    agent.best_chromosome = chromosome
    
    print("\nPlaying game with manual chromosome...")
    info = agent.play_game(render=True, max_pieces=200)
    
    print("\n" + "="*60)
    print(f"Score: {info.get('score', 0)}")
    print(f"Lines: {info.get('lines_cleared', 0)}")
    print("="*60)


if __name__ == "__main__":
    import sys
    import numpy as np
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        test_manual_chromosome()
    else:
        quick_demo()
