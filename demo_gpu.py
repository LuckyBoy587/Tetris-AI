"""
GPU-Optimized Quick Demo
Demonstrates fast training using parallel processing and GPU acceleration.
"""

from genetic_agent_gpu import GeneticAgentGPU, TetrisChromosome
from tetris_env import TetrisEnv
import numpy as np


def quick_demo_gpu():
    """Run a quick demo of the GPU-optimized genetic algorithm."""
    
    print("="*70)
    print("GPU-OPTIMIZED GENETIC ALGORITHM TETRIS AGENT - QUICK DEMO")
    print("="*70)
    
    # Create a GPU-optimized agent with larger population
    print("\n🚀 Creating GPU-Optimized Genetic Algorithm agent...")
    agent = GeneticAgentGPU(
        population_size=100,     # Larger population for better parallel performance
        mutation_rate=0.15,
        mutation_strength=0.3,
        crossover_rate=0.8,
        elitism_count=4,         # Keep top 4 performers
        tournament_size=5,
        n_workers=None           # Use all available CPU cores
    )
    
    print(f"   Population size: {agent.population_size}")
    print(f"   Parallel workers: {agent.n_workers}")
    print(f"   Mutation rate: {agent.mutation_rate}")
    print(f"   Crossover rate: {agent.crossover_rate}")
    
    # Train for generations - much faster with parallel processing!
    print("\n⚡ Training for 10 generations with parallel processing...")
    print("   (This will be MUCH faster than the standard version!)")
    agent.evolve(
        generations=10,
        games_per_eval=5,        # More games for better evaluation
        max_pieces=500,          # Longer games for better fitness assessment
        verbose=True
    )
    
    # Show results
    print("\n✓ Training Complete!")
    print(f"   Best fitness: {agent.best_chromosome.fitness:.2f}")
    print(f"   Best genes: {agent.best_chromosome.genes}")
    
    # Save the agent
    print("\n💾 Saving agent to 'demo_agent_gpu.pkl'...")
    agent.save('demo_agent_gpu.pkl')
    
    # Play a demonstration game
    print("\n🎮 Playing a demonstration game...")
    print("   Close the Pygame window to end the game early.")
    info = agent.play_game(render=True, max_pieces=500)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print(f"Final Score: {info.get('score', 0)}")
    print(f"Lines Cleared: {info.get('lines_cleared', 0)}")
    print(f"Pieces Placed: {info.get('total_pieces', 0)}")
    print("\n📝 Next steps:")
    print("  1. Continue training with more generations:")
    print("     python genetic_agent_gpu.py --mode continue --file demo_agent_gpu.pkl --generations 50 --population 200")
    print("\n  2. Train with larger population for better results:")
    print("     python genetic_agent_gpu.py --mode train --population 200 --generations 30 --games 10")
    print("\n  3. Visualize training progress:")
    print("     python visualize_training.py --file demo_agent_gpu.pkl")


def intensive_training():
    """
    Run intensive training optimized for GPU/multi-core systems.
    This is what you should run on P100 for best results!
    """
    
    print("="*70)
    print("🔥 INTENSIVE GPU-OPTIMIZED TRAINING 🔥")
    print("="*70)
    print("\nThis configuration is optimized for maximum performance on P100 GPU")
    print("and multi-core systems. It will train a large population for many")
    print("generations to find the best Tetris playing strategy.\n")
    
    # Create a large-scale agent
    print("🚀 Creating large-scale genetic algorithm agent...")
    agent = GeneticAgentGPU(
        population_size=200,      # Large population
        mutation_rate=0.12,
        mutation_strength=0.25,
        crossover_rate=0.85,
        elitism_count=8,          # Keep top 8 performers
        tournament_size=7,
        n_workers=None            # Use all available CPU cores
    )
    
    print(f"   Population size: {agent.population_size}")
    print(f"   Parallel workers: {agent.n_workers}")
    
    # Intensive training
    print("\n⚡ Starting intensive training session...")
    print("   This will utilize all CPU cores for maximum speed!")
    
    agent.evolve(
        generations=50,           # Many generations
        games_per_eval=10,        # More games per evaluation for accuracy
        max_pieces=800,           # Longer games
        verbose=True
    )
    
    # Save results
    print("\n💾 Saving trained agent...")
    agent.save('tetris_agent_intensive.pkl')
    
    print("\n" + "="*70)
    print("✓ INTENSIVE TRAINING COMPLETE!")
    print("="*70)
    print(f"Best fitness achieved: {agent.best_chromosome.fitness:.2f}")
    print(f"Optimal genes: {agent.best_chromosome.genes}")
    print("\nAgent saved to: tetris_agent_intensive.pkl")
    
    # Optionally play a demo
    play_demo = input("\n🎮 Play a demonstration game? (y/n): ")
    if play_demo.lower() == 'y':
        agent.play_game(render=True, max_pieces=1000)


def test_manual_chromosome():
    """Test a manually created chromosome with specific weights."""
    
    print("="*70)
    print("TESTING MANUAL CHROMOSOME")
    print("="*70)
    
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
    agent = GeneticAgentGPU()
    agent.best_chromosome = chromosome
    
    print("\n🎮 Playing game with manual chromosome...")
    info = agent.play_game(render=True, max_pieces=500)
    
    print("\n" + "="*70)
    print(f"Score: {info.get('score', 0)}")
    print(f"Lines: {info.get('lines_cleared', 0)}")
    print("="*70)


def benchmark_performance():
    """
    Benchmark the performance of the GPU-optimized version.
    """
    import time
    
    print("="*70)
    print("⏱️  PERFORMANCE BENCHMARK")
    print("="*70)
    
    print("\nTesting parallel evaluation speed...")
    
    agent = GeneticAgentGPU(
        population_size=100,
        n_workers=None
    )
    
    print(f"Population size: {agent.population_size}")
    print(f"Parallel workers: {agent.n_workers}")
    
    # Time one generation
    print("\nRunning 1 generation benchmark...")
    start_time = time.time()
    
    agent.evolve(
        generations=1,
        games_per_eval=3,
        max_pieces=500,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    total_games = agent.population_size * 3
    games_per_sec = total_games / elapsed
    
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    print(f"Total games evaluated: {total_games}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Speed: {games_per_sec:.1f} games/second")
    print(f"Average time per game: {elapsed/total_games:.3f} seconds")
    print("\n💡 Tip: With more CPU cores and GPU acceleration, you can")
    print("   process hundreds or thousands of games per second!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "--intensive":
            intensive_training()
        elif mode == "--manual":
            test_manual_chromosome()
        elif mode == "--benchmark":
            benchmark_performance()
        else:
            print("Usage:")
            print("  python demo_gpu.py              # Run quick demo")
            print("  python demo_gpu.py --intensive  # Run intensive training")
            print("  python demo_gpu.py --manual     # Test manual chromosome")
            print("  python demo_gpu.py --benchmark  # Benchmark performance")
    else:
        quick_demo_gpu()
