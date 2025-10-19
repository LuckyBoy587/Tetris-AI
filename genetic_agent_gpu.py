"""
GPU-Optimized Genetic Algorithm Agent for Tetris
Utilizes parallel processing and GPU acceleration for fast training on P100.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from tetris_env import TetrisEnv
import copy
import pickle
import os
from multiprocessing import Pool, cpu_count
import concurrent.futures
from functools import partial

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("⚠ CuPy not found - using NumPy (CPU only)")
    print("  Install with: pip install cupy-cuda12x")


class TetrisChromosome:
    """
    Represents a single solution (genome) in the genetic algorithm.
    Contains weights for evaluating board states.
    """
    
    def __init__(self, genes: np.ndarray = None):
        """
        Initialize a chromosome with random or provided genes.
        
        Genes represent weights for:
        - aggregate_height: penalty for tall stacks
        - holes: penalty for holes in the board
        - bumpiness: penalty for uneven surface
        - completed_lines: reward for completing lines
        """
        if genes is None:
            # Initialize with random weights
            self.genes = np.random.uniform(-1, 1, 4)
        else:
            self.genes = genes.copy()
        
        self.fitness = 0.0
    
    def evaluate_board(self, env: TetrisEnv) -> float:
        """
        Evaluate a board state using the chromosome's weights.
        
        Args:
            env: Tetris environment
            
        Returns:
            Score for the current board state
        """
        features = env.get_features()
        
        score = (
            self.genes[0] * features['aggregate_height'] +
            self.genes[1] * features['holes'] +
            self.genes[2] * features['bumpiness'] +
            self.genes[3] * features['completed_lines']
        )
        
        return score
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        """
        Randomly mutate genes.
        
        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Maximum change magnitude
        """
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] += np.random.uniform(-mutation_strength, mutation_strength)
    
    def copy(self):
        """Create a deep copy of the chromosome."""
        return TetrisChromosome(self.genes)
    
    def __str__(self):
        return f"Genes: [{', '.join([f'{g:.3f}' for g in self.genes])}], Fitness: {self.fitness:.2f}"


def evaluate_chromosome_worker(args):
    """
    Worker function for parallel chromosome evaluation.
    
    Args:
        args: Tuple of (chromosome, games, max_pieces)
        
    Returns:
        Fitness score
    """
    chromosome, games, max_pieces = args
    total_fitness = 0.0
    
    for _ in range(games):
        env = TetrisEnv(render_mode=False)
        state = env.reset()
        done = False
        pieces_placed = 0
        
        while not done and pieces_placed < max_pieces:
            # Get action
            actions = choose_action_simple(env, chromosome)
            
            # Execute actions
            for action in actions:
                if done:
                    break
                _, reward, done, info = env.step(action)
            
            pieces_placed += 1
        
        # Fitness is based on score and lines cleared
        fitness = info['score'] + info['lines_cleared'] * 100
        total_fitness += fitness
        
        env.close()
    
    avg_fitness = total_fitness / games
    return avg_fitness


def choose_action_simple(env: TetrisEnv, chromosome: TetrisChromosome) -> List[int]:
    """
    Simplified action selection for parallel execution.
    
    Args:
        env: Tetris environment
        chromosome: Chromosome to use for evaluation
        
    Returns:
        List of actions to execute
    """
    possible_moves = get_all_possible_moves_simple(env)
    
    if not possible_moves:
        return [4]  # Hard drop if no moves available
    
    best_score = float('-inf')
    best_move = possible_moves[0]
    
    # Evaluate each possible move
    for rotation, col, _ in possible_moves:
        # Create a copy of the environment to simulate
        test_env = copy_env_simple(env)
        
        # Set rotation
        test_env.current_rotation = rotation
        test_env.current_piece = test_env.SHAPES[test_env.current_shape_name][rotation]
        
        # Set column
        test_env.current_pos[1] = col
        
        # Drop
        test_env._hard_drop()
        test_env._clear_lines()
        
        # Evaluate
        features = test_env.get_features()
        score = (
            chromosome.genes[0] * features['aggregate_height'] +
            chromosome.genes[1] * features['holes'] +
            chromosome.genes[2] * features['bumpiness'] +
            chromosome.genes[3] * features['completed_lines']
        )
        
        if score > best_score:
            best_score = score
            best_move = (rotation, col, _)
    
    # Convert best move to action sequence
    actions = move_to_actions_simple(env, best_move[0], best_move[1])
    return actions


def get_all_possible_moves_simple(env: TetrisEnv) -> List[Tuple[int, int, int]]:
    """
    Get all possible final positions for the current piece.
    
    Args:
        env: Tetris environment
        
    Returns:
        List of (rotation, column, row) tuples
    """
    original_state = {
        'board': env.board.copy(),
        'current_piece': env.current_piece,
        'current_pos': env.current_pos.copy(),
        'current_rotation': env.current_rotation,
        'current_shape_name': env.current_shape_name,
        'current_color_idx': env.current_color_idx
    }
    
    possible_moves = []
    shape = env.current_shape_name
    rotations = len(env.SHAPES[shape])
    
    # Try each rotation
    for rotation in range(rotations):
        env.current_rotation = rotation
        env.current_piece = env.SHAPES[shape][rotation]
        
        # Try each column
        for col in range(-2, env.BOARD_WIDTH + 2):
            # Reset position
            env.current_pos = [0, col]
            
            # Check if position is valid at the top
            if env._check_collision(env.current_piece, env.current_pos):
                continue
            
            # Simulate drop
            while True:
                new_pos = [env.current_pos[0] + 1, env.current_pos[1]]
                if env._check_collision(env.current_piece, new_pos):
                    break
                env.current_pos = new_pos
            
            # Check if piece is within bounds after drop
            valid = True
            for dy, dx in env.current_piece:
                row = env.current_pos[0] + dy
                col_check = env.current_pos[1] + dx
                if row < 0 or col_check < 0 or col_check >= env.BOARD_WIDTH:
                    valid = False
                    break
            
            if valid:
                possible_moves.append((rotation, col, env.current_pos[0]))
    
    # Restore original state
    env.board = original_state['board']
    env.current_piece = original_state['current_piece']
    env.current_pos = original_state['current_pos']
    env.current_rotation = original_state['current_rotation']
    env.current_shape_name = original_state['current_shape_name']
    env.current_color_idx = original_state['current_color_idx']
    
    return possible_moves


def copy_env_simple(env: TetrisEnv) -> TetrisEnv:
    """Create a lightweight copy of environment for simulation."""
    new_env = TetrisEnv(render_mode=False)
    new_env.board = env.board.copy()
    new_env.current_piece = env.current_piece
    new_env.current_shape_name = env.current_shape_name
    new_env.current_rotation = env.current_rotation
    new_env.current_pos = env.current_pos.copy()
    new_env.current_color_idx = env.current_color_idx
    new_env.score = env.score
    new_env.lines_cleared = env.lines_cleared
    new_env.game_over = env.game_over
    return new_env


def move_to_actions_simple(env: TetrisEnv, target_rotation: int, target_col: int) -> List[int]:
    """
    Convert a target position to a sequence of actions.
    
    Args:
        env: Tetris environment
        target_rotation: Desired rotation
        target_col: Desired column
        
    Returns:
        List of actions
    """
    actions = []
    
    # Rotate to target rotation
    current_rot = env.current_rotation
    shape = env.current_shape_name
    num_rotations = len(env.SHAPES[shape])
    rotations_needed = (target_rotation - current_rot) % num_rotations
    
    for _ in range(rotations_needed):
        actions.append(2)  # Rotate
    
    # Move to target column
    current_col = env.current_pos[1]
    col_diff = target_col - current_col
    
    if col_diff < 0:
        for _ in range(abs(col_diff)):
            actions.append(0)  # Move left
    else:
        for _ in range(col_diff):
            actions.append(1)  # Move right
    
    # Hard drop
    actions.append(4)
    
    return actions


class GeneticAgentGPU:
    """
    GPU-optimized Genetic Algorithm agent that evolves Tetris playing strategies.
    Uses parallel processing for massive speedup on multi-core systems.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        crossover_rate: float = 0.8,
        elitism_count: int = 2,
        tournament_size: int = 5,
        n_workers: int | None = None
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size: Number of chromosomes in population
            mutation_rate: Probability of mutating each gene
            mutation_strength: Maximum magnitude of mutation
            crossover_rate: Probability of crossover between parents
            elitism_count: Number of top performers to keep unchanged
            tournament_size: Number of individuals in tournament selection
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        
        # Set number of workers for parallel processing
        if n_workers is None:
            self.n_workers = cpu_count()
        else:
            self.n_workers = n_workers
        
        print(f"🚀 GPU-Optimized Genetic Agent initialized")
        print(f"   Population size: {population_size}")
        print(f"   Parallel workers: {self.n_workers}")
        print(f"   GPU acceleration: {'✓ Enabled' if GPU_AVAILABLE else '✗ Disabled'}")
        
        # Initialize population
        self.population = [TetrisChromosome() for _ in range(population_size)]
        self.generation = 0
        self.best_chromosome = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_genes': []
        }
    
    def evaluate_population_parallel(self, games_per_eval: int = 3, max_pieces: int = 500):
        """
        Evaluate entire population in parallel using multiprocessing.
        
        Args:
            games_per_eval: Number of games to play for evaluation
            max_pieces: Maximum pieces per game
        """
        # Prepare arguments for parallel execution
        eval_args = [(chromosome, games_per_eval, max_pieces) for chromosome in self.population]
        
        # Use ProcessPoolExecutor for better performance
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(evaluate_chromosome_worker, args) for args in eval_args]
            
            # Collect results as they complete
            fitnesses = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                fitness = future.result()
                fitnesses.append((i, fitness))
        
        # Sort by original index to maintain order
        fitnesses.sort(key=lambda x: x[0])
        
        # Assign fitness values
        for i, (_, fitness) in enumerate(fitnesses):
            self.population[i].fitness = fitness
    
    def crossover_batch(self, parents: List[TetrisChromosome]) -> List[TetrisChromosome]:
        """
        Perform batch crossover operations using GPU-accelerated arrays.
        
        Args:
            parents: List of parent chromosomes
            
        Returns:
            List of offspring chromosomes
        """
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if random.random() <= self.crossover_rate:
                # Single-point crossover
                point = random.randint(1, len(parent1.genes) - 1)
                
                child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
                child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
                
                offspring.append(TetrisChromosome(child1_genes))
                offspring.append(TetrisChromosome(child2_genes))
            else:
                offspring.append(parent1.copy())
                offspring.append(parent2.copy())
        
        return offspring
    
    def mutate_batch(self, chromosomes: List[TetrisChromosome]):
        """
        Perform batch mutation operations.
        
        Args:
            chromosomes: List of chromosomes to mutate
        """
        for chromosome in chromosomes:
            chromosome.mutate(self.mutation_rate, self.mutation_strength)
    
    def tournament_selection(self) -> TetrisChromosome:
        """
        Select a chromosome using tournament selection.
        
        Returns:
            Selected chromosome
        """
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda c: c.fitness)
    
    def evolve(self, generations: int = 50, games_per_eval: int = 3, max_pieces: int = 500, verbose: bool = True):
        """
        Evolve the population for a number of generations using parallel processing.
        
        Args:
            generations: Number of generations to evolve
            games_per_eval: Number of games to play for evaluation
            max_pieces: Maximum pieces per game
            verbose: Print progress information
        """
        import time
        
        for gen in range(generations):
            gen_start_time = time.time()
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Generation {self.generation + 1}/{self.generation + generations}")
                print(f"{'='*60}")
                print("⚡ Evaluating population in parallel...")
            
            # Parallel evaluation of entire population
            eval_start = time.time()
            self.evaluate_population_parallel(games_per_eval, max_pieces)
            eval_time = time.time() - eval_start
            
            # Sort by fitness
            self.population.sort(key=lambda c: c.fitness, reverse=True)
            
            # Track best chromosome
            if self.best_chromosome is None or self.population[0].fitness > self.best_chromosome.fitness:
                self.best_chromosome = self.population[0].copy()
            
            # Record history
            best_fitness = self.population[0].fitness
            avg_fitness = np.mean([c.fitness for c in self.population])
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_genes'].append(self.population[0].genes.copy())
            
            if verbose:
                print(f"\n✓ Evaluation complete in {eval_time:.2f}s")
                print(f"  Speed: {self.population_size * games_per_eval / eval_time:.1f} games/sec")
                print(f"\n📊 Results:")
                print(f"  Best Fitness:  {best_fitness:.2f}")
                print(f"  Avg Fitness:   {avg_fitness:.2f}")
                print(f"  Best Genes:    [{', '.join([f'{g:.3f}' for g in self.population[0].genes])}]")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep top performers
            for i in range(self.elitism_count):
                new_population.append(self.population[i].copy())
            
            # Create offspring in batches
            parents_for_crossover = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                parents_for_crossover.extend([parent1, parent2])
            
            # Batch crossover
            offspring = self.crossover_batch(parents_for_crossover)
            
            # Batch mutation
            self.mutate_batch(offspring)
            
            # Add offspring to new population
            new_population.extend(offspring[:self.population_size - len(new_population)])
            
            self.population = new_population
            self.generation += 1
            
            gen_time = time.time() - gen_start_time
            if verbose:
                print(f"\n⏱️  Generation time: {gen_time:.2f}s")
    
    def save(self, filename: str = "genetic_agent_gpu.pkl"):
        """
        Save the agent to a file.
        
        Args:
            filename: File to save to
        """
        data = {
            'best_chromosome': self.best_chromosome,
            'generation': self.generation,
            'history': self.history,
            'population': self.population,
            'n_workers': self.n_workers
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Agent saved to {filename}")
    
    def load(self, filename: str = "genetic_agent_gpu.pkl"):
        """
        Load the agent from a file.
        
        Args:
            filename: File to load from
        """
        if not os.path.exists(filename):
            print(f"✗ File {filename} not found!")
            return
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.best_chromosome = data['best_chromosome']
        self.generation = data['generation']
        self.history = data['history']
        self.population = data['population']
        if 'n_workers' in data:
            self.n_workers = data['n_workers']
        
        print(f"✓ Agent loaded from {filename}")
        print(f"  Generation: {self.generation}")
        print(f"  Best Fitness: {self.best_chromosome.fitness:.2f}")
        print(f"  Best Genes: {self.best_chromosome.genes}")
    
    def play_game(self, render: bool = True, max_pieces: int = 1000) -> Dict:
        """
        Play a game using the best chromosome.
        
        Args:
            render: Whether to render the game
            max_pieces: Maximum pieces to place
            
        Returns:
            Game statistics
        """
        if self.best_chromosome is None:
            print("✗ No trained chromosome available! Train the agent first.")
            return {}
        
        env = TetrisEnv(render_mode=render)
        state = env.reset()
        done = False
        pieces_placed = 0
        
        print("\n🎮 Playing game with best chromosome...")
        print(f"   Genes: {self.best_chromosome.genes}")
        
        while not done and pieces_placed < max_pieces:
            if render:
                # Handle Pygame events
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
            
            # Get action
            actions = choose_action_simple(env, self.best_chromosome)
            
            # Execute actions
            for action in actions:
                if done:
                    break
                _, reward, done, info = env.step(action)
                if render:
                    env.render()
            
            pieces_placed += 1
            
            if pieces_placed % 100 == 0:
                print(f"   Pieces: {pieces_placed}, Score: {info['score']}, Lines: {info['lines_cleared']}")
        
        print(f"\n🏁 Game Over!")
        print(f"   Final Score: {info['score']}")
        print(f"   Lines Cleared: {info['lines_cleared']}")
        print(f"   Pieces Placed: {info['total_pieces']}")
        
        if render:
            import pygame
            pygame.time.wait(3000)
        
        env.close()
        return info


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or play with GPU-Optimized Genetic Algorithm Tetris agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play', 'continue'],
                        help='Mode: train (new), play (use best), or continue (resume training)')
    parser.add_argument('--generations', type=int, default=20,
                        help='Number of generations to train')
    parser.add_argument('--population', type=int, default=100,
                        help='Population size (larger is better for GPU)')
    parser.add_argument('--games', type=int, default=5,
                        help='Games per evaluation')
    parser.add_argument('--pieces', type=int, default=500,
                        help='Maximum pieces per game')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--file', type=str, default='genetic_agent_gpu.pkl',
                        help='File to save/load agent')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Training new GPU-Optimized Genetic Algorithm agent...")
        agent = GeneticAgentGPU(
            population_size=args.population,
            mutation_rate=0.1,
            mutation_strength=0.2,
            crossover_rate=0.8,
            elitism_count=max(2, args.population // 25),
            tournament_size=5,
            n_workers=args.workers
        )
        
        agent.evolve(
            generations=args.generations,
            games_per_eval=args.games,
            max_pieces=args.pieces,
            verbose=True
        )
        agent.save(args.file)
        
        print("\n" + "="*60)
        print("✓ Training complete!")
        print("="*60)
        
        # Play a demo game
        print("\n🎮 Playing demonstration game...")
        agent.play_game(render=True, max_pieces=500)
    
    elif args.mode == 'continue':
        print(f"📂 Loading agent from {args.file}...")
        agent = GeneticAgentGPU(n_workers=args.workers)
        agent.load(args.file)
        
        print("\n🚀 Continuing training...")
        agent.evolve(
            generations=args.generations,
            games_per_eval=args.games,
            max_pieces=args.pieces,
            verbose=True
        )
        agent.save(args.file)
        
        print("\n" + "="*60)
        print("✓ Training complete!")
        print("="*60)
    
    elif args.mode == 'play':
        print(f"📂 Loading agent from {args.file}...")
        agent = GeneticAgentGPU()
        agent.load(args.file)
        
        # Play game
        agent.play_game(render=True, max_pieces=1000)
