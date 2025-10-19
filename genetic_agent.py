"""
Genetic Algorithm Agent for Tetris
Evolves weights for heuristic evaluation of board states.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from tetris_env import TetrisEnv
import copy
import pickle
import os


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


class GeneticAgent:
    """
    Genetic Algorithm agent that evolves Tetris playing strategies.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        crossover_rate: float = 0.8,
        elitism_count: int = 2,
        tournament_size: int = 5
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
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        
        # Initialize population
        self.population = [TetrisChromosome() for _ in range(population_size)]
        self.generation = 0
        self.best_chromosome = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_genes': []
        }
    
    def get_all_possible_moves(self, env: TetrisEnv) -> List[Tuple[int, int, int]]:
        """
        Get all possible final positions for the current piece.
        
        Args:
            env: Tetris environment
            
        Returns:
            List of (rotation, column, evaluation_score) tuples
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
    
    def choose_action(self, env: TetrisEnv, chromosome: TetrisChromosome) -> List[int]:
        """
        Choose the best action sequence based on the chromosome's weights.
        
        Args:
            env: Tetris environment
            chromosome: Chromosome to use for evaluation
            
        Returns:
            List of actions to execute
        """
        possible_moves = self.get_all_possible_moves(env)
        
        if not possible_moves:
            return [4]  # Hard drop if no moves available
        
        best_score = float('-inf')
        best_move = possible_moves[0]
        
        # Evaluate each possible move
        for rotation, col, _ in possible_moves:
            # Create a copy of the environment to simulate
            test_env = self._copy_env(env)
            
            # Set rotation
            test_env.current_rotation = rotation
            test_env.current_piece = test_env.SHAPES[test_env.current_shape_name][rotation]
            
            # Set column
            test_env.current_pos[1] = col
            
            # Drop
            test_env._hard_drop()
            test_env._clear_lines()
            
            # Evaluate
            score = chromosome.evaluate_board(test_env)
            
            if score > best_score:
                best_score = score
                best_move = (rotation, col, _)
        
        # Convert best move to action sequence
        actions = self._move_to_actions(env, best_move[0], best_move[1])
        return actions
    
    def _copy_env(self, env: TetrisEnv) -> TetrisEnv:
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
    
    def _move_to_actions(self, env: TetrisEnv, target_rotation: int, target_col: int) -> List[int]:
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
    
    def evaluate_chromosome(self, chromosome: TetrisChromosome, games: int = 3, max_pieces: int = 500) -> float:
        """
        Evaluate a chromosome's fitness by playing games.
        
        Args:
            chromosome: Chromosome to evaluate
            games: Number of games to play
            max_pieces: Maximum pieces per game
            
        Returns:
            Average fitness score
        """
        total_fitness = 0.0
        
        for _ in range(games):
            env = TetrisEnv(render_mode=False)
            state = env.reset()
            done = False
            pieces_placed = 0
            
            while not done and pieces_placed < max_pieces:
                # Get action
                actions = self.choose_action(env, chromosome)
                
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
        chromosome.fitness = avg_fitness
        return avg_fitness
    
    def tournament_selection(self) -> TetrisChromosome:
        """
        Select a chromosome using tournament selection.
        
        Returns:
            Selected chromosome
        """
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda c: c.fitness)
    
    def crossover(self, parent1: TetrisChromosome, parent2: TetrisChromosome) -> Tuple[TetrisChromosome, TetrisChromosome]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring chromosomes
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        point = random.randint(1, len(parent1.genes) - 1)
        
        child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
        
        return TetrisChromosome(child1_genes), TetrisChromosome(child2_genes)
    
    def evolve(self, generations: int = 50, games_per_eval: int = 3, verbose: bool = True):
        """
        Evolve the population for a number of generations.
        
        Args:
            generations: Number of generations to evolve
            games_per_eval: Number of games to play for evaluation
            verbose: Print progress information
        """
        for gen in range(generations):
            # Evaluate population
            if verbose:
                print(f"\nGeneration {self.generation + 1}/{self.generation + generations}")
                print("Evaluating population...")
            
            for i, chromosome in enumerate(self.population):
                fitness = self.evaluate_chromosome(chromosome, games=games_per_eval)
                if verbose and (i + 1) % 10 == 0:
                    print(f"  Evaluated {i + 1}/{self.population_size} chromosomes")
            
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
                print(f"\nResults:")
                print(f"  Best Fitness: {best_fitness:.2f}")
                print(f"  Avg Fitness: {avg_fitness:.2f}")
                print(f"  Best Chromosome: {self.population[0]}")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep top performers
            for i in range(self.elitism_count):
                new_population.append(self.population[i].copy())
            
            # Create offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate
                child1.mutate(self.mutation_rate, self.mutation_strength)
                child2.mutate(self.mutation_rate, self.mutation_strength)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
            self.generation += 1
    
    def save(self, filename: str = "genetic_agent.pkl"):
        """
        Save the agent to a file.
        
        Args:
            filename: File to save to
        """
        data = {
            'best_chromosome': self.best_chromosome,
            'generation': self.generation,
            'history': self.history,
            'population': self.population
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Agent saved to {filename}")
    
    def load(self, filename: str = "genetic_agent.pkl"):
        """
        Load the agent from a file.
        
        Args:
            filename: File to load from
        """
        if not os.path.exists(filename):
            print(f"File {filename} not found!")
            return
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.best_chromosome = data['best_chromosome']
        self.generation = data['generation']
        self.history = data['history']
        self.population = data['population']
        
        print(f"Agent loaded from {filename}")
        print(f"Generation: {self.generation}")
        print(f"Best Chromosome: {self.best_chromosome}")
    
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
            print("No trained chromosome available! Train the agent first.")
            return {}
        
        env = TetrisEnv(render_mode=render)
        state = env.reset()
        done = False
        pieces_placed = 0
        
        print("\nPlaying game with best chromosome...")
        print(f"Genes: {self.best_chromosome.genes}")
        
        while not done and pieces_placed < max_pieces:
            if render:
                # Handle Pygame events
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
            
            # Get action
            actions = self.choose_action(env, self.best_chromosome)
            
            # Execute actions
            for action in actions:
                if done:
                    break
                _, reward, done, info = env.step(action)
                if render:
                    env.render()
            
            pieces_placed += 1
            
            if pieces_placed % 100 == 0:
                print(f"Pieces: {pieces_placed}, Score: {info['score']}, Lines: {info['lines_cleared']}")
        
        print(f"\nGame Over!")
        print(f"Final Score: {info['score']}")
        print(f"Lines Cleared: {info['lines_cleared']}")
        print(f"Pieces Placed: {info['total_pieces']}")
        
        if render:
            import pygame
            pygame.time.wait(3000)
        
        env.close()
        return info


# Example usage and training
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or play with Genetic Algorithm Tetris agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play', 'continue'],
                        help='Mode: train (new), play (use best), or continue (resume training)')
    parser.add_argument('--generations', type=int, default=20,
                        help='Number of generations to train')
    parser.add_argument('--population', type=int, default=30,
                        help='Population size')
    parser.add_argument('--games', type=int, default=3,
                        help='Games per evaluation')
    parser.add_argument('--file', type=str, default='genetic_agent.pkl',
                        help='File to save/load agent')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training new Genetic Algorithm agent...")
        agent = GeneticAgent(
            population_size=args.population,
            mutation_rate=0.1,
            mutation_strength=0.2,
            crossover_rate=0.8,
            elitism_count=2,
            tournament_size=5
        )
        
        agent.evolve(generations=args.generations, games_per_eval=args.games, verbose=True)
        agent.save(args.file)
        
        print("\n" + "="*50)
        print("Training complete!")
        print("="*50)
        
        # Play a demo game
        print("\nPlaying demonstration game...")
        agent.play_game(render=True, max_pieces=500)
    
    elif args.mode == 'continue':
        print(f"Loading agent from {args.file}...")
        agent = GeneticAgent()
        agent.load(args.file)
        
        print("\nContinuing training...")
        agent.evolve(generations=args.generations, games_per_eval=args.games, verbose=True)
        agent.save(args.file)
        
        print("\n" + "="*50)
        print("Training complete!")
        print("="*50)
    
    elif args.mode == 'play':
        print(f"Loading agent from {args.file}...")
        agent = GeneticAgent()
        agent.load(args.file)
        
        # Play game
        agent.play_game(render=True, max_pieces=1000)
