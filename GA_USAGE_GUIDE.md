# Genetic Algorithm Agent - Usage Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Training Process](#training-process)
4. [Understanding the Algorithm](#understanding-the-algorithm)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Genetic Algorithm (GA) agent evolves optimal strategies for playing Tetris by learning weights for evaluating board states. Unlike deep learning approaches, this is a lightweight, interpretable method that can train on a CPU in minutes.

### Key Features
- **Fast Training**: Trains in 10-30 minutes on a modern CPU
- **Interpretable**: Uses 4 simple weights for board evaluation
- **No GPU Required**: Pure Python implementation
- **Deterministic**: Same parameters = same results (with seed)
- **Configurable**: Extensive hyperparameter control

---

## Quick Start

### 1. Five-Minute Demo
```powershell
# Run a quick demo (5 generations, ~2 minutes)
python demo.py
```

This will:
- Train a small agent (10 individuals, 5 generations)
- Save to `demo_agent.pkl`
- Play a demonstration game

### 2. Serious Training
```powershell
# Train a competitive agent (30-60 minutes)
python genetic_agent.py --mode train --generations 50 --population 50 --games 5
```

### 3. Watch Your Agent Play
```powershell
# After training
python genetic_agent.py --mode play --file genetic_agent.pkl
```

### 4. Visualize Progress
```powershell
# Generate training plots
python visualize_training.py --file genetic_agent.pkl
```

---

## Training Process

### Stage 1: Quick Exploration (Generations 1-10)
**Goal**: Find generally good weight directions

```powershell
python genetic_agent.py --mode train --generations 10 --population 30 --games 2
```

**Expected Results**:
- Best Fitness: 500-2000
- Lines Cleared: 5-20 per game
- Weights: Starting to favor line clears, penalize holes

### Stage 2: Refinement (Generations 11-30)
**Goal**: Fine-tune weights for better performance

```powershell
# Continue from Stage 1
python genetic_agent.py --mode continue --generations 20 --population 50 --games 3
```

**Expected Results**:
- Best Fitness: 2000-5000
- Lines Cleared: 20-50 per game
- Weights: Well-balanced, stable

### Stage 3: Optimization (Generations 31+)
**Goal**: Squeeze out maximum performance

```powershell
# Continue from Stage 2
python genetic_agent.py --mode continue --generations 20 --population 50 --games 5
```

**Expected Results**:
- Best Fitness: 5000-15000+
- Lines Cleared: 50-200+ per game
- Weights: Highly optimized, consistent

---

## Understanding the Algorithm

### The Chromosome (Genome)

Each chromosome has 4 genes representing weights for board features:

```python
genes = [
    aggregate_height_weight,  # Gene 0: Penalty for tall stacks
    holes_weight,             # Gene 1: Penalty for holes
    bumpiness_weight,         # Gene 2: Penalty for uneven surface
    completed_lines_weight    # Gene 3: Reward for completing lines
]
```

### Board Evaluation

For each possible move, the agent evaluates the resulting board:

```python
score = (
    w0 * aggregate_height +  # Lower is better
    w1 * holes +             # Fewer is better
    w2 * bumpiness +         # Lower is better
    w3 * completed_lines     # More is better
)
```

The move with the **highest score** is chosen.

### Example Evolved Weights

A well-trained agent might have:
```
Aggregate Height: -0.510  (strong penalty for height)
Holes:           -0.356  (penalty for holes)
Bumpiness:       -0.184  (moderate penalty)
Completed Lines:  0.760  (strong reward)
```

### Evolution Process

```
Generation N:
├─ 1. Evaluate all chromosomes (play games)
├─ 2. Rank by fitness (score + lines)
├─ 3. Select parents (tournament selection)
├─ 4. Crossover (combine parent genes)
├─ 5. Mutate (random changes)
└─ 6. Create Generation N+1
```

---

## Hyperparameter Tuning

### Population Size
**What it does**: Number of different weight combinations to test

| Population | Training Time | Exploration | Best For |
|-----------|--------------|-------------|----------|
| 10-20     | Fast         | Low         | Quick tests, demos |
| 30-50     | Medium       | Good        | **Recommended** |
| 100+      | Slow         | High        | Final optimization |

```powershell
# Example: Large population for thorough search
python genetic_agent.py --mode train --population 100 --generations 30
```

### Mutation Rate
**What it does**: Probability of randomly changing each gene

- **Too Low (< 0.05)**: Gets stuck in local optima
- **Good (0.1-0.2)**: Balances stability and exploration
- **Too High (> 0.3)**: Too chaotic, can't converge

**Default**: 0.1 (10% chance per gene)

### Mutation Strength
**What it does**: Maximum size of random change

- **Low (0.1)**: Small adjustments, slow improvement
- **Medium (0.2)**: **Recommended** - balanced
- **High (0.5)**: Large jumps, might overshoot

**Default**: 0.2

### Crossover Rate
**What it does**: Probability of combining parent genes

- **Low (< 0.5)**: More asexual reproduction
- **High (0.7-0.9)**: **Recommended** - good mixing
- **1.0**: Always crossover (can be too aggressive)

**Default**: 0.8

### Games Per Evaluation
**What it does**: How many games to average for fitness

| Games | Time/Gen | Accuracy | Best For |
|-------|----------|----------|----------|
| 1-2   | Fast     | Noisy    | Early generations |
| 3-5   | Medium   | Good     | **Recommended** |
| 10+   | Slow     | Excellent| Final evaluations |

```powershell
# Example: High accuracy evaluation
python genetic_agent.py --mode train --games 10 --population 30 --generations 20
```

### Recommended Configurations

**Fast Training (10-15 minutes)**:
```powershell
python genetic_agent.py --mode train --population 20 --generations 15 --games 2
```

**Balanced (30-45 minutes)**:
```powershell
python genetic_agent.py --mode train --population 50 --generations 30 --games 3
```

**High Quality (2-3 hours)**:
```powershell
python genetic_agent.py --mode train --population 100 --generations 50 --games 5
```

---

## Advanced Usage

### Custom Chromosome

Test your own hand-tuned weights:

```python
from genetic_agent import TetrisChromosome, GeneticAgent
import numpy as np

# Create custom weights
my_genes = np.array([-0.6, -0.8, -0.2, 1.0])
chromosome = TetrisChromosome(my_genes)

# Test it
agent = GeneticAgent()
agent.best_chromosome = chromosome
agent.play_game(render=True)
```

### Parallel Training (Multiple Runs)

Train multiple agents with different random seeds:

```powershell
# Run 1
python genetic_agent.py --mode train --file agent1.pkl --generations 30

# Run 2
python genetic_agent.py --mode train --file agent2.pkl --generations 30

# Run 3
python genetic_agent.py --mode train --file agent3.pkl --generations 30

# Compare
python visualize_training.py --compare agent1.pkl agent2.pkl agent3.pkl
```

### Programmatic Usage

```python
from genetic_agent import GeneticAgent

# Create and configure agent
agent = GeneticAgent(
    population_size=50,
    mutation_rate=0.15,
    mutation_strength=0.25,
    crossover_rate=0.85,
    elitism_count=3,
    tournament_size=5
)

# Train
agent.evolve(generations=30, games_per_eval=3, verbose=True)

# Save
agent.save('my_custom_agent.pkl')

# Access best chromosome
print(f"Best genes: {agent.best_chromosome.genes}")
print(f"Best fitness: {agent.best_chromosome.fitness}")

# Play game
info = agent.play_game(render=True, max_pieces=500)
```

### Incremental Training

Train in stages with different hyperparameters:

```python
from genetic_agent import GeneticAgent

agent = GeneticAgent(population_size=30)

# Stage 1: Broad exploration
agent.mutation_strength = 0.3
agent.evolve(generations=10, games_per_eval=2)

# Stage 2: Refinement
agent.mutation_strength = 0.2
agent.population_size = 50
agent.evolve(generations=20, games_per_eval=3)

# Stage 3: Fine-tuning
agent.mutation_strength = 0.1
agent.evolve(generations=20, games_per_eval=5)

agent.save('staged_agent.pkl')
```

---

## Troubleshooting

### Problem: Agent performs poorly (< 1000 fitness)

**Possible Causes**:
- Not enough generations
- Population too small
- Games per evaluation too low (noisy fitness)

**Solutions**:
```powershell
# Increase training
python genetic_agent.py --mode continue --generations 30 --population 50 --games 5
```

### Problem: Training is too slow

**Possible Causes**:
- Population too large
- Too many games per evaluation
- Pieces per game too high

**Solutions**:
```powershell
# Reduce computational load
python genetic_agent.py --mode train --population 30 --games 2 --generations 20
```

Or modify `genetic_agent.py`:
```python
# In evaluate_chromosome method
max_pieces = 200  # Instead of 500
```

### Problem: Fitness not improving after generation 20

**Possible Causes**:
- Converged to local optimum
- Mutation rate too low
- Population diversity lost

**Solutions**:
1. Increase mutation rate:
```python
agent.mutation_rate = 0.2  # From 0.1
agent.mutation_strength = 0.3  # From 0.2
```

2. Inject diversity:
```python
# Add random chromosomes
for _ in range(10):
    agent.population.append(TetrisChromosome())
```

3. Restart with different parameters

### Problem: Pygame window crashes or freezes

**Possible Causes**:
- Running in headless environment
- Pygame event loop blocked

**Solutions**:
```powershell
# Use headless mode
python genetic_agent.py --mode train --population 30 --generations 20
# Training is always headless, only "play" mode renders
```

### Problem: "No trained chromosome available"

**Cause**: Trying to play without training first

**Solution**:
```powershell
# Train first
python genetic_agent.py --mode train --generations 10

# Then play
python genetic_agent.py --mode play
```

---

## Performance Benchmarks

Typical results after various training durations:

| Generations | Population | Games/Eval | Time    | Best Fitness | Lines/Game |
|------------|-----------|------------|---------|--------------|------------|
| 5          | 20        | 2          | 2 min   | 500-1500     | 5-15       |
| 15         | 30        | 3          | 10 min  | 1500-3000    | 15-30      |
| 30         | 50        | 3          | 30 min  | 3000-7000    | 30-70      |
| 50         | 50        | 5          | 90 min  | 7000-15000   | 70-150     |
| 100        | 100       | 5          | 6 hours | 15000-30000+ | 150-300+   |

*Results may vary based on hardware and random seed*

---

## Tips for Best Results

1. **Start Small**: Use `demo.py` to verify setup
2. **Increase Gradually**: Don't jump to 100 generations immediately
3. **Save Often**: Training saves automatically, but keep backups
4. **Monitor Progress**: Use `visualize_training.py` to check convergence
5. **Compare Runs**: Train multiple agents and pick the best
6. **Adjust for Hardware**: Reduce population/games if training is too slow
7. **Be Patient**: Good agents need 30-50 generations minimum

---

## Next Steps

After training a GA agent, consider:

1. **Compare with other algorithms**: Train a DQN or PPO agent
2. **Feature engineering**: Add new board features
3. **Multi-objective optimization**: Optimize for speed AND score
4. **Ensemble methods**: Combine multiple trained agents
5. **Human comparison**: Play manually and compare scores

Happy evolving! 🧬🎮
