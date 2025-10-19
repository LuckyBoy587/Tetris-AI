# Quick Start Guide

Welcome to Tetris AI with Genetic Algorithm! This guide will get you up and running in 5 minutes.

## Setup (2 minutes)

### 1. Install Dependencies
```powershell
# Make sure you're in the project directory
cd "D:\AI or Machine Learning\Tetris AI"

# Activate virtual environment (if you have one)
.\.venv\Scripts\Activate.ps1

# Install packages
pip install pygame numpy matplotlib
```

### 2. Verify Installation
```powershell
python test_installation.py
```

You should see all tests pass with ✓ marks.

## Your First AI Agent (3 minutes)

### Option 1: Quick Demo (Recommended)
```powershell
python demo.py
```

This will:
- Train a small agent (5 generations, ~2 minutes)
- Save it as `demo_agent.pkl`
- Play a demonstration game
- Show you the results

**Expected output:**
- Best Fitness: 500-1500
- Lines Cleared: 5-20
- A Pygame window showing the AI playing

### Option 2: Manual Quick Train
```powershell
python genetic_agent.py --mode train --generations 10 --population 20 --games 2
```

Then watch it play:
```powershell
python genetic_agent.py --mode play
```

## Understanding What You See

### During Training
You'll see output like:
```
Generation 1/10
Evaluating population...
  Evaluated 10/20 chromosomes
  Evaluated 20/20 chromosomes

Results:
  Best Fitness: 1250.00
  Avg Fitness: 650.00
  Best Chromosome: Genes: [-0.450, -0.620, -0.180, 0.720], Fitness: 1250.00
```

**What this means:**
- **Best Fitness**: Score of the best-performing chromosome
- **Avg Fitness**: Average score across all chromosomes
- **Genes**: The four weights being evolved
  - Gene 0: Height penalty (negative = avoid tall stacks)
  - Gene 1: Holes penalty (negative = avoid holes)
  - Gene 2: Bumpiness penalty (negative = prefer flat surface)
  - Gene 3: Lines reward (positive = prioritize clearing lines)

### During Play
You'll see a Pygame window with:
- **Left side**: The Tetris game board
- **Right side**: Statistics panel showing:
  - Score
  - Lines cleared
  - Pieces placed
  - Current board features

## Next Steps

### 1. Visualize Training (if you have matplotlib)
```powershell
python visualize_training.py --file demo_agent.pkl
```

This creates plots showing:
- Fitness improvement over generations
- How genes evolved
- Final gene weights

### 2. Train a Better Agent
```powershell
# 30-minute training session
python genetic_agent.py --mode train --generations 30 --population 50 --games 3
```

Expected results:
- Best Fitness: 3000-7000
- Lines/Game: 30-70
- Much better performance!

### 3. Continue Training Existing Agent
```powershell
# Continue from where you left off
python genetic_agent.py --mode continue --generations 20 --file genetic_agent.pkl
```

### 4. Compare Multiple Agents
```powershell
# Train multiple agents
python genetic_agent.py --mode train --file agent1.pkl --generations 30
python genetic_agent.py --mode train --file agent2.pkl --generations 30
python genetic_agent.py --mode train --file agent3.pkl --generations 30

# Compare them
python visualize_training.py --compare agent1.pkl agent2.pkl agent3.pkl
```

## Common Commands Cheat Sheet

```powershell
# Test installation
python test_installation.py

# Quick demo
python demo.py

# Train new agent (fast)
python genetic_agent.py --mode train --generations 15 --population 30 --games 2

# Train new agent (quality)
python genetic_agent.py --mode train --generations 50 --population 50 --games 5

# Continue training
python genetic_agent.py --mode continue --generations 20

# Watch agent play
python genetic_agent.py --mode play

# Visualize training
python visualize_training.py --file genetic_agent.pkl

# Test manual weights
python demo.py --manual
```

## Troubleshooting

### "No module named 'pygame'"
```powershell
pip install pygame numpy matplotlib
```

### "No trained chromosome available"
You need to train first:
```powershell
python genetic_agent.py --mode train --generations 10
```

### Training is too slow
Reduce population or games:
```powershell
python genetic_agent.py --mode train --population 20 --games 2 --generations 15
```

### Want better results
Train longer with more evaluations:
```powershell
python genetic_agent.py --mode train --population 50 --games 5 --generations 50
```

## Understanding Performance

After training, your agent's fitness tells you how well it plays:

| Fitness Range | Performance Level | What It Means |
|--------------|------------------|---------------|
| < 500        | Beginner         | Just learning, places 5-10 pieces |
| 500-2000     | Novice          | Can clear a few lines, 10-20 pieces |
| 2000-5000    | Intermediate    | Decent strategy, 20-50 pieces |
| 5000-10000   | Advanced        | Good player, 50-100 pieces |
| 10000-20000  | Expert          | Very strong, 100-200 pieces |
| 20000+       | Master          | Exceptional, 200+ pieces |

## Where to Learn More

1. **GA_USAGE_GUIDE.md** - Comprehensive guide with:
   - Detailed explanations
   - Hyperparameter tuning
   - Advanced techniques
   - Troubleshooting

2. **README.md** - Full project documentation

3. **Source Code**:
   - `tetris_env.py` - The Tetris game environment
   - `genetic_agent.py` - The genetic algorithm implementation
   - `visualize_training.py` - Training visualization tools

## Tips for Success

1. **Start Small**: Run `demo.py` first to verify everything works
2. **Be Patient**: Good agents need 30-50 generations
3. **Save Often**: Training auto-saves, but keep backups
4. **Visualize**: Use plots to understand what's happening
5. **Experiment**: Try different hyperparameters
6. **Compare**: Train multiple agents and pick the best

## Have Fun!

You now have everything you need to train AI agents to play Tetris using evolutionary algorithms. Experiment, learn, and enjoy watching your agents evolve! 🧬🎮

Happy training!
