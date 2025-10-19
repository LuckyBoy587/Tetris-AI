# Tetris AI - Complete Project Structure

## File Overview

```
Tetris AI/
│
├── Core Environment
│   └── tetris_env.py              # Tetris game environment (RL interface)
│
├── Genetic Algorithm Agent
│   ├── genetic_agent.py           # GA implementation & training
│   ├── visualize_training.py      # Training progress visualization
│   └── demo.py                    # Quick demonstration script
│
├── Testing & Setup
│   ├── test_installation.py       # Verify installation
│   └── requirements.txt           # Python dependencies
│
├── Documentation
│   ├── README.md                  # Main project documentation
│   ├── QUICKSTART.md              # 5-minute getting started guide
│   ├── GA_USAGE_GUIDE.md          # Comprehensive GA usage guide
│   └── PROJECT_STRUCTURE.md       # This file
│
└── Configuration
    └── .gitignore                 # Git ignore patterns
```

## File Details

### Core Environment

#### tetris_env.py (550 lines)
**Purpose**: Pygame-based Tetris game that works as an RL environment

**Key Classes**:
- `TetrisEnv`: Main environment class with reset/step/render API

**Key Features**:
- 10x20 game board
- 5 actions (left, right, rotate, soft drop, hard drop)
- 7 tetromino shapes with rotations
- Board state and feature extraction
- Reward calculation
- Line clearing logic

**Usage**:
```python
from tetris_env import TetrisEnv
env = TetrisEnv(render_mode=True)
state = env.reset()
state, reward, done, info = env.step(action)
```

### Genetic Algorithm Agent

#### genetic_agent.py (600+ lines)
**Purpose**: Complete GA implementation for evolving Tetris strategies

**Key Classes**:
- `TetrisChromosome`: Represents a solution (4 genes = 4 weights)
- `GeneticAgent`: Manages population, evolution, and gameplay

**Key Features**:
- Population-based evolution
- Tournament selection
- Single-point crossover
- Gaussian mutation
- Elitism
- Fitness evaluation through gameplay
- Save/load trained agents
- Play mode with rendering

**Usage**:
```python
from genetic_agent import GeneticAgent
agent = GeneticAgent(population_size=50)
agent.evolve(generations=30, games_per_eval=3)
agent.save('my_agent.pkl')
agent.play_game(render=True)
```

**Command Line**:
```bash
# Train
python genetic_agent.py --mode train --generations 30 --population 50

# Continue training
python genetic_agent.py --mode continue --generations 20

# Play
python genetic_agent.py --mode play
```

#### visualize_training.py (250 lines)
**Purpose**: Generate plots and statistics from trained agents

**Key Functions**:
- `plot_training_history()`: Create 4-subplot training visualization
- `compare_agents()`: Compare multiple agents side-by-side

**Generates**:
- Best/average fitness over time
- Generation-to-generation improvement
- Gene weight evolution
- Final gene weights bar chart

**Usage**:
```bash
python visualize_training.py --file genetic_agent.pkl
python visualize_training.py --compare agent1.pkl agent2.pkl agent3.pkl
```

#### demo.py (120 lines)
**Purpose**: Quick demonstration of the GA agent

**Features**:
- Fast training (5 generations, 10 population)
- Automatic gameplay demo
- Manual chromosome testing
- Beginner-friendly output

**Usage**:
```bash
python demo.py              # Quick demo
python demo.py --manual     # Test manual weights
```

### Testing & Setup

#### test_installation.py (200 lines)
**Purpose**: Verify that all components are working

**Tests**:
- ✓ Package imports (pygame, numpy, matplotlib)
- ✓ TetrisEnv functionality
- ✓ GeneticAgent functionality
- ✓ Visualization module

**Usage**:
```bash
python test_installation.py
```

**Expected Output**:
```
Testing imports...
  ✓ pygame 2.5.2
  ✓ numpy 1.24.3
  ✓ matplotlib 3.7.1

Testing TetrisEnv...
  ✓ TetrisEnv created
  ✓ reset() works
  ...

✓ All tests passed!
```

#### requirements.txt
**Purpose**: Python package dependencies

**Contents**:
```
pygame>=2.5.0
numpy>=1.24.0
matplotlib>=3.7.0
```

**Installation**:
```bash
pip install -r requirements.txt
```

### Documentation

#### README.md (200+ lines)
**Purpose**: Main project documentation

**Sections**:
- Overview and features
- Tech stack and dependencies
- Installation instructions
- Usage examples
- API reference
- Scripts and commands
- Project structure
- License and acknowledgements

**Audience**: All users

#### QUICKSTART.md (350+ lines)
**Purpose**: Get users running in 5 minutes

**Sections**:
- 2-minute setup
- 3-minute first agent
- Understanding output
- Next steps
- Command cheat sheet
- Troubleshooting
- Performance benchmarks

**Audience**: Beginners

#### GA_USAGE_GUIDE.md (600+ lines)
**Purpose**: Comprehensive guide to genetic algorithm training

**Sections**:
1. Overview and features
2. Quick start examples
3. Training process (3 stages)
4. Understanding the algorithm
5. Hyperparameter tuning
6. Advanced usage
7. Troubleshooting
8. Performance benchmarks
9. Tips and best practices

**Audience**: Intermediate to advanced users

#### PROJECT_STRUCTURE.md (This file)
**Purpose**: Complete overview of all project files

**Audience**: Developers wanting to understand the codebase

### Configuration

#### .gitignore
**Purpose**: Specify files to exclude from version control

**Ignores**:
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`)
- IDE files (`.idea/`, `.vscode/`)
- Trained models (`*.pkl`)
- Generated plots (`*_training_plot.png`)
- OS files (`.DS_Store`, `Thumbs.db`)

## Data Flow

```
User Input
    ↓
genetic_agent.py
    ↓
Creates → TetrisChromosome (4 weights)
    ↓
Plays games using → tetris_env.py
    ↓
Evaluates fitness → Score + Lines
    ↓
Evolves population → Selection, Crossover, Mutation
    ↓
Saves best → genetic_agent.pkl
    ↓
Visualizes → visualize_training.py → PNG plots
```

## Typical Workflow

### First-Time User
1. Install: `pip install -r requirements.txt`
2. Verify: `python test_installation.py`
3. Demo: `python demo.py`
4. Read: `QUICKSTART.md`

### Training Session
1. Train: `python genetic_agent.py --mode train --generations 30`
2. Visualize: `python visualize_training.py`
3. Play: `python genetic_agent.py --mode play`
4. Continue: `python genetic_agent.py --mode continue --generations 20`

### Development
1. Read: `tetris_env.py` for environment API
2. Read: `genetic_agent.py` for GA implementation
3. Modify: Add new features or algorithms
4. Test: `python test_installation.py`

## Key Algorithms

### Tetris Environment
- **Collision Detection**: Check piece boundaries and overlaps
- **Line Clearing**: Detect and remove complete rows
- **Feature Extraction**: Calculate height, holes, bumpiness
- **Reward Function**: Score-based with penalties for height/holes

### Genetic Algorithm
- **Selection**: Tournament selection (5 individuals)
- **Crossover**: Single-point crossover at random gene
- **Mutation**: Gaussian mutation with configurable rate/strength
- **Elitism**: Keep top N performers unchanged
- **Fitness**: Average score over multiple games

## Performance Characteristics

### Memory Usage
- Environment: ~1 MB per instance
- Population of 50: ~50 MB
- Training history: ~1-10 MB

### Computation Time (CPU)
- 1 game (500 pieces): ~1-5 seconds
- 1 generation (pop=50, games=3): ~2-10 minutes
- 30 generations: ~30-60 minutes
- 100 generations: ~2-5 hours

### File Sizes
- `tetris_env.py`: ~20 KB
- `genetic_agent.py`: ~30 KB
- Trained agent (`.pkl`): ~10-50 KB
- Training plot (`.png`): ~100-500 KB

## Extension Points

Want to extend the project? Here are the key extension points:

### 1. Add New Features to Environment
**File**: `tetris_env.py`
**Method**: `get_features()`
**Example**: Add "wells" (deep single-column gaps)

### 2. Change Reward Function
**File**: `tetris_env.py`
**Method**: `step()`
**Example**: Add time bonus, combo rewards

### 3. Modify Evolution Strategy
**File**: `genetic_agent.py`
**Methods**: `crossover()`, `mutate()`, `tournament_selection()`
**Example**: Try uniform crossover, adaptive mutation

### 4. Add New Visualization
**File**: `visualize_training.py`
**Function**: Add new plotting function
**Example**: Plot diversity over time

### 5. Implement New AI Algorithm
**New File**: `dqn_agent.py`, `ppo_agent.py`, etc.
**Interface**: Use `TetrisEnv` API
**Example**: Deep Q-Network, A3C, etc.

## Dependencies Graph

```
genetic_agent.py
    ├── requires: tetris_env.py
    ├── requires: numpy
    ├── requires: pygame (indirect)
    └── imports: pickle, random, copy

visualize_training.py
    ├── requires: matplotlib
    ├── requires: numpy
    └── requires: pickle

demo.py
    ├── requires: genetic_agent.py
    └── requires: numpy

test_installation.py
    ├── requires: tetris_env.py
    ├── requires: genetic_agent.py
    └── optional: visualize_training.py
```

## Version History

### Current Version (v1.0)
- ✓ Tetris environment with Pygame rendering
- ✓ Genetic Algorithm agent
- ✓ Training visualization
- ✓ Complete documentation
- ✓ Test suite

### Future Versions (Planned)
- v1.1: Deep Q-Network (DQN) agent
- v1.2: Human playable mode with keyboard
- v1.3: Replay system
- v2.0: Multi-agent tournament mode

## Contact & Support

For questions, issues, or contributions:
- Check documentation: `README.md`, `QUICKSTART.md`, `GA_USAGE_GUIDE.md`
- Run tests: `python test_installation.py`
- GitHub: (Add your repository URL)

## License

(Add license information)

---

**Last Updated**: 2025-10-19
**Project Status**: Active Development
**Python Version**: 3.8+
**Tested On**: Windows, macOS, Linux
