# Tetris AI Environment (Pygame)

A lightweight, self-contained Tetris environment designed for reinforcement learning experiments. Implemented with Pygame for rendering and NumPy for state handling. The environment is provided as a single file (`tetris_env.py`) exposing a `TetrisEnv` class with a familiar `reset/step/render/close` API.

> Status: Includes Tetris environment and Genetic Algorithm agent for AI training.


## Overview

- Board size: 10 x 20
- Actions (int):
  - `0` = move left
  - `1` = move right
  - `2` = rotate clockwise
  - `3` = soft drop (down one cell)
  - `4` = hard drop (instant drop)
- State helpers:
  - `get_state()` returns a NumPy array representing the board (occupancy/colors)
  - `get_features()` returns common Tetris features such as aggregate height, holes, bumpiness, completed lines
- Rendering:
  - Real-time rendering via Pygame when `render_mode=True`
  - Headless mode available with `render_mode=False` for training


## Tech Stack and Dependencies

- Language: Python 3.8+ (tested versions not specified; 3.10+ recommended)
- Libraries:
  - `pygame` - for game rendering
  - `numpy` - for state handling and computations
  - `matplotlib` - for training visualization (optional)
- Package manager: pip (no poetry/conda configuration present)

> TODO: Add a pinned dependencies file (e.g., `requirements.txt`) with tested versions.


## Requirements

- Operating system: Windows, macOS, or Linux
- Python 3.8 or newer
- For rendering: a display capable environment (Pygame). For servers/CI, use `render_mode=False` to avoid opening a window.


## Installation

It is recommended to use a virtual environment.

- Windows (PowerShell):
  ```powershell
  cd "D:\AI or Machine Learning\Tetris AI"
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install pygame numpy matplotlib
  ```

- macOS/Linux (bash):
  ```bash
  cd "D:/AI or Machine Learning/Tetris AI"   # adjust the path for your system
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install pygame numpy matplotlib
  ```

> Note: matplotlib is optional and only needed for training visualization.

**Verify Installation:**
```powershell
python test_installation.py
```

This will test all components and confirm everything is working correctly.


## Usage

### Quick Demo

Run a 5-minute demonstration to see the GA agent in action:

```powershell
python demo.py
```

This trains a small agent and plays a demonstration game.

### Basic Environment Usage

Basic example showing environment loop with random actions:

```python
from tetris_env import TetrisEnv
import random

# Enable rendering for a quick visual check; use render_mode=False for headless training
env = TetrisEnv(render_mode=True)

state = env.reset()

done = False
while not done:
    action = random.randint(0, 4)
    state, reward, done, info = env.step(action)
    env.render()  # no-op if render_mode=False

env.close()
```

### Headless Training

Headless training-style loop (no window):

```python
from tetris_env import TetrisEnv

env = TetrisEnv(render_mode=False)
state = env.reset()

for episode in range(10):
    done = False
    state = env.reset()
    total_reward = 0.0
    while not done:
        # TODO: replace with your agent's action selection
        action = 4  # e.g., always hard drop (for demo only)
        state, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode} reward: {total_reward}")

env.close()
```


## Public API (TetrisEnv)

- `__init__(render_mode: bool = True)`
  - If `render_mode=True`, Pygame window is created in `render()`.
- `reset() -> state`
  - Starts a new game and returns the initial state.
- `step(action: int) -> (state, reward, done, info)`
  - Applies the action. `done=True` when game over.
- `render() -> None`
  - Renders the current frame if `render_mode=True`; otherwise, no-op.
- `close() -> None`
  - Cleans up Pygame resources if used.
- Helpers for RL features:
  - `get_state()`
  - `get_features()`

See `tetris_env.py` for additional internal helpers such as collision detection, line clearing, and feature calculations.


## Scripts

### Genetic Algorithm Agent

Train an AI agent using Genetic Algorithms to evolve optimal Tetris-playing strategies.

**Train a new agent:**
```powershell
python genetic_agent.py --mode train --generations 20 --population 30 --games 3
```

**Continue training an existing agent:**
```powershell
python genetic_agent.py --mode continue --generations 10 --file genetic_agent.pkl
```

**Play a game with the trained agent:**
```powershell
python genetic_agent.py --mode play --file genetic_agent.pkl
```

**Visualize training progress:**
```powershell
python visualize_training.py --file genetic_agent.pkl
```

**Command-line arguments:**
- `--mode`: `train` (new agent), `play` (watch trained agent), or `continue` (resume training)
- `--generations`: Number of generations to evolve (default: 20)
- `--population`: Population size (default: 30)
- `--games`: Number of games per fitness evaluation (default: 3)
- `--file`: File to save/load agent (default: genetic_agent.pkl)

### How the Genetic Algorithm Works

The GA agent evolves weights for four key Tetris features:
1. **Aggregate Height**: Total height of all columns (minimized)
2. **Holes**: Empty cells with blocks above them (minimized)
3. **Bumpiness**: Height variation between adjacent columns (minimized)
4. **Completed Lines**: Number of full lines (maximized)

The algorithm:
1. Creates a population of chromosomes (each with random weights)
2. Evaluates fitness by playing Tetris games
3. Selects best performers using tournament selection
4. Creates offspring through crossover and mutation
5. Repeats for multiple generations

Typical training results in agents that can clear 100+ lines and achieve scores of 10,000+.

> TODO:
> - Add a minimal `play.py` to play with keyboard controls (if desired)
> - Add more RL algorithms (DQN, PPO, etc.)


## Environment Variables

No environment variables are required.

> Optional/Advanced:
> - For headless servers running Pygame, consider SDL environment variables or virtual displays (outside the scope of this repo). Not currently configured here.


## Tests

No tests are included.

> TODO:
> - Add unit tests for line clearing, collision logic, and feature calculations
> - Add a smoke test that runs a short random episode headlessly

Quick manual smoke test:

```python
from tetris_env import TetrisEnv

env = TetrisEnv(render_mode=False)
state = env.reset()
for _ in range(100):
    state, reward, done, info = env.step(4)  # hard drop repeatedly
    if done:
        break
env.close()
print("OK")
```


## Project Structure

```
D:/AI or Machine Learning/Tetris AI/
├── tetris_env.py           # Pygame-based Tetris RL environment (TetrisEnv)
├── genetic_agent.py        # Genetic Algorithm agent for training
├── visualize_training.py   # Training visualization tools
├── demo.py                 # Quick demonstration script
├── test_installation.py    # Installation verification
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── QUICKSTART.md           # 5-minute getting started guide
├── GA_USAGE_GUIDE.md       # Comprehensive GA usage guide
└── PROJECT_STRUCTURE.md    # Detailed file descriptions
```

Entry point(s):
- `tetris_env.py`: Import and instantiate `TetrisEnv` in your own scripts
- `genetic_agent.py`: Standalone script for training and playing with GA agent
- `visualize_training.py`: Standalone script for plotting training progress
- `demo.py`: Quick demo for first-time users
- `test_installation.py`: Verify everything is working

**New to this project?** Start with `QUICKSTART.md` for a 5-minute tutorial!


## License

No license file was found in the repository.

> TODO: Add a LICENSE file to specify terms. If you intend this to be open source, consider MIT, Apache-2.0, or GPL-3.0.


## Acknowledgements

- Built with [Pygame](https://www.pygame.org/news) and [NumPy](https://numpy.org/).


## Changelog

Not available.

> TODO: Add a `CHANGELOG.md` once releases or notable changes begin.
