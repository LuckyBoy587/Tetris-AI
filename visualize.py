import argparse
import time
import torch

from tetris_env import TetrisEnv, encode_state
from tetris_ai import TetrisDQNAgent
import pygame


def visualize(model_path: str, episodes: int = 1, delay: float = 0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create env with rendering enabled
    env = TetrisEnv(render_mode=True)

    # Build agent and load weights
    agent = TetrisDQNAgent(TetrisEnv.STATE_SIZE, TetrisEnv.NUMBER_OF_ACTIONS, device=device)
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    for ep in range(1, episodes + 1):
        state = env.reset()
        state = encode_state(env)
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # handle pygame events to allow closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    print("Visualization interrupted by user.")
                    break

            # select greedy action
            action = agent.select_action(state, epsilon=0.0)
            print(f"Selected action: {action}")
            next_state, reward, terminated, info = env.step(action)
            state = encode_state(env)
            total_reward += reward
            steps += 1
            done = done or terminated

            env.render()

        print(f"Episode {ep}: Reward={total_reward}, Steps={steps}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained DQN model playing Tetris")
    parser.add_argument('model_path', type=str, help='Path to saved model file')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between frames (seconds)')
    args = parser.parse_args()

    visualize(args.model_path, episodes=args.episodes, delay=args.delay)
