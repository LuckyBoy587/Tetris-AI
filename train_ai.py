import time
import os
import math

import torch

from tetris_ai import TetrisDQNAgent
from tetris_env import TetrisEnv, encode_state


def train_ai(num_episodes: int = 1000,
             max_steps_per_episode: int = 1000,
             save_path: str = "dqn_tetris.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = TetrisDQNAgent(TetrisEnv.STATE_SIZE, TetrisEnv.NUMBER_OF_ACTIONS, device=device)
    env = TetrisEnv(render_mode=False)

    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 20000  # decay steps

    total_steps = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = encode_state(env)
        episode_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * total_steps / eps_decay)
            action = agent.select_action(state, epsilon=eps_threshold)

            next_state, reward, done, info = env.step(action)
            next_state_enc = encode_state(env)

            agent.push_transition(state, action, reward, next_state_enc, done)
            loss = agent.optimize_step()

            state = next_state_enc
            episode_reward += reward
            step += 1
            total_steps += 1


        # periodic save
        if episode % 50 == 0:
            # simple logging
            print(f"Episode {episode:4d} | Steps {step:4d} | Reward {episode_reward:.1f} | TotalSteps {total_steps} | Eps {eps_threshold:.3f}")
            try:
                agent.save(save_path)
                print(f"Saved model to {save_path}")
            except Exception as e:
                print(f"Failed to save model: {e}")

    # final save
    try:
        agent.save(save_path)
    except Exception:
        pass


if __name__ == "__main__":
    # small smoke test: run a few episodes to ensure nothing crashes
    start = time.time()
    train_ai(num_episodes=3, max_steps_per_episode=200)
    print(f"Done in {time.time() - start:.1f}s")