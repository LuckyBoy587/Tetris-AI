from tetris_env import TetrisEnv, encode_state
import random


def run_random_episodes(n=5, max_steps=1000):
    env = TetrisEnv(render_mode=False)
    results = []
    for ep in range(n):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = random.randrange(env.NUMBER_OF_ACTIONS)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        results.append((total_reward, steps, info))
    return results


if __name__ == '__main__':
    res = run_random_episodes(5, max_steps=500)
    for i, (r, s, info) in enumerate(res, 1):
        print(f"Episode {i}: Reward={r:.2f}, Steps={s}, Score={info['score']}, Lines={info['lines_cleared']}")
