import random
from collections import deque, namedtuple
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class TetrisDQNAgent:
    """
    DQN agent wrapper. Not a torch.nn.Module itself to keep compatibility with existing imports.

    Public API:
      - select_action(state, epsilon)
      - push_transition(s,a,r,ns,done)
      - optimize_step()
      - save(path), load(path)
    """

    def __init__(self, state_size: int, action_size: int, device: Optional[torch.device] = None):
        self.state_size = state_size
        self.action_size = action_size

        # device selection (use CUDA if available)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # networks
        self.policy_net = MLP(state_size, action_size).to(self.device)
        self.target_net = MLP(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # replay buffer and optimizer
        self.replay_buffer = ReplayBuffer(100000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        # training hyperparams
        self.batch_size = 64
        self.gamma = 0.99
        self.target_update = 1000  # steps
        self.learn_step_counter = 0

    def preprocess(self, state: np.ndarray) -> torch.Tensor:
        """Convert numpy state to torch tensor on the agent's device."""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        tensor = torch.from_numpy(state.astype(np.float32)).to(self.device)
        return tensor

    def select_action(self, state: np.ndarray, epsilon: float = 0.01) -> int:
        """Epsilon-greedy action selection. State is numpy array (flat)."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        self.policy_net.eval()
        with torch.no_grad():
            s = self.preprocess(state).unsqueeze(0)  # batch dim
            q_values = self.policy_net(s)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize_step(self):
        """Perform one optimization step if enough samples are available."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        transitions = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.from_numpy(np.vstack(transitions.state).astype(np.float32)).to(self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        non_final_mask = torch.tensor([not d for d in transitions.done], dtype=torch.bool, device=self.device)

        non_final_next_states = None
        if any(not d for d in transitions.done):
            next_states = [s for s, d in zip(transitions.next_state, transitions.done) if not d]
            non_final_next_states = torch.from_numpy(np.vstack(next_states).astype(np.float32)).to(self.device)

        # Compute Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute target
        next_q_values = torch.zeros((self.batch_size, 1), device=self.device)
        if non_final_next_states is not None:
            with torch.no_grad():
                next_q = self.target_net(non_final_next_states)
                next_q_values[non_final_mask] = next_q.max(1)[0].unsqueeze(1)

        expected_q = reward_batch + (self.gamma * next_q_values)

        loss = self.criterion(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.detach().cpu().item())

    def save(self, path: str):
        torch.save({'policy_state': self.policy_net.state_dict(),
                    'target_state': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(d['policy_state'])
        self.target_net.load_state_dict(d.get('target_state', d['policy_state']))
        if 'optimizer' in d:
            try:
                self.optimizer.load_state_dict(d['optimizer'])
            except Exception:
                pass

    # convenience: make the agent callable like before for compatibility with train_ai.py
    def __call__(self, state, epsilon: float = 0.01):
        return self.select_action(state, epsilon)