import random
import torch
from model import DQN


class Agent:
    def __init__(self, input_dim, action_dim):
        self.dqn = DQN(input_dim, action_dim)
        self.target_dqn = DQN(input_dim, action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)
        self.memory = []
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.dqn(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(3)]], dtype=torch.long)

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])

        # Q-value calculation
        state_action_values = self.dqn(state_batch).gather(1, action_batch)
        next_state_values = self.target_dqn(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + (GAMMA * next_state_values)

        # Loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()