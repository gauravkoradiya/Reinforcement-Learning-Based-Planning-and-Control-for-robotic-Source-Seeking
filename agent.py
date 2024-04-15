import random
import pandas as pd
import torch
from tqdm import tqdm
from model import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import randint
import random
from collections import defaultdict
from environment import FlyEnvironment


class DQNAgent:
    """
    usage:
        # load trajectory data
        df = pd.read_csv('./Data/processed/combined_trajectories.csv')

        # Initializing and using the environment
        env = FlyEnvironment(grid_size=(df.x.min(), df.x.max(), df.y.min(), df.y.max()), source_location=(250000, 250000), detection_radius=10000)
        curent_state = env.reset()

        agent = DQNAgent(state_size=len(env.state),
                        action_size=len(env.action_space),
                        batch_size=128)

        # Fill memory with random experiences
        # Assuming the batch size is 32, fill with double that amount
        for _ in tqdm(range(10000), desc=""):
            state = env.reset()
            state = np.reshape(state, [len(env.state)])
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [len(env.state)])
            agent.remember(state, action, reward, next_state, done)

        # Train the agent
        history = agent.train()
        print("Agent trained on a minibatch")
    """
    def __init__(self, state_size, action_size, batch_size=64,epsilon=0.1, gamma=0.99, learning_rate=0.001, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = list() #deque(maxlen=200000)
        self.gamma = gamma
        self.epsilon =  epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(torch.FloatTensor(state).unsqueeze(0))
        return np.argmax(act_values.cpu().data.numpy())
    
    def get_random_contiguous_batches(self):
        # contiguous_batches = []
        num_batches = len(self.memory)//self.batch_size # Number of batches to retrieve
        for _ in range(num_batches):
            # Ensure the random start index allows for a full batch
            start_index = randint(0, len(self.memory) - self.batch_size)
            batch = self.memory[start_index : start_index + self.batch_size]
            yield batch
            # contiguous_batches.append(batch)
            
        # return contiguous_batches
    def get_sequential_contiguous_batches(self):
        num_batches = len(self.memory) // self.batch_size
        
        for batch_index in range(num_batches):
            start_index = batch_index * self.batch_size
            end_index = start_index + self.batch_size
            batch = self.memory[start_index:end_index]
            yield batch

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        steps_done = 0
        history = defaultdict(list)
        # Usage example
       
        # Retrieve random contiguous batches
        contiguous_batches = self.get_sequential_contiguous_batches()#self.get_random_contiguous_batches()
        for batch in tqdm(contiguous_batches, desc="Batches", total=len(self.memory) // self.batch_size, leave=False):
            batch = list(zip(*batch))
            state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32)
            action_batch = torch.tensor(np.array(batch[1]), dtype=torch.int64).unsqueeze(1)
            reward_batch = torch.tensor(np.array(batch[2]), dtype=torch.float32)
            next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32)

            # Q-value calculation
            state_action_values = self.model(state_batch).gather(1, action_batch)
            next_state_values = self.target_model(next_state_batch).max(1)[0].detach()
            expected_state_action_values = reward_batch + (self.gamma * next_state_values)

            # Loss
            batch_loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            history['train/loss'].append(batch_loss.item())
            
            # Optimize
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Periodically update the target network
            steps_done += 1
            if steps_done % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.load_state_dict(self.model.state_dict())   
        return history
# if __name__ == "__main__":

#     # load trajectory data
#     df = pd.read_csv('./Data/processed/combined_trajectories.csv')

#     # Initializing and using the environment
#     env = FlyEnvironment(grid_size=(df.x.min(), df.x.max(), df.y.min(), df.y.max()), source_location=(250000, 250000), detection_radius=10000)
#     curent_state = env.reset()

#     agent = DQNAgent(state_size=len(env.state),
#                      action_size=len(env.action_space),
#                      batch_size=128)

#     # Fill memory with random experiences
#     # Assuming the batch size is 32, fill with double that amount
#     for _ in tqdm(range(10000), desc=""):
#         state = env.reset()
#         state = np.reshape(state, [len(env.state)])
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [len(env.state)])
#         agent.remember(state, action, reward, next_state, done)

#     # Train the agent
#     history = agent.train()
#     print("Agent trained on a minibatch")


