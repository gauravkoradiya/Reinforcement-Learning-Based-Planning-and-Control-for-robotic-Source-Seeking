# Define the environment:

# State: [f, df, w] where these values can be derived from kinematic parameters such as slip, thrust, yaw, etc.
# Action: [stop, curved walk, sharp turn]
# Reward: A function of distance from the odor source and plume encounter rate.


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Neural Network model for Q-value approximation
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



