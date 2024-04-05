import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Neural Network model for Q-value approximation
class DQN(nn.Module):
    """
    DQN model for Q-value approximation. The model consists of 5 fully connected layers with ReLU activation functions. The output layer has a linear activation function. The input size is the state size and the output size is the action size.
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(state_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.leakyrelu = nn.LeakyReLU()
        self.output = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = self.leakyrelu(self.linear1(x))
        x = self.leakyrelu(self.linear2(x))
        x = self.leakyrelu(self.linear3(x))
        x = self.leakyrelu(self.linear4(x))
        x = self.leakyrelu(self.linear5(x))
        return self.output(x)
