from math import e
import re
import torch
import torch.nn as nn
from agent import DQNAgent
from model import DQN
from collections import defaultdict
from environment import FlyEnvironment
import numpy as np
import pandas as pd
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging.config
import os
import glob
import time 
import yaml
import argparse

from torch.utils.tensorboard import SummaryWriter

LOGGER_CONFIG_PATH = "./logger_config.yaml"
DATA_PATH = "./Data/processed"

# Load the config file
with open(os.path.join(LOGGER_CONFIG_PATH), 'rt') as f:
    logger_config = yaml.safe_load(f.read())
    # Configure the logging module with the config file
    logging.config.dictConfig(logger_config)
logger = logging.getLogger('my_logger')

# Initialize tensorboard writer
writer = SummaryWriter()

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
EPOCHS = 10
MAX_STEPS = 2000000
BATCH_SIZE = 256
EPSILON = 0.2
EPSILON_MIN = 0.05
EPSILON_DECAY = 200


def data_processing(path:str, version:int=1, process_data:bool=False):
    """
    Process the data from the given path and return the trajectories with actions.

    Args:
        path (str): Path to the data directory
        version (int, optional): version of action_calculation method to use. Defaults to 1.
        process_data (bool, optional): Whether to process the data or not. Defaults to False. If true, then it will calculate the action based on the version provided and add the next_x, next_y and done columns to the dataframe.

    Returns:
        list(dataframe): list of trajectories
    """
    all_files = glob.glob(os.path.join(path, "*.csv"))
    trajectories = []
    for filename in tqdm(all_files, desc="Processing files", total=len(all_files)):
        trajectory = pd.read_csv(os.path.join(filename))
        if process_data:
            # Assigning subsequent coordinates
            trajectory['next_x'] = trajectory['x'].shift(-1)
            trajectory['next_y'] = trajectory['y'].shift(-1)
            trajectory['done'] = False
            # Mark the last row's 'done' as True
            trajectory.at[trajectory.index[-1], 'done'] = True 
            if version == 1:
                trajectory = compute_action_v1(trajectory)
            else:
                trajectory = compute_action_v2(trajectory)
        trajectories.append(trajectory)
    logger.info(f"Processed {len(trajectories)} trajectories.")
    return trajectories

def compute_action_v1(df):
    
    """
    Using tan agle to calculate the curvature of the trajectory. Segment the trajectory based on the differenet threshold of theta.
    Args:
        df (pd.DataFrame): Dataframe containing x and y coordinates. 
    """

    df['dx'] = - df['x'].diff(-1)
    df['dy'] = - df['y'].diff(-1)

    # Normalize to get only the direction
    directions = np.arctan2(df['dy'], df['dx'])
    
    # Categorize directions into actions
    # Note: Adjust thresholds as needed to distinguish between curved and sharp turns
    conditions = [
        (df['dx'] == 0) & (df['dy'] == 0),  # Stop
        (directions.between(-np.pi/8, np.pi/8, inclusive='both')),  # Right
        (directions.between(np.pi/8, 3*np.pi/8, inclusive='both')),  # Up-Right
        (directions.between(3*np.pi/8, 5*np.pi/8, inclusive='both')),  # Up
        (directions.between(5*np.pi/8, 7*np.pi/8, inclusive='both')),  # Up-Left
        (directions.between(-7*np.pi/8, -5*np.pi/8, inclusive='both')),  # Down-Left
        (directions.between(-5*np.pi/8, -3*np.pi/8, inclusive='both')),  # Down
        (directions.between(-3*np.pi/8, -np.pi/8, inclusive='both')),  # Down-Right
         # Handle 'left' by combining angles close to -π and π
        ((directions > 7*np.pi/8) | (directions < -7*np.pi/8)),  # Left
    ]
    choices =  ['stop', 'right', 'up_right', 'up', 'up_left', 'bottom_left', 'down', 'bottom_right', 'left']
    df['action_theta'] = np.select(conditions, choices, default='stop')
    return df

def compute_action_v2(df):
    """

    Args:
        df (pd.DataFrame): Dataframe containing x and y coordinates

    Returns:
        df : pd.DataFrame
    """
    # Calculate differences in x and y
    df['dx'] = - df['x'].diff(-1)
    df['dy'] = - df['y'].diff(-1)
    
    # Define movement directions based on dx and dy
    conditions = [
        (df['dx'] == 0) & (df['dy'] == 0),
        (df['dx'] == 0) & (df['dy'] > 0),
        (df['dx'] == 0) & (df['dy'] < 0),
        (df['dx'] < 0) & (df['dy'] == 0),
        (df['dx'] > 0) & (df['dy'] == 0),
        (df['dx'] > 0) & (df['dy'] > 0),
        (df['dx'] < 0) & (df['dy'] > 0),
        (df['dx'] > 0) & (df['dy'] < 0),
        (df['dx'] < 0) & (df['dy'] < 0)
    ]
    actions = ['stop', 'up', 'down', 'left', 'right', 'up_right', 'up_left', 'bottom_right', 'bottom_left']
    df['action_coordinates'] = np.select(conditions, actions, default='stop')  # Default to 'stop' if none of the conditions match
    
    return df


def trainnig(agent, env, trajectories, epochs=EPOCHS):
    """
    Train the agent on the environment for the specified number of epochs.

    Args:
        agent (DQNAgent): Agent object e.g DQNAgent
        env (FLyEnvironment): Environment object e.g FlyEnvironment
        epochs (int, optional): NUmber of times the model train on set of trajectories. Defaults to 10.
    """

    episode_losses = []
    episode_rewards = []
    for episode, trajectory in enumerate(tqdm(trajectories, desc="Episodes", total=len(trajectories))):
        total_reward = 0  # Track total reward per episode
        for i in tqdm(range(len(trajectory)-1), desc="Rolling Over Trajactory", total=len(trajectory), leave=False):
            state = trajectory.iloc[i][['x', 'y', 'light']].values
            next_state = trajectory.iloc[i+1][['x', 'y', 'light']].values

            # Assuming you have a way to determine the action and reward from your data
            action = env.action_space.index(trajectory.iloc[i]['action_theta']) #trajectory.iloc[i]['action_coordinates']

            reward = -1 if trajectory.iloc[i]['light'] == 0 else 1 # Determine the reward received for the transition
            done = (i == len(trajectory) - 2)  #trajectory.iloc[i]['done'] # done =  # Mark done on the last transition

            # Reshape for compatibility with your DQN model input
            state = np.reshape(state, [agent.state_size]).astype(np.float32)
            next_state = np.reshape(next_state, [agent.state_size]).astype(np.float32)
                
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward

            if done:
                break

        # Train the agent on the current trajectory
        epoch_losses = []
        for epoch in tqdm(range(epochs), desc="Training Epochs", leave=False):
            history = agent.train()
            avg_epoch_loss = np.mean(history['train/loss'])
            epoch_losses.append(avg_epoch_loss)
            writer.add_scalar(f"Training || Episode: {episode} || Average Loss/Epoch", avg_epoch_loss, epoch)

        # Optionally reduce epsilon after each episode to reduce exploration over time
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        # Log results in tensorboard
        avg_episode_loss = np.mean(epoch_losses)
        episode_losses.append(avg_episode_loss)
        episode_rewards.append(total_reward)

        writer.add_scalar("Training || Average Loss/Episode", avg_episode_loss, episode)
        writer.add_scalar("Training || Reward/Episode", total_reward, episode)

        # # Calculate and log the average loss for the epoch
        # avg_epoch_loss = np.mean(epoch_losses)
        # writer.add_scalar('Average Epoch Loss/Trainining', avg_epoch_loss, epoch)

        # Optionally save the model after each epoch
        torch.save(agent.model.state_dict(), os.path.join('models','model_episode_{0}_{1}.pth'.format(episode, time.time())))
        writer.flush()

def evaluate(agent, env, num_episodes=10):
    """
    Evaluate the agent's performance on the environment.
    
    Args:
        agent: The DQNAgent instance.
        env: The environment instance, which should be compatible with the agent.
        num_episodes (int): Number of episodes to evaluate the agent.
    """
    episode_rewards = []
    for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", total=num_episodes):
        state = env.reset()  # Initialize the environment and get the initial state
        state = np.reshape(state, [agent.state_size]).astype(np.float32)
        done = False
        total_reward = 0
        steps = 0
        step_positions = [state[:2]]  # Assuming the first two elements of state are x and y positions

        while (not done) and (steps < MAX_STEPS):
            action = agent.act(state)  # Select the best action for the current state
            next_state, reward, done, _ = env.step(action)  # Take the action in the environment
            next_state = np.reshape(next_state, [agent.state_size]).astype(np.float32)

            step_positions.append(next_state[:2])  # Track the position after taking the action
            state = next_state
            total_reward += reward
            steps += 1

        # log the total reward for the episode
        writer.add_scalar("Evaluation: Reward/Episode", total_reward, episode)
        episode_rewards.append(total_reward)
        
        # Optionally visualize the trajectory
        step_positions = np.array(step_positions)
        fig, ax = plt.subplots()
        # ax.plot(step_positions[:, 0], step_positions[:, 1], marker='o')
        # Agent Trjaectory
        ax.plot(step_positions[:, 0], step_positions[:, 1], color='blue', marker='o', linestyle='-', linewidth=1, markersize=2, label='Agent Path')
        
        # Starting position
        ax.scatter(*step_positions[0], color='green', s=100, label='Start', edgecolors='darkgreen', zorder=5)
        
        # Ending position
        ax.scatter(*step_positions[-1], color='red', s=100, label='End', edgecolors='darkred', zorder=5)

        # Source location
        ax.plot(env.source_location[0], env.source_location[1], 'r*', markersize=15, label='Source')

        #plot the detection boundary
        detection_circle = plt.Circle(env.source_location, env.detection_radius, color='r', fill=False, linestyle='--', label='Detection Boundary')
        ax.add_artist(detection_circle)
        # arena_circle = plt.Circle(self.source_location, 80000, color='black', fill=False, linestyle='-', label='Arena Boundary')
        # ax.add_artist(arena_circle)

        ax.set_title(f"Episode {episode+1}: Total Reward = {total_reward}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        # ax.set_xlim(env.x_min, env.x_max)
        # ax.set_ylim(env.y_min, env.y_max)
        plt.legend()
        # Log the matplotlib figure in TensorBoard
        writer.add_figure(f"Evaluation: Trajectory/Episode_{episode+1} || Reward: {total_reward}", fig, global_step=episode)
    
    # Calculate and log the average reward over all episodes
    avg_reward = np.mean(episode_rewards)
    return avg_reward


def main():
    try:
        parser=argparse.ArgumentParser()
        parser.add_argument("--training", help="Train the model", action="store_true", default=False)
        parser.add_argument("--inference", help="Evaluate model the model", action="store_true", default=True)
        parser.add_argument("model_path", nargs='?',default=None, help="Path to the model file")
        args=parser.parse_args()

        #1. Load the data
        if args.training:
            trajectories =  data_processing(DATA_PATH) #[pd.read_csv(os.path.join('./Data/sample_trajectory.csv'))] #  # 
            df = pd.concat(trajectories, ignore_index=True) # merge trajectories 
        else:
            df = pd.read_csv(os.path.join('Data', "combined_trajectories.csv"))
        
        # 2. Initialize environment and agent
        env = FlyEnvironment(grid_size = (df.x.min(), df.x.max(), df.y.min(), df.y.max()), source_location=(250000,250000), detection_radius=15000)
        env.reset()
        state_size = len(env.state)  # Define state size based on your environment
        action_size = len(env.action_space)  # Define action size based on your environment
        agent = DQNAgent(state_size = state_size, action_size = action_size, batch_size = BATCH_SIZE, gamma = GAMMA, epsilon = EPSILON, epsilon_min= EPSILON_MIN, epsilon_decay = EPSILON_DECAY, learning_rate = LEARNING_RATE)

        # 3. Run training loop
        if args.training:
            trainnig(agent = agent, env = env, trajectories = trajectories, epochs=EPOCHS)

        # 4. Run inference loop
        if args.inference:
            if args.model_path is not None:
                agent.model.load_state_dict(torch.load(os.path.join(args.model_path)))
                logger.info("Model loaded successfully from {0}".format(args.model_path))
            agent.model.eval()
            mean_reward = evaluate(agent=agent, env=env, num_episodes=10)
        
        #log hyperparameters
        writer.add_hparams({'batch_size': BATCH_SIZE, 'gamma': GAMMA, 'epsilon': EPSILON, 'epsilon_min': EPSILON_MIN, 'epsilon_decay': EPSILON_DECAY, 'learning_rate': LEARNING_RATE}, {'mean_reward': mean_reward})
        writer.flush()
        writer.close()

    except Exception as e:
        logger.error(f"Error: {e}",exc_info=True)

if __name__ == '__main__':
    main()