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
from omegaconf import DictConfig, OmegaConf
import hydra

from torch.utils.tensorboard import SummaryWriter

LOGGER_CONFIG_PATH = "./config/logger_config.yaml"
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
GAMMA = 0.90
LEARNING_RATE = 0.001
EPOCHS = 5
MAX_STEPS = 5000000
BATCH_SIZE = 256
EPSILON = 0.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.95


def data_processing(path: str, version: int = 1, process_data: bool = False) -> list[pd.DataFrame]:
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
    choices = ['stop', 'right', 'up_right', 'up', 'up_left',
               'bottom_left', 'down', 'bottom_right', 'left']
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
    actions = ['stop', 'up', 'down', 'left', 'right',
               'up_right', 'up_left', 'bottom_right', 'bottom_left']
    # Default to 'stop' if none of the conditions match
    df['action_coordinates'] = np.select(conditions, actions, default='stop')

    return df


def compute_iou(actual_trajectory, generated_trajectory):
    """
     Compute the Intersection over Union (IoU) between the actual trajectory of a biological fruit fly
     and a trajectory generated by an RL agent.

     Args:
         actual_trajectory (np.array): Actual trajectory as a numpy array of shape (n, 2) where n is the number of points.
         generated_trajectory (np.array): Generated trajectory as a numpy array of shape (m, 2) where m is the number of points.

     Returns:
         float: The IoU score.
     """
    # Convert trajectories to sets of tuples for easy intersection and union operations
    actual_set = set(tuple(map(tuple, actual_trajectory)))
    generated_set = set(tuple(map(tuple, generated_trajectory)))

    # Calculate intersection and union
    intersection = actual_set.intersection(generated_set)
    union = actual_set.union(generated_set)

    # Compute IoU
    iou = len(intersection) / len(union) if union else 0
    return iou


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
    for episode, trajectory in enumerate(tqdm(trajectories, desc="Training: Episodes", total=len(trajectories))):
        total_reward = 0  # Track total reward per episode
        agent.memory.clear()  # Clear the replay memory for each episode
        for i in tqdm(range(len(trajectory)-1), desc="Rolling Over Trajactory", total=len(trajectory),leave=False ):
            state = trajectory.iloc[i][['x', 'y', 'light']].values
            next_state = trajectory.iloc[i+1][['x', 'y', 'light']].values

            # Assuming you have a way to determine the action and reward from your data
            # trajectory.iloc[i]['action_coordinates']
            action = env.action_space.index(trajectory.iloc[i]['action_theta'])

            # Determine the reward received for the transition
            reward = -1 if trajectory.iloc[i]['light'] == 0 else 1
            # trajectory.iloc[i]['done'] # done =  # Mark done on the last transition
            done = (i == len(trajectory) - 2)

            # Reshape for compatibility with your DQN model input
            state = np.reshape(state, [agent.state_size]).astype(np.float32)
            next_state = np.reshape(
                next_state, [agent.state_size]).astype(np.float32)

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
            writer.add_scalar(
                f"Training: Episode/{episode} || Average Loss/Epoch", avg_epoch_loss, epoch)

        # Optionally reduce epsilon after each episode to reduce exploration over time
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Log results in tensorboard
        avg_episode_loss = np.mean(epoch_losses)
        episode_losses.append(avg_episode_loss)
        episode_rewards.append(total_reward)

        writer.add_scalar("Training || Average Loss/Episode",
                          avg_episode_loss, episode)
        writer.add_scalar("Training || Reward/Episode", total_reward, episode)

        # # Calculate and log the average loss for the epoch
        # avg_epoch_loss = np.mean(epoch_losses)
        # writer.add_scalar('Average Epoch Loss/Trainining', avg_epoch_loss, epoch)

        # Optionally save the model after each epoch
        torch.save(agent.model.state_dict(), os.path.join(
            'models', 'model_episode_{0}_{1}.pth'.format(episode, time.time())))

        writer.flush()
    return episode_losses, episode_rewards


def generate_trajectory(agent, env, init_position=None, num_episodes=10):
    """
    Evaluate the agent's performance on the environment.

    Args:
        agent: The DQNAgent instance.
        env: The environment instance, which should be compatible with the agent.
        num_episodes (int): Number of episodes to evaluate the agent.
        init_position (tuple): Initial position of the agent
    """
    for episode in tqdm(range(num_episodes), desc=f"{navigation} : Evaluating Episodes", total=num_episodes):
        # Rest the environment and get the initial state
        if init_position is None:
            state = env.reset()
        else:
            state = env.reset(x=init_position[0], y=init_position[1])

        # state = init_position #env.reset()  # Initialize the environment and get the initial state
        state = np.reshape(state, [agent.state_size]).astype(np.float32)
        done = False
        total_reward = 0
        steps = 0
        # Assuming the first two elements of state are x and y positions
        step_positions = [state[:2]]
        with tqdm(total=MAX_STEPS, desc="Evaluating Steps", leave=False) as pbar:
            while (not done) and (steps < MAX_STEPS):
                # Select the best action for the current state
                action = agent.act(state)
                # Take the action in the environment
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(
                    next_state, [agent.state_size]).astype(np.float32)

                # Track the position after taking the action
                step_positions.append(next_state[:2])
                state = next_state
                total_reward += reward
                steps += 1
                pbar.update(1)

        # log the total reward for the episode
        writer.add_scalar(
            f"Evaluation: Reward/Episode/{navigation}", total_reward, episode)

        # Optionally visualize the trajectory
        step_positions = np.array(step_positions)
        fig, ax = plt.subplots()

        # ax.plot(step_positions[:, 0], step_positions[:, 1], marker='o')
        # Agent Trjaectory
        ax.plot(step_positions[:, 0], step_positions[:, 1], color='blue',
                marker='o', linestyle='-', linewidth=1, markersize=2, label='Agent Path')

        # Starting position
        ax.scatter(*step_positions[0], color='green', s=100,
                   label='Start', edgecolors='darkgreen', zorder=5)

        # Ending position
        ax.scatter(*step_positions[-1], color='red', s=100,
                   label='End', edgecolors='darkred', zorder=5)

        # Source location
        ax.plot(env.source_location[0], env.source_location[1],
                'r*', markersize=15, label='Source')

        # plot the detection boundary
        detection_circle = plt.Circle(env.source_location, env.detection_radius,
                                      color='r', fill=False, linestyle='--', label='Detection Boundary')
        ax.add_artist(detection_circle)
        # arena_circle = plt.Circle(self.source_location, 80000, color='black', fill=False, linestyle='-', label='Arena Boundary')
        # ax.add_artist(arena_circle)

        ax.set_title(f"Episode {episode+1}: Total Reward = {total_reward}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_xlim(env.x_min, env.x_max)
        ax.set_ylim(env.y_min, env.y_max)
        plt.legend()

        # Log the matplotlib figure in TensorBoard
        writer.add_figure(
            f"Evaluation: Trajectory/{navigation}/Episode_{episode+1} || Reward: {total_reward}", fig, global_step=episode)


def evaluate(agent, env, trajectories, navigation="random_walk"):
    """
    Evaluate the agent's performance on the environment.

    Args:
        agent: The DQNAgent instance.
        env: The environment instance, which should be compatible with the agent.
        num_episodes (int): Number of episodes to evaluate the agent.
    """
    episode_rewards = []
    episode_iou = []

    for episode, test_trajectory in enumerate(tqdm(trajectories, desc=f"Evaluating Trajectories", total=len(trajectories))):

        # Rest the environment and get the initial state
        state = env.reset(
            x=test_trajectory.iloc[0]['x'], y=test_trajectory.iloc[0]['y'], light=test_trajectory.iloc[0]['light'])
        state = np.reshape(state, [agent.state_size]).astype(np.float32)
        done = False
        total_reward = 0
        # # steps = 0
        # Assuming the first two elements of state are x and y positions
        step_positions = [state[:2]]

        # with tqdm(total=len(test_trajectory), desc="Evaluating Steps", leave=False) as pbar:
        #     while (not done) and (steps < MAX_STEPS):
        for index, row in tqdm(test_trajectory.iterrows(), total=test_trajectory.shape[0]):
            # state = row[['x', 'y', 'light']].values.astype(np.float32)

            # Select the best action for the current state
            action = agent.act(state)

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(
                next_state, [agent.state_size]).astype(np.float32)

            # Track the position after taking the action
            step_positions.append(next_state[:2])
            state = next_state
            total_reward += reward
            # steps += 1
            # pbar.update(1)

            if done:
                break

        # log the total reward and IOU for the episode
        predicted_trajectory = np.array(step_positions, dtype=np.float32)
        iou_score = compute_iou(test_trajectory[['x', 'y']].values.astype(
            np.float32), predicted_trajectory)
        writer.add_scalar(
            f"Evaluation: Episode/{navigation}/Reward", total_reward, episode)
        writer.add_scalar(
            f"Evaluation: Episode/{navigation}/IoU_Score", total_reward, iou_score)

        episode_rewards.append(total_reward)
        episode_iou.append(iou_score)

        # Visualize the both geneated and actual trajectory
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        fig.suptitle(
            f"Episode {episode}: Comparison of RL Generated Trjaectory V/S Fruit Fly Generated Trajectory \n Total Reward : {total_reward}, IoU Score: {iou_score}", fontsize=16)

        # Agent Generated Trjaectory
        ax[0].plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], color='blue',
                   marker='o', linestyle='-', linewidth=1, markersize=2, label='Agent Path')

        # Starting position
        ax[0].scatter(*predicted_trajectory[0], color='green', s=100,
                      label='Start', edgecolors='darkgreen', zorder=5)

        # Ending position
        ax[0].scatter(*predicted_trajectory[-1], color='red', s=100,
                      label='End', edgecolors='darkred', zorder=5)

        # Source location
        ax[0].plot(env.source_location[0], env.source_location[1],
                   'r*', markersize=15, label='Source')

        # plot the detection boundary
        detection_circle = plt.Circle(env.source_location, env.detection_radius,
                                      color='r', fill=False, linestyle='--', label='Detection Boundary')
        ax[0].add_artist(detection_circle)
        # arena_circle = plt.Circle(self.source_location, 80000, color='black', fill=False, linestyle='-', label='Arena Boundary')
        # ax.add_artist(arena_circle)

        ax[0].set_title(f"Generated Trajectory: Episode {episode+1}")
        ax[0].set_xlabel("X Position")
        ax[0].set_ylabel("Y Position")
        ax[0].legend()

        # Actual Trajectory
        ax[1].plot(test_trajectory['x'], test_trajectory['y'], color='green',
                   marker='o', linestyle='-', linewidth=1, markersize=2, label='Actual Path')

        # Starting position
        ax[1].scatter(test_trajectory.iloc[0]['x'], test_trajectory.iloc[0]['y'], color='green', s=100,
                      label='Start', edgecolors='darkgreen', zorder=5)

        # Ending position
        ax[1].scatter(test_trajectory.iloc[-1]['x'], test_trajectory.iloc[-1]['y'], color='red', s=100,
                      label='End', edgecolors='darkred', zorder=5)

        # make yellow if light is on
        ax[1].scatter(test_trajectory[test_trajectory['light'] == 1]['x'], test_trajectory[test_trajectory['light'] == 1]['y'], color='yellow', s=100,
                      label='Light On', zorder=5)

        ax[1].set_title(f"Actual Trajectory: Episode {episode+1}")
        ax[1].set_xlabel("X Position")
        ax[1].set_ylabel("Y Position")
        ax[1].legend()

        # Set plot limits
        for axs in ax:
            axs.set_xlim(env.x_min, env.x_max)
            axs.set_ylim(env.y_min, env.y_max)

        # Log the matplotlib figure in TensorBoard
        writer.add_figure(
            f"Evaluation: Trajectory/{navigation}/Episode_{episode+1}", fig, global_step=episode)

        plt.close(fig)  # Close the figure to free memory

    # Calculate and log the average reward over all episodes
    avg_reward = np.mean(episode_rewards)
    avg_iou = np.mean(episode_iou)
    return avg_reward, avg_iou

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    
    try:
        # 1. Initialize environment and agent
        env = FlyEnvironment(grid_size=(83010.0, 370690.0, 42922.0, 332320.0), source_location=(
            268355.0, 187621.0), detection_radius=15000)
        init_position = env.reset()
        # Define state size based on your environment
        state_size = len(env.state)
        # Define action size based on your environment
        action_size = len(env.action_space)
        agent = DQNAgent(state_size=state_size, action_size=action_size, batch_size=int(cfg.hyperparameters.batch_size), gamma=float(cfg.hyperparameters.gamma),
                         epsilon=float(cfg.hyperparameters.epsilon), epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, learning_rate=float(cfg.hyperparameters.learning_rate))

        # 2. Load data and Run training loop
        if cfg.training:
            train_data_path = os.path.join(cfg.data_path, "train")
            # [pd.read_csv(os.path.join('./Data/sample_trajectory.csv'))] #  #
            train_trajectories = data_processing(train_data_path)
            episode_losses, episode_rewards = trainnig(
                agent=agent, env=env, trajectories=train_trajectories, epochs=EPOCHS)
            writer.add_hparams(
                hparam_dict={
                    "batch_size": cfg.hyperparameters.batch_size,
                    "gamma": cfg.hyperparameters.gamma,
                    "epsilon": cfg.hyperparameters.epsilon,
                    'learning_rate': cfg.hyperparameters.learning_rate
                },
                metric_dict={
                    "train_reward": np.mean(episode_rewards),
                    "train_loss": np.mean(episode_losses),
                },
            )
            

        # 3. Load data and Run Evaluation loop
        if cfg.evaluation:
            test_data_path = os.path.join(cfg.data_path, "test")
            # pd.read_csv(os.path.join('Data', "combined_trajectories.csv"))
            test_trajectories = data_processing(test_data_path)
            if cfg.model_path is not None:
                agent.model.load_state_dict(
                    torch.load(os.path.join(cfg.model_path)))
                logger.info(
                    "Model loaded successfully from {0}".format(cfg.model_path))
            agent.model.eval()
            avg_mean_reward, avg_iou_score = evaluate(
                agent=agent, env=env, trajectories=test_trajectories)
            logger.info(
                f"Mean reward is: {avg_mean_reward}, IoU Score is: {avg_iou_score}")
            writer.add_hparams(
                hparam_dict={
                    "batch_size": cfg.hyperparameters.batch_size,
                    "gamma": cfg.hyperparameters.gamma,
                    "epsilon": cfg.hyperparameters.epsilon,
                    'learning_rate': cfg.hyperparameters.learning_rate
                },
                metric_dict={
                    "test_avg_reward": avg_mean_reward,
                    "test_avg_iou_score": avg_iou_score
                }
            )

        # 5. Generate Trajectory
        # generate_trajectory(agent=agent, env=env, init_position=init_position, num_episodes=10, navigation="model_based")

        # log hyperparameters
        # writer.add_hparams({'batch_size': BATCH_SIZE, 'gamma': GAMMA, 'epsilon': EPSILON, 'epsilon_min': EPSILON_MIN, 'epsilon_decay': EPSILON_DECAY, 'learning_rate': LEARNING_RATE}, {'mean_reward': mean_reward})
         # add the hyperparameters and metrics to TensorBoard

        writer.flush()
        writer.close()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
