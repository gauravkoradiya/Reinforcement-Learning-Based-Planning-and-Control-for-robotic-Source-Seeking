import gradio as gr
import torch
from agent import DQNAgent
from model import DQN
from environment import FlyEnvironment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assume the agent and environment setup code is available and imported

def run_agent(start_x, start_y, source_x, source_y, model_path):
    # Load the trained model
    state_size = 3  # Update based on your model
    action_size = 9  # Update based on your model
    agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon=0)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    
    # Set up the environment with the provided start and source locations
    env = FlyEnvironment(grid_size=(-500, 500, -500, 500), source_location=(source_x, source_y), detection_radius=50)
    env.state = np.array([start_x, start_y, 0])  # Update based on your environment setup
    
    # Run the agent in the environment
    state = env.reset_to_location(start_x, start_y)
    done = False
    steps = 0
    MAX_STEPS = 200  # Limit the number of steps for demonstration
    positions = [(start_x, start_y)]
    
    while not done and steps < MAX_STEPS:
        action = agent.act(np.array(env.state))
        next_state, _, done, _ = env.step(action)
        positions.append((next_state[0], next_state[1]))
        steps += 1
    
    # Plot the trajectory
    positions = np.array(positions)
    fig, ax = plt.subplots()
    ax.plot(positions[:, 0], positions[:, 1], '-o', label='Agent Path')
    ax.plot(source_x, source_y, 'r*', markersize=15, label='Source')
    ax.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agent Trajectory')
    plt.grid(True)
    
    return fig

# Gradio interface
iface = gr.Interface(
    fn=run_agent,
    inputs=[gr.Number(label="Start X"),
            gr.Number(label="Start Y"),
            gr.Number(label="Source X"),
            gr.Number(label="Source Y"),
            gr.File(file_count="single", label="Model File")],
    outputs="plot",
    title="RL-Based Fruit Flies Demonstration",
    description="A live demonstration of reinforcement learning-based fruit flies navigating towards a source.",
    article="""
    <h2>About Us</h2>
    <p>This project demonstrates the application of reinforcement learning to mimic the navigation behavior of fruit flies towards a source...</p>
    """
)

# Run the app
if __name__ == "__main__":
    iface.launch()
