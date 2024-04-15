import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt


class FlyEnvironment(gym.Env):
    """
    First, we need to define the environment, which includes the state space, action space, and the mechanism to transition from one state to another given an action. The environment also needs to provide a reward signal based on the current state or the transition just made.
    
    usage:
        # load trajectory data
        df = pd.read_csv('./Data/processed/combined_trajectories.csv')

        # Initializing and using the environment
        env = FlyEnvironment(grid_size = (df.x.min(), df.x.max(), df.y.min(), df.y.max()), source_location=(250000,250000), detection_radius=10000)
        curent_state = env.reset()
        env.render()
        
    NOTE: 
    # Define the environment:
        # State: [f, df, w] where these values can be derived from kinematic parameters such as slip, thrust, yaw, etc.
        # Action: [stop, curved walk, sharp turn]
        # Reward: A function of distance from the odor source and plume encounter rate.
    """
    def __init__(self, grid_size, source_location, detection_radius):
        """
        Initializes the environment.
        
        grid_size: Size of the squared grid.
        source_location: Tuple (x, y) indicating the source location.
        detection_radius: Radius in which the light condition turns on, indicating proximity to the source.
        """
        self.x_min, self.x_max, self.y_min, self.y_max = grid_size[0], grid_size[1], grid_size[2], grid_size[3]
        self.source_location = source_location
        self.detection_radius = detection_radius
        self.state = self.reset()  # To be initialized as (x, y, light_condition)
        self.max_magnitude = 1000
        # self.action_effects = {
        #     0: (0, 0),
        #     1: (0, 10),
        #     2: (0, -10),
        #     3: (-10, 0),
        #     4: (10, 0),
        #     5: (10, 10),
        #     6: (-10, 10),
        #     7: (10, -10), # use left sharp turn formula 
        #     8: (-10, -10), # use left sharp turn formula
        # }
        self.action_space = ['stop', 'up', 'down', 'left', 'right', 'up_right', 'up_left', 'bottom_right', 'bottom_left']

        # Define angle ranges for each action (in radians)
        self.action_angle_ranges = {
            0: (0, 0),
            1: (3*np.pi/8, 5*np.pi/8),
            2: (-5*np.pi/8, -3*np.pi/8),
            3: ((7*np.pi/8, np.pi), (-np.pi, -7*np.pi/8)),  # Special case for 'left'
            4: (-np.pi/8, np.pi/8),
            5: (np.pi/8, 3*np.pi/8),
            6: (5*np.pi/8, 7*np.pi/8),
            7: (-3*np.pi/8, -np.pi/8),
            8: (-7*np.pi/8, -5*np.pi/8),
        }

    def reset(self, x=None, y=None, light=0):
        """
        Resets the environment to an initial state.
        """
        if x is None or y is None:
            x = np.random.randint(low = self.x_min, high = self.x_max)
            y = np.random.randint(low = self.y_min, high = self.y_max)
        self.state = (x, y, light)
        return self.state

    def is_near_to_source(self, position):
        """
        Determines if the given position is within the detection radius of the source.
        """
        distance_to_source = np.linalg.norm(np.array(position) - np.array(self.source_location))
        if distance_to_source < self.detection_radius:
            return 1
        else:
            return 0
    
    def step(self, action):
        """
        Executes the given action and returns the new state, reward, and done flag.

        """
        x, y, light_condition = self.state

        # Adjust for 'left' action which spans across -π to π
        if action == 3:  # Assuming 'left' corresponds to action index 3
            # Handle 'left' by randomly choosing an angle from one of its two ranges
            if np.random.rand() > 0.5:
                angle_range = (7*np.pi/8, np.pi)
            else:
                angle_range = (-np.pi, -7*np.pi/8)
        else:
            angle_range = self.action_angle_ranges[action]

        angle_min, angle_max = angle_range

        if action != 0:
            # Select a random angle within the action's range
            angle = np.random.uniform(angle_min, angle_max)

            # Select a random magnitude within the defined range
            magnitude = np.random.uniform(1, self.max_magnitude)

            # Calculate dx and dy based on the angle and magnitude
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
        else:
            dx, dy = 0, 0

        # Update position with boundaries consideration
        # dx, dy = self.action_effects[action]
        new_x = np.clip(x + dx, self.x_min, self.x_max - 1).astype(int)
        new_y = np.clip(y + dy, self.y_min, self.y_max - 1).astype(int)
        new_light_condition = self.is_near_to_source((new_x, new_y))
        self.state = (new_x, new_y, new_light_condition)

        reward = self.get_reward(new_x, new_y, new_light_condition)
        done = self.is_done(new_x, new_y)
        return self.state, reward, done, {}

    def get_reward(self, x, y, light_condition):
        """
        Defines the reward function based on the current state.
        """
        # Reward conditions based on proximity to source and light condition
        if light_condition:
            distance_to_source = np.linalg.norm(np.array((x, y)) - np.array(self.source_location))
            return self.detection_radius - distance_to_source  # Closer to source => higher reward
        return -1  # Encourage moving towards the light

    def is_done(self, x, y):
        """
        Determines if the episode is done.
        """
        return (x, y) == self.source_location

    def render(self):
        """
        Renders the current state of the environment.
        """
        fig,ax = plt.subplots(figsize=(15, 15))
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.plot(self.source_location[0], self.source_location[1], 'r*', markersize=15, label='Source')
        ax.plot(self.state[0], self.state[1], 'b^', markersize=10, label='Fly')
        detection_circle = plt.Circle(self.source_location, self.detection_radius, color='r', fill=False, linestyle='--', label='Detection Boundary')

        # arena_circle = plt.Circle(self.source_location, 80000, color='black', fill=False, linestyle='-', label='Arena Boundary')

        ax.add_artist(detection_circle)
        # ax.add_artist(arena_circle)
        ax.grid(which='both')
        plt.legend()
        plt.show()

# if __name__ == '__main__':
    
#     # load trajectory data
#     df = pd.read_csv('./Data/processed/combined_trajectories.csv')

#     # Initializing and using the environment
#     env = FlyEnvironment(grid_size = (df.x.min(), df.x.max(), df.y.min(), df.y.max()), source_location=(250000,250000), detection_radius=10000)
#     curent_state = env.reset()
#     env.render()