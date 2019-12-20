
# %% Import the libraries
import gym
import collections
from tensorboardX import SummaryWriter

# %% Hyperparameters
ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20

# %% The agent class
class Agent:

    # The constructor
    def __init__(self):
        # The environment
        self.env = gym.make(ENV_NAME)
        # Reset the state and get the initial observation
        self.state = self.env.reset()
        # The rewards table
        self.rewards_table = collections.defaultdict(float)
        # The transitions table
        self.transitions_table = collections.defaultdict(collections.Counter)
        # The value table
        self.values_table = collections.defaultdict(float)

    # Playing N random steps for gathering experience
    def play_n_random_steps(self, count):
        # Iterate through counts
        for _ in range(count):
            # Choose a random action
            action = self.env.action_space.sample()
            # Take the action and get the observation, reward, is_done, info
            next_state, reward, is_done, info = self.env.step(action)
            # Update the reward table
            self.rewards_table[(self.state, action, next_state)] = reward
            # Update the transitions table
            self.transitions_table[(self.state, action)][next_state] += 1
            # Update the current state
            self.state = self.env.reset() if is_done else next_state

    # Function for selecting the best action
    def select_action(self, state):
        # Initialize the best action and the best action value
        best_action, best_action_value = None, None
        # Iterate through each possible action
        for i_action in range(self.env.action_space.n):
            # Get the action value
            action_value = self.values_table[(state, i_action)]
            # If the best action value is none OR it's less than action value
            if (best_action_value is None) or (best_action_value < action_value):
                # Update the best action value
                best_action_value = action_value
                # Update the best action
                best_action = i_action
        return best_action

    # Function for playing one full episode
    def play_episode(self, env):
        # Initialize the total reward
        total_reward = 0.0
        # Reset the state and get the initial observation
        state = env.reset()
        # Infinite loop
        while True:
            # Select the best action
            action = self.select_action(state)
            # Take the action and get the observation, reward, is_done, info
            next_state, reward, is_done, info = env.step(action)
            # Render environment
            #env.render()
            # Update the rewards table
            self.rewards_table[(state, action, next_state)] = reward
            # Update the transitions state
            self.transitions_table[(state, action)][next_state] += 1
            # Update the total reward
            total_reward += reward
            # If terminal state
            if is_done:
                # Break the loop
                break
            # Update the current state
            state = next_state
        return total_reward

    # Function for value iteration
    def value_iteration(self):
        # Iterate through each state
        for i_state in range(self.env.observation_space.n): 
            # Iterate through each action
            for i_action in range(self.env.action_space.n):
                # Initialize the action value
                action_value = 0.0
                # Extract the transition counter for given state and action
                target_counts = self.transitions_table[(i_state, i_action)]
                # Get the total action execution times from current state
                total = sum(target_counts.values())
                # Iterate through each target state and its count
                for target_state, count in target_counts.items():
                    # Get the reward
                    reward = self.rewards_table[(i_state, i_action, target_state)]
                    # Select the best action
                    best_action = self.select_action(target_state)
                    # Update the action value using Bellman Equation
                    action_value += (count/total) * (reward + GAMMA * self.values_table[(target_state, best_action)])
                    # Update the value table
                    self.values_table[(i_state, i_action)] = action_value


# Excecute the program
if __name__ == "__main__":
    # Create the environment
    test_env = gym.make(ENV_NAME)
    # Initialize the agent
    agent = Agent()
    # Summary writer for tensorboard
    writer = SummaryWriter(comment="-q-iteration")
    # Initialize the iteration number 
    iteration_number = 0
    # Initialize the best reward
    best_reward = 0.0
    # Infinite loop
    while True:
        # Increment the iteration number 
        iteration_number += 1
        # Perform 100 random steps for gathering experience
        agent.play_n_random_steps(100)
        # Run the value iteration over all states
        agent.value_iteration()
        # Initialize the reward
        reward = 0.0
        # Iterate through TEST_EPISODE
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        # Divide the reward over TEST_EPISODES
        reward /= TEST_EPISODES
        # Add the reward to tensorboard
        writer.add_scalar("reward", reward, iteration_number)
        # If reward is higher thatn the best_reward
        if (reward > best_reward):
            # Print the reward and best reward
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            # Update the best reward
            best_reward = reward
        # If reward is higher than 0.8 then the episode is solved
        if reward > 0.8:
            # Print the iteration number
            print("Solved in %d iterations!" % (iteration_number))
            # Break the loop
            break
    # Close the writer
    writer.close()
