
# %% Import the libraries
import gym
import collections
from tensorboardX import SummaryWriter

# %% Hyperparameters
ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODE = 20


# %% The agent class
class Agent:

    # The constructor
    def __init__(self):
        # The environement
        self.env = gym.make(ENV_NAME)
        # Reset the state and get the initial observation
        self.state = self.env.reset()
        # Initialize the rewards table
        self.rewards_table = collections.defaultdict(float)
        # Initialize the transitions table
        self.transitions_table = collections.defaultdict(collections.Counter)
        # Initialize the state values table
        self.values_table = collections.defaultdict(float)

    # Function for playing N random steps for gathering experience
    def play_n_random_step(self, count):
        # Iterate through counts
        for _ in range(count):
            # Choose a random action
            action = self.env.action_space.sample()
            # Take the action and get the observation, reward, done, info
            new_state, reward, is_done, info = self.env.step(action)
            # Update the rewards table
            self.rewards_table[(self.state, action, new_state)] = reward
            # Update the transition table
            self.transitions_table[(self.state, action)][new_state] += 1
            # Update the state
            self.state = self.env.reset() if is_done else new_state

    # Function for calculating the action value
    def action_value_calculation(self, state, action):
        # Get the transition counter for given state and action
        target_counts = self.transitions_table[(state, action)]
        # Get the total action execution times for given state
        total_actions = sum(target_counts.values())
        # Initialize the action value
        action_value = 0.0
        # Iterate through each target state and its count
        for target_state, count in target_counts.items():
            # Get the reward 
            reward = self.rewards_table[(state, action, target_state)]
            # Update action value using Bellman equation
            action_value += (count/total_actions)*(reward+GAMMA*self.values_table[target_state])
        return action_value

    # Function for selecting the best action
    def select_action(self, state):
        # Initialize the best action and the best action value
        best_action, best_action_value = None, None
        # Iterate through all possible action
        for i_action in range(self.env.action_space.n):
            # Calculate the action value
            action_value = self.action_value_calculation(state, i_action)
            # If best action value is None OR it's less than the action value
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
            # Choose an action
            action = self.select_action(state)
            # Take action and get the observation, reward, is_done, info
            next_state, rewrad, is_done, info = env.step(action)
            # Update the rewards table
            self.rewards_table[(state, action, next_state)] = rewrad
            # Update the transition table
            self.transitions_table[(state, action)][next_state] += 1
            # Update the total reward
            total_reward += rewrad
            # If terminal state
            if is_done:
                # Break the loop
                break
            # Update the current state
            state = next_state
        return total_reward

    # Function for value  iteration
    def value_iteration(self):
        # Iterate through state space 
        for i_state in range(self.env.observation_space.n):
            # Get the state values
            state_values = [self.action_value_calculation(i_state, i_action) for i_action in range(self.env.action_space.n)]
            # Update the state table
            self.values_table[i_state] = max(state_values)


# %% Execute the program
if __name__ == "__main__":
    # Create the environment
    test_env = gym.make(ENV_NAME)
    # Initialize the agent
    agent = Agent()
    # Summary writer for tensorboard
    writer = SummaryWriter(comment="-v-learning")
    # Initialize the iteration number
    iteration_number = 0
    # Initialize the best reward
    best_reward = 0.0
    # Infinite loop
    while True:
        # Increment the iteration number
        iteration_number += 1
        # Play 100 random episodes
        agent.play_n_random_step(count = 100)
        # Run the value iteration over all state
        agent.value_iteration()
        # Initialize the reward
        reward = 0.0
        # Iterate through test episodes
        for _ in range(TEST_EPISODE):
            # Update the reward
            reward += agent.play_episode(test_env)
        # Divide the reward over TEST_EPISODE
        reward /= TEST_EPISODE
        # Add reward to tensorboard
        writer.add_scalar("reward", reward, iteration_number)
        # If reward is higher than best reward
        if (reward>best_reward):
            # Print the reward and best reward
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            # Update the best reward
            best_reward = reward
            # If reward is higher than 0.8 then it's solved
            if (reward>0.8):
                print("Solved in %d iterations!" % (iteration_number))
                break
    # Close the writer
    writer.close()
