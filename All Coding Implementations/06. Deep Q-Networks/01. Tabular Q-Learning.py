
# %% Import the libraries
import gym
import collections
from tensorboardX import SummaryWriter

# %% Hyperparameters
ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2        # Learning rate
TEST_EPISODES = 20


# %% The agent class
class Agent:

    # Constructor function
    def __init__(self):
        # Make the environment
        self.env = gym.make(ENV_NAME)
        # The initial state
        self.state = self.env.reset()
        # Initialize an empty table for Q(s, a)
        self.values = collections.defaultdict(float)

    # Function for gaining the next transition (e.g. S, A, R, S')
    def sample_env(self):
        # Choose a random action
        action = self.env.action_space.sample()
        # Get the old state
        old_state = self.state
        # Take action and get the observation, reward, is_done, info
        next_state, reward, is_done, info = self.env.step(action)
        # Update the current state
        self.state = self.env.reset() if is_done else next_state
        # Return (S, A, R, S')
        return (old_state, action, reward, next_state)

    # Function for finding the best action from current 
    def best_action(self, state):
        # Initialize the best action and best action value
        best_action, best_action_value = None, None
        # Iterate through action space
        for i_action in range(self.env.action_space.n):
            # Fetch the action value
            action_value = self.values[(state, i_action)]
            # If best action value is none or it's less than current action value
            if (best_action_value is None) or (best_action_value < action_value):
                # Update the best action value
                best_action_value = action_value
                # Update the best action
                best_action = i_action
        return best_action_value, best_action

    # Function for updating the action value table
    def update_value(self, state, action, reward, next_state):
        # Find the best action to take
        best_action_value, _ = self.best_action(next_state)
        # Calculate the next action vaue using Bellman Approximation
        new_value = reward + GAMMA * best_action_value
        # Fetch the old action value
        old_value = self.values[(state, action)]
        # Use "blending" technique to update the action value
        self.values[(state, action)] = (1 - ALPHA) * old_value + ALPHA * new_value

    # Function for playing one full episode
    def play_episode(self, env):
        # Initialize the total reward
        total_reward = 0.0
        # Get the initial observation
        state = env.reset()
        # Infinite loop
        while True:
            # Choose the best action to take
            _, best_action = self.best_action(state)
            # Take the action and get the observation, reward, is_done, info
            next_state, reward, is_done, info = env.step(best_action)
            # Add the reward to total reward
            total_reward += reward
            # If terminal state
            if is_done:
                # Break the loop
                break
            # Update the current state
            state = next_state
        return total_reward


# %% Execute the program
if __name__ == "__main__":
    # Create the environment
    test_env = gym.make(ENV_NAME)
    # Initialize the agent
    agent = Agent()
    # Initialize the summary writer
    writer = SummaryWriter(comment = "-q-learning")
    # Initialize the iteration number
    iter_num = 0
    # Initialize the best reward
    best_reward = 0.0
    # Infinite loop
    while True:
        # Increment the iteration number 
        iter_num += 1
        # Optain the next transition
        state, action, reward, next_state = agent.sample_env()
        # Update the action value table
        agent.update_value(state, action, reward, next_state)
        # Initialize the reward
        reward = 0.0
        # Iterate through TEST_EPISODE
        for _ in range(TEST_EPISODES):
            # Update the reward by playing one full episode
            reward += agent.play_episode(test_env)
        # Divide reward by TEST_EPISODE
        reward /= TEST_EPISODES
        # Add the reward to tensorboard
        writer.add_scalar("reward", reward, iter_num)
        # If reward is higher than best reward
        if reward > best_reward:
            # Print the reward and best reward
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            # Update the best reward
            best_reward = reward
        # If reward is higher than 0.8 then it's SOLVED
        if (reward > 0.8):
            # Print the iteration number that is solved
            print("Solved in %d iterations!" % (iter_num))
            # Break the loop
            break
    # Close the writer
    writer.close()


#%%
