

# %% Import the libaries
import gym
import collections
from tensorboardX import SummaryWriter


# %% Hyperparameters
ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODE = 20


# %% The agent class
class Agent:

    # Constructor function
    def __init__(self):
        # The environment
        self.env = gym.make(ENV_NAME)
        # Get the initial state
        self.state = self.env.reset()
        # Initialize the reward table   # {("source state", "action", "target state"): immediate reward}
        self.reward = collections.defaultdict(float)
        # Initialize the transitions table    # {("state", "action"): {"target state":  "count of times we seen"}}
        self.transits = collections.defaultdict(collections.Counter)
        # Initialize the value table      # {"state": "value of the state"}
        self.values = collections.defaultdict(float)

    # Function for playing N random steps and gathering experience
    def play_n_random_steps(self, count):
        # Iterate through count
        for _ in range(count):
            # Sample a random action
            action = self.env.action_space.sample()
            # Take action and get the observation (new state), reward, done, info
            new_state, reward, done, info = self.env.step(action)
            # Update the reward table
            self.reward[(self.state, action, new_state)] = reward
            # Update the transition table
            self.transits[(self.state, action)][new_state] += 1
            # Update the current state
            self.state = self.env.reset() if done else new_state

    # Function for calculating the action value
    def action_value(self, state, action):
        # Extract the transition counter for current state and action
        transition_counts = self.transits[(state, action)]
        # Get the total number of times we took action in the given state
        total = sum(transition_counts.values())
        # Initialize the action value
        action_value = 0.0
        # Iterate through target states and their counts
        for target_state, count in transition_counts.items():
            # Get the reward from reward table
            reward = self.reward[(state, action, target_state)]
            # Calculat the action value using Bellman Equation
            action_value += (count/total) * (reward + GAMMA * self.values[target_state])
        return action_value

    # Function for selecting the best action
    def select_action(self, state):
        # Initialize the best action and best value
        best_action, best_value = None, None
        # Iterate through action space numbers
        for action in range(self.env.action_space.n):
            # Calculate the action value
            action_value = self.action_value(state, action)
            # If best_value is none or less than action value
            if (best_value is None) or (best_value < action_value):
                # Update the best_value
                best_value = action_value
                # Update the best_action
                best_action = action
        return best_action

    # Function for playing one full episode
    def play_episode(self, env):
        # Initialize the total reward
        total_reward = 0.0
        # Reset the state 
        state = env.reset()
        # Infinite loop
        while True:
            # Choose the action to take
            action = self.select_action(state)
            # Take action and get the observation (next_state), reward, done, info
            next_state, reward, done, info = env.step(action)
            # Update the reward
            self.reward[(state, action, next_state)] = reward
            # Update the transition table
            self.transits[(state, action)][next_state] += 1
            # Update the toal reward 
            total_reward += reward
            # If terminal state
            if done:
                break
            # Update the state
            state = next_state
        return total_reward

    # Function for value iteration
    def value_iteration(self):
        # Iterate through state space numbers
        for state in range(self.env.observation_space.n):
            # Calculate the values
            state_values = [self.action_value(state, action) for action in range(self.env.action_space.n)]
            # Update the state table
            self.values[state] = max(state_values)


# %% Execute the program
if __name__ == "__main__":
    # Create the environment
    test_env = gym.make(ENV_NAME)
    # Intialize the agent
    agent = Agent()
    # Summary write of data for TensorBoard
    writer = SummaryWriter(comment = "-v-learning")
    # Initialize the iteration number
    iter_no = 0
    # Initialize the best reward = 0.0
    best_reward = 0.0
    # Infinite loop
    while True:
        # Increment the iteration number
        iter_no += 1
        # Perform 100 random steps to fill the reward and transition table with data
        agent.play_n_random_steps(100)
        # Run value iteration
        agent.value_iteration()
        # Initialize the reward
        reward = 0.0
        # Iterate through the TEST_EPISODE (episode number)
        for _ in range(TEST_EPISODE):
            # Update the reward
            reward += agent.play_episode(test_env)
        # Divide summary to TEST_EPISODE
        reward /= TEST_EPISODE
        # Write the value for reward into TensorBoard
        writer.add_scalar("reward", reward, iter_no)
        # If reward is higher than best_reward
        if reward >  best_reward:
            # Print the reward and the best reward
            print("Best Reward Update from {} to {}".format(best_reward, reward))
            # Updaet the best reward
            best_reward = reward
            # If reward is higher than 0.8 then break the loop
            if reward > 0.8:
                print("Solved in {} Iterations!".format(iter_no))
                break
    # Close the writer
    writer.close()
