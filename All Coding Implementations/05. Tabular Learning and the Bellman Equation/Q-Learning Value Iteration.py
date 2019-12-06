

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

    # The constructor function
    def __init__(self):
        # The environment
        self.env = gym.make(ENV_NAME)
        # The initial state
        self.state = self.env.reset()
        # Initialize the reward table   # {("source state", "action", "target state"): immediate reward}
        self.rewards = collections.defaultdict(float)
        # Initialize the transitions table    # {("state", "action"): {"target state":  "count of times we seen"}}
        self.transitions = collections.defaultdict(collections.Counter)
        # Initialize the (action) value table      # {("state", "action"): "value of the Q-function"}
        self.values = collections.defaultdict(float)

    # Function for playing N random steps (for gathering initial experience)
    def play_n_random_steps(self, count):
        # Iterate through count
        for _ in range(count):
            # Sample a random action
            action = self.env.action_space.sample()
            # Take the given action and get the observation (next state), reward, done, info
            next_state, reward, done, info = self.env.step(action)
            # Update the rewards table
            self.rewards[(self.state, action, next_state)] = reward
            # Update the transitions table
            self.transitions[(self.state, action)][next_state] += 1
            # Update the state
            self.state = self.env.reset() if done else next_state

    # Function for selecting the best action
    def select_action(self, state):
        # Initialize the best action and best value
        best_action, best_action_value = None, None
        # Iterate through the action space number
        for action in range(self.env.action_space.n):
            # Fetch the action value
            action_value = self.values[(state, action)]
            # If best_action_value is none or less than action_value
            if (best_action_value is None) or (best_action_value < action_value):
                # Update the best_action_value
                best_action_value = action_value
                # Update the best action
                best_action = action
        return best_action

    # Function for playing one full episode
    def play_episode(self, env):
        # Initialize the total reward
        total_reward = 0.0
        # Get the initial state
        state = env.reset()
        # Infinite loop
        while True:
            # Select the best action
            action = self.select_action(state)
            # Take the given action and get the observation (next state), reward, done, info
            next_state, reward, done, info = env.step(action)
            # Update the rewards table
            self.rewards[(state, action, next_state)] = reward
            # Update the transitions table
            self.transitions[(state, action)][next_state] += 1
            # Update the total reward
            total_reward += reward
            # If terminal state
            if done:
                # Break the loop
                break
            # Update the state
            state = next_state
        return total_reward

    # Function for calculating the value iteration
    def value_iteration(self):
        # Iterate through state space numbers
        for state in range(self.env.observation_space.n):
            # Iterate through action space numbers
            for action in range(self.env.action_space.n):
                # Initialize the action value
                action_value = 0.0
                # Fetch the transition count for given state and action
                transition_counts = self.transitions[(state, action)]
                # Get the total count
                total = sum(transition_counts.values())
                # Iterate through target states and counters
                for target_state, count in transition_counts.items():
                    # Fetch the reward
                    reward = self.rewards[(state, action, target_state)]
                    # Select the best action
                    best_action = self.select_action(target_state)
                    # Update the action value using Bellman Equation
                    action_value += (count / total) * (reward + GAMMA * self.values[(target_state, best_action)])
                    # Update the value table
                    self.values[(state, action)] = action_value

# %% Execute the program
if __name__ == "__main__":
    # Initialize the environment
    test_env = gym.make(ENV_NAME)
    # Initialize the agent
    agent = Agent()
    # Summary writer for tensor board
    writer = SummaryWriter(comment="-q-iteration")
    # Initialize the iterateion number
    iter_no = 0
    # Initialize the best reward
    best_reward = 0.0
    # Infinite loop
    while True:
        # Increment the iteration number
        iter_no += 1
        # Perform 100 random steps for filling up the reward and transitions table with fresh data
        agent.play_n_random_steps(100)
        # Run the value iteration function
        agent.value_iteration()
        # Initialize the reward
        reward = 0.0
        # Iterate through TEST_EPISODES
        for _ in range(TEST_EPISODES):
            # Update the reward
            reward += agent.play_episode(test_env)
        # Divide the reward by TEST_EPISODES
        reward /= TEST_EPISODES
        # Write the value for reward into TensorBoard
        writer.add_scalar("reward", reward, iter_no)
        # If reward is higher than best_reward
        if reward > best_reward:
            # Print the reward and best reward
            print("Best Reward Updated from {} to {}!".format(best_reward, reward))
            # Update the best reward
            best_reward = reward
        # If reward is higher than 0.8 then it is SOLVED
        if reward > 0.8:
            print("SOLVED IN {} ITERATIONS!".format(iter_no))
            break
    # Close the writer
    writer.close()
