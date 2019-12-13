
# %% Import the libraries
import gym
from collections import namedtuple
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter



# %% Hyper parameters
HIDDEN_SIZE = 128   # Count of neurons
BATCH_SIZE = 16
PERCENTILE = 70     # For reward boundary
LEARNING_RATE = 0.01



# %% Neural network class
class Net(nn.Module):

    # The constructor
    def __init__(self, observation_size, hidden_size, n_actions):
        """
        Constructor funtion.

        PARAMETERS
        ===========================
            - obs_size: Observation size.
            - hidden_size: Number of neurons in the hidden layers.
            - n_actions: Number of actions.
        """
        # Call the parent's constructor to initialize itself
        super(Net, self).__init__()
        # Create a sequential layers with layers
        self.net = nn.Sequential(nn.Linear(in_features = observation_size, out_features = hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(in_features = hidden_size, out_features = n_actions))

    # Forward pass function
    def forward(self, x):
        """
        Function for forward pass.

        PARAMETERS
        ===========================
            - x: Input data

        RETURNS
        ===========================
            - Data after forward pass
        """
        return self.net(x)



#%% Nametuple for episode step (storing observation and action)
EpisodeStep = namedtuple("EpisodeStep", field_names = ["observation", "action"])



# %% Nametuple for episode (storing episode step and reward)
Episode = namedtuple("Episode", field_names = ["reward", "steps"])



# %%  Function for generating batches
def iterate_batches(env, net, batch_size):
    """
    Function for creating episode batches.j

    PARAMETERS
    ===========================
        - env: The environment 
        - net: The neural network
        - batch_size: The size of batches

    RETURNS
    ===========================
        - Yielding batches
    """
    # Initialize a list for a batch
    batch = []
    # Initialize the episode reward
    episode_reward = 0.0
    # Initialize a list for episode steps
    episode_steps = []
    # Reset the environment and get the first observation
    observations = env.reset()
    # Initialize the softmax function
    softmax_function = nn.Softmax(dim = 1)
    # Infinite loop
    while True:
        # Convert the observation to a tensor
        observations_v = torch.FloatTensor([observations])
        # Forward pass + Apply the softmax
        action_probabilities_v = softmax_function(net.forward(observations_v))
        # Get the action probabilities data + Convert them to array + Get the first dimensional of it (for getting the action probabilities)
        action_probabilities = action_probabilities_v.data.numpy()[0]
        # Random sample of action
        action = np.random.choice(len(action_probabilities), p = action_probabilities)
        # Render environment
        env.render()
        # Take the action and get the next observation, reward, terminal state
        next_observations, reward, is_done, _ = env.step(action)
        # Add the reward to episode's total reward
        episode_reward += reward
        # Add the (observation, action) to episode step
        episode_steps.append(EpisodeStep(observation = observations, action = action))
        # If terminal state
        if is_done:
            # Add the finalize episode to batch
            batch.append(Episode(reward = episode_reward, steps = episode_steps))
            # Reset the total episode reward
            episode_reward = 0.0
            # Reset the episode steps
            episode_steps = []
            # Reset the environment and get the next observation
            next_observations = env.reset()
            # If length of batch reached to batch size
            if len(batch) == batch_size:
                # Yield the batch
                yield batch
                # Reset the batch list
                batch = []
        # Update the current observation
        observations = next_observations



# %% Function for filtering batches to get the elite episodes
def filter_batch(batch, percentile):
    """
    Filtering the batches to get the elite episodes. We use the boundary reward (e.g. using percentile value) for filteration.

    PARAMETERS
    ===========================
        - betch: Batches to filter.
        - percentile: Value for boundary reward.

    RETURNS
    ===========================
        - train_obs_v: The training observations.
        - train_act_v: The training actions.
        - reward_bound: The boundary of reward (used for TensorBoard only).
        - reward_mean: The reward mean (used for TensorBoard only).
    """
    # Get the reward for each batch
    rewards = list(map(lambda s: s.reward, batch))
    # Obtain the reward boundary
    reward_boundary = np.percentile(rewards, percentile)
    # Calculate the reward mean
    reward_mean = float(np.mean(rewards))
    # Initialize the training observations and actions
    training_observations, training_actions = [], []
    # Iterate through episodes in the batch
    for i_episode in batch:
        # If reward is below reward boundary
        if i_episode.reward < reward_boundary:
            # Go back to start of loop
            continue
        # Populate the training observations with elite episodes
        training_observations.extend(map(lambda step: step.observation, i_episode.steps))
        # Populate the training actions with elite episodes
        training_actions.extend(map(lambda step: step.action, i_episode.steps))
    # Convert the training observations and action to tensors
    training_observations_v, training_actions_v = torch.FloatTensor(training_observations), torch.LongTensor(training_actions)
    # Return 
    return training_observations_v, training_actions_v, reward_boundary, reward_mean



# %% Execute the main program
if __name__ == "__main__":
    # make the CartPole-v0 environment
    env = gym.make("CartPole-v0")
    # Get the observation size
    observation_size = env.observation_space.shape[0]
    # Get the number of actions
    num_actions = env.action_space.n
    # Initialize the network
    net = Net(observation_size = observation_size, hidden_size = HIDDEN_SIZE, n_actions = num_actions)
    # Initialize the cross entropy loss
    objective = nn.CrossEntropyLoss()
    # Initialize the optimizer
    optimizer = optim.Adam(params = net.parameters(), lr = LEARNING_RATE)
    # Initialize the summary writer for TensorBoard
    writer = SummaryWriter(comment = "-cartpole")
    # Iterate through batches
    for index, i_episode in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # Filter the episode to get the elite episode
        training_observations_v, training_actions_v, reward_boundary, reward_mean = filter_batch(i_episode, PERCENTILE)
        # Reset the optimizer's gradient
        optimizer.zero_grad()
        # Forward pass
        action_scores_v = net(training_observations_v)
        # Calculate the loss
        loss_v = objective(action_scores_v, training_actions_v)
        # Backward propagation (calculate gradients)
        loss_v.backward()
        # Use optimizer to adjust the network
        optimizer.step()
        # Print some information for monitoring
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (index, loss_v.item(), reward_mean, reward_boundary))
        # Add values to TensorBoard
        writer.add_scalar("loss", loss_v.item(), index)
        writer.add_scalar("reward_boundary", reward_boundary, index)
        writer.add_scalar("reward_mean", reward_mean, index)
        # Stop training if reward mean is bigger than 199
        if reward_mean > 199:
            print("Solved!")
            break
    # Close the writer
    writer.close()

# TensorBoard:
# tensorboard --logdir runs --host localhost
