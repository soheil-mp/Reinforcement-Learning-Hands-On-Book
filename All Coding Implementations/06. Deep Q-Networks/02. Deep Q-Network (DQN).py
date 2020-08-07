
# %% Import the libraries
import numpy as np
import collections
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from DQN import wrappers
from DQN import dqn_model


# %% Hyperparameters
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
DEVICE_TYPE = "cpu"
MEAN_REWARD_BOUND = 19.5                    # Reward boundary; If reach to this amount, we will stop training

GAMMA = 0.99                                # Gamma value in Bellman Approximation
BATCH_SIZE = 32                             # Batch size sampled from replay buffer
REPLAY_SIZE = 10000                         # Maximum capacity of the replay buffer
REPLAY_START_SIZE = 10000                   # Count of frames before starting we start training
LEARNING_RATE = 1e-4                        # Learning rate for optimizer
SYNC_TARGET_FRAMES = 1000                   # Frequency we sync the weight's of the network with target network

EPSILON_START = 1.0                         # Start epsilon
EPSILON_DECAY_LAST_FRAME = 10**5            # Decay epsilon for total of 100'000 frames
EPSILON_FINAL = 0.02                        # End epsilon


# %% Initialize the experience replay buffer for keeping the transitions obtained from environment
Experience = collections.namedtuple(typename = "Experience", field_names = ["state", "action", "reward", "is_done", "new_state"])


# %% Class for experience replay buffer
class ExperienceBuffer:

    # Constructor
    def __init__(self, buffer_capacity):
        # Initialize the buffer to have a limited capacity
        self.buffer = collections.deque(maxlen = buffer_capacity)

    # Function for returning the length
    def __len__(self):
        return len(self.buffer)

    # Function for appending experience into buffer
    def append(self, experience):
        self.buffer.append(experience)

    # Function for sampling from buffer
    def sample(self, batch_size):
        # Create a list of random indices
        indices = np.random.choice(len(self.buffer), batch_size, replace = False)
        # Get the indices from buffer
        indices_value = [self.buffer[index] for index in indices]
        # Repack the indices value
        states, actions, rewards, is_dones, next_states = zip(*indices_value)
        # Convert to numpy array
        output = np.array(states), \
                 np.array(actions), \
                 np.array(rewards, dtype = np.float32), \
                 np.array(is_dones, dtype = np.float32), \
                 np.array(next_states)
        return output


# %% Class for agent
class Agent:

    # Constructor
    def __init__(self, env, experience_buffer):
        # Initialize the environment
        self.env = env
        # Initialize the experience buffer 
        self.experience_buffer = experience_buffer
        # Reset the environment and get the initial observation
        self.state = self.env.reset()
        # Initialize the total reward
        self.total_reward = 0.0

    # Function for playing one step
    def play_step(self, net, epsilon = 0.0, device = DEVICE_TYPE):
        # Initialize the done reward
        done_reward = None
        # If a random number is LESS than epsilon
        if np.random.random() < epsilon:
            # Take a random action in action space
            action = env.action_space.sample()
        # If a random numbrt is MORE than epsilon
        else:
            # Convert states into array
            state_array = np.array([self.state], copy = False)
            # Convert states array into tensor
            state_tensor = torch.tensor(state_array).to(device)
            # Forward propagation
            q_values_tensor = net(state_tensor)
            # Get the maximum of Q-Values which is the action to take
            _, action_tensor = torch.max(q_values_tensor, dim = 1)
            # Get the action in integer form
            action = int(action_tensor.item())
        # Take action and get (new_state, reward, is_done, info)
        new_state, reward, is_done, info = self.env.step(action)
        # Add reward to total reward
        self.total_reward += reward
        # Create the experience
        exp = Experience(self.state, action, reward, is_done, new_state)
        # Appedn the experience into the buffer
        self.experience_buffer.append(exp)
        # Update the current state
        self.state = new_state
        # If terminal state
        if is_done:
            # Assign the total reward to done reward
            done_reward = self.total_reward
            # Reset the environment
            self.state = self.env.reset()
            # Make the total reward to 0
            self.total_reward = 0.0
        return done_reward


# %% Function for calculating the loss
def calculate_loss(batch, net, target_net, device = DEVICE_TYPE):
    # Repack the batch
    states, actions, rewards, is_dones, next_states = batch
    # Convert states array to tensors
    states_tensor = torch.tensor(states).to(device)
    # Convert actions array into tensors
    actions_tensor = torch.tensor(actions).to(device)
    # Convert rewards array into tensors
    rewards_tensor = torch.tensor(rewards).to(device)
    # Convert is_dones into bytes tensors
    is_dones_mask = torch.ByteTensor(is_dones).to(device)
    # Convert next states array into tensors
    next_states_tensor = torch.tensor(next_states).to(device)
    # Forward propagation
    out = net(states_tensor)
    # Get the state-action values
    state_action_values = out.gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
    # Forward propagation for target network + Get the maximum value
    next_state_values = target_net(next_states_tensor).max(1)[0]
    # next state values for the last step should zero
    next_state_values[is_dones_mask] = 0.0
    # Detach next state values in order to prevent gradient frmo flowing into the neural network
    next_state_values = next_state_values.detach()
    # Calculate the Bellman Approximation
    expected_state_action_values = rewards_tensor + GAMMA * next_state_values
    # Calculate the MSE loss
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    return loss


# %% Execute the program
if __name__ == "__main__":
    # Set the device type
    device = torch.device(DEVICE_TYPE)
    # Create an environment
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    # Initialize the network
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    # Initialize the target network
    target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    # Initialize th writer for tensorboard
    writer = SummaryWriter(comment = "-" + DEFAULT_ENV_NAME)
    # Initialize the experience replay buffer
    buffer = ExperienceBuffer(REPLAY_SIZE)
    # Initialize the agent
    agent = Agent(env, buffer)
    # Set the epsilon
    epsilon = EPSILON_START
    # Initialize the adam optimizer
    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)
    # Initialize the total reward
    total_rewards = []
    # Initialize the frame index
    frame_index = 0
    # Initialize the frame for tracking the speed
    frame_ts = 0
    # Start the time ts
    ts = time.time()
    # Initialize the best mean reward
    best_mean_reward = None
    # Infinite loop
    while True:
        # Increment the frame index
        frame_index += 1
        # Update the epsilon
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_index / EPSILON_DECAY_LAST_FRAME)
        # Perform a step and get the final
        reward = agent.play_step(net, epsilon, device = device)
        # If there is reward
        if reward is not None:
            # Append the reward into the total reward
            total_rewards.append(reward)
            # Calculate the speed
            speed = (frame_index - frame_ts) / (time.time() - ts)
            # Update the frame
            frame_ts = frame_index
            # Update the current time
            ts = time.time()
            # Get the reward mean of last 100 episodes
            mean_reward = np.mean(total_rewards[-100:])
            # Print the progress
            print("Frame Index: %d - Number of Done Games: %d - Epsilon: %.3f - Speed: %.2f f/s" % (frame_index, 
                                                                                                    len(total_rewards), 
                                                                                                    epsilon,
                                                                                                    speed))
            # Add values to tensorboard
            writer.add_scalar("Epsilon", epsilon, frame_index)
            writer.add_scalar("Speed", speed, frame_index)
            writer.add_scalar("Reward - 100", mean_reward, frame_index)
            writer.add_scalar("Reward", reward, frame_index)
            # If best mean reward is none OR less than mean reward
            if (best_mean_reward is None) or (best_mean_reward < mean_reward):
                # Save the network
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
                # If there is best mean reward
                if best_mean_reward is not None:
                    # Print the reward update
                    print(">>> Best mean reward updated %.3f -> %.3f <<<" % (best_mean_reward, mean_reward))
                # Update the best mean reward
                best_mean_reward = mean_reward
            # If mean reward exceed the reward boundary
            if mean_reward > MEAN_REWARD_BOUND:
                # Environment solved
                print("Solved in {} frames!".format(frame_index))
                # Break the loop
                break
        # If buffer is full (in our case, it's 10k transitions)
        if len(buffer) < REPLAY_START_SIZE:
            # Go to beginning of the loop
            continue
        # Every SYNC_TARGET_FRAMES times, sync network's weight's
        if frame_index % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())
        # Zero out the optimizer's gradient
        optimizer.zero_grad()
        # Sample batches from experience replay buffer
        batch = buffer.sample(BATCH_SIZE)
        # Calculate the loss
        loss_t = calculate_loss(batch, net, target_net, device = device)
        # Backpropagation
        loss_t.backward()
        # Perfom the optimizer
        optimizer.step()
    # Close the writer
    writer.close()
