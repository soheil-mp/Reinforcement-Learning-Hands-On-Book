
# %% Import the libraries
import cv2
import gym
import gym.spaces
import numpy as np
import collections


# %% Create the environment
class FireResetEnv(gym.Wrapper):

    # Constructor
    def __init__(self, env=None):
        # Call parent's constructor to initialize themselves
        super(FireResetEnv, self).__init__(env)
        # Make sure the second item in action space is "FIRE"
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        # Make sure action space is 3 or more
        assert len(env.unwrapped.get_action_meanings()) >= 3

    # Step function
    def step(self, action):
        # Take the give action and return (next_state, reward, is_done, info)
        return self.env.step(action)

    # Reset function
    def reset(self):
        # Reset the environment
        self.env.reset()
        # Take action 1 (e.g. FIRE) and return (next_state, reward, is_done, info)
        obs, _, done, _ = self.env.step(1)
        # If terminal state then reset
        if done:
            self.env.reset()
        # Take action 2 (e.g. RIGHT) and return (next_state, reward, is_done, info)
        obs, _, done, _ = self.env.step(2)
        # If terminal state then reset
        if done:
            self.env.reset()
        return obs


# %% Class for choosing action every k steps (Usually k=3 or k=4) and pixels from two consecutive frames
class MaxAndSkipEnv(gym.Wrapper):

    # Constuctor
    def __init__(self, env=None, skip=4):
        # Call parent's constructor to initialize themselves
        super(MaxAndSkipEnv, self).__init__(env)
        # Initialize the observation buffer with maximum length of 2
        self._obs_buffer = collections.deque(maxlen=2)
        # Skipping number (usually 3 or 4)
        self._skip = skip

    # Step function
    def step(self, action):
        # Initialize the total rewrad
        total_reward = 0.0
        # Initialize is_done
        done = None
        # Iterate through skips
        for _ in range(self._skip):
            # Take action and get (next_state, reward, is_done, info)
            obs, reward, done, info = self.env.step(action)
            # Add observation into observation buffer
            self._obs_buffer.append(obs)
            # Add reward into the total reward
            total_reward += reward
            # If terminals state then break the loop
            if done:
                break
        # Get the maximum of every pixel in the last 2 frames and use it as an observation
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    # Reset function
    def reset(self):
        # Clear out the observation buffer
        self._obs_buffer.clear()
        # Reset the environment and get the iniital observation
        obs = self.env.reset()
        # Append the initial observation into observation buffer 
        self._obs_buffer.append(obs)
        return obs

# %% Class for preprocessing the frames
class ProcessFrame84(gym.ObservationWrapper):
    """
    Processing environment's frame by:
        1. Convert 210x160 pixels with RGB color channels into a grayscale 84x84 image.
        2. Resize the image
        3. Crop the image
    """

    # Constructor
    def __init__(self, env=None):
        # Call parent's constructo to initialize themselves
        super(ProcessFrame84, self).__init__(env)
        # Initialize the observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    # Function for processing the observation
    def observation(self, obs):
        return ProcessFrame84.process(obs)

    # Function for processing the frames
    @staticmethod
    def process(frame):
        # If the frame size is 100'800
        if frame.size == 210 * 160 * 3:
            # Convert the frame into 210x160x3 + Convert it into float
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        # If the frame size is 120'000
        elif frame.size == 250 * 160 * 3:
            # Convert the frame into 250x160x3 + Convert it into float
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        # If the frame size is something else
        else:
            # Throw and error
            assert False, "Unknown resolution."
        # Multiply each color channel to some specific number
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # Resize the image into 84x110
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        # Crop the screen
        x_t = resized_screen[18:102, :]
        # Reshape the image into 84x84x1
        x_t = np.reshape(x_t, [84, 84, 1])
        # Convert to uint8 data type and return it
        return x_t.astype(np.uint8)


# %% Class for reverseing the image axis
class ImageToPyTorch(gym.ObservationWrapper):

    # Constructor
    def __init__(self, env):
        # Call parent's constructor to initialize themselves
        super(ImageToPyTorch, self).__init__(env)
        # Get the current observation space
        old_shape = self.observation_space.shape
        # Initialize the observation space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    # Function for observation
    def observation(self, observation):
        # Move the axis of observation
        return np.moveaxis(observation, 2, 0)

# %% Class for scaling pixels into ranges from 0.0 to 1.0 + Convert observation data to floats
class ScaledFloatFrame(gym.ObservationWrapper):

    # Observation function
    def observation(self, obs):
        # Convert to float + scale pixel values to [0, 1]
        return np.array(obs).astype(np.float32) / 255.0


# %% Class for stacking the subsequent frames (usually 4) together. This gives network the information about the dynamics of the game's objects.
class BufferWrapper(gym.ObservationWrapper):

    # Constructor
    def __init__(self, env, n_steps, dtype=np.float32):
        # Call parent's constructor to initialize themselves
        super(BufferWrapper, self).__init__(env)
        # Initialize the data type
        self.dtype = dtype
        # Initialize the of old observation space 
        old_space = env.observation_space
        # Initialize new observation space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    # Reset function
    def reset(self):
        # Initialize the buffer
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        # Add initial observation into the observation space
        return self.observation(self.env.reset())

    # Observation function
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


# %% Function for creating an environment and applying the wrappers to it
def make_env(env_name):
    # Create an environment
    env = gym.make(env_name)
    # Choosing action every k steps (Usually k=3 or k=4) and pixels from two consecutive frames
    env = MaxAndSkipEnv(env)
    # Press the "FIRE" in the begining
    env = FireResetEnv(env)
    # Process the frames
    env = ProcessFrame84(env)
    # Change the shape of observation from HWC to the CHW format
    env = ImageToPyTorch(env)
    # Create a stack of subsequent frames along the first dimension and return them as an observation
    env = BufferWrapper(env, 4)
    # Convert to floats + Scale pixel values to [0, 1]
    return ScaledFloatFrame(env)


# %% Make the environment (FOR TEST PUPORSES ONLY)
#env = make_env("PongNoFrameskip-v4")