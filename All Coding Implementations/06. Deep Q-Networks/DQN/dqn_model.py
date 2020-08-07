
# %% Import the libraries
import numpy as np
import torch
import torch.nn as nn


# %% Deep Q-Network (DQN) class
class DQN(nn.Module):

    # Constructor
    def __init__(self, input_shape, n_actions):
        # Call parent's constructor to initialize themselves
        super(DQN, self).__init__()
        # Network (convolutional layers)
        self.conv = nn.Sequential(nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size = 8, stride = 4),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
                                  nn.ReLU())
        # Get the output shape
        conv_out_size = self.conv(torch.zeros(1, *input_shape))
        conv_out_size = int(np.prod(conv_out_size.size()))
        # Network (fully connected layer)
        self.fcl = nn.Sequential(nn.Linear(in_features = conv_out_size, out_features = 512),
                                 nn.ReLU(),
                                 nn.Linear(in_features = 512, out_features = n_actions))

    # Function for forward propagation
    def forward(self, x):
        # Pass the input to convolutional layers
        conv_out = self.conv(x)
        # Flattent the output to (batch_size, parameters)
        conv_out = conv_out.view(x.size()[0], -1)
        # Pass the output to fully connected layers
        fcl_out = self.fcl(conv_out)
        return fcl_out
