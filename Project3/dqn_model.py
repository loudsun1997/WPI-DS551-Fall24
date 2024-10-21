#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        # The input size for fc1 depends on the output size after the conv layers
        # With 84x84 input, conv layers reduce it to 7x7 feature map
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)  # Output layer for Q-values

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        x = F.relu(self.conv1(x))  # [batch_size, 32, 20, 20]
        x = F.relu(self.conv2(x))  # [batch_size, 64, 9, 9]
        x = F.relu(self.conv3(x))  # [batch_size, 64, 7, 7]

        # Flatten the output from the conv layers to feed into the fully connected layers
        x = x.view(x.size(0), -1)  # [batch_size, 64*7*7]

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))  # [batch_size, 512]
        x = self.fc2(x)          # [batch_size, num_actions] (Q-values)

        ###########################
        return x
