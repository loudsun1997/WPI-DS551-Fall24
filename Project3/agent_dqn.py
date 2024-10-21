#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.q_net = DQN()  # Define Q-network
        self.target_q_net = DQN()  # Define target Q-network
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # Sync target net
        self.target_q_net.eval()

        self.replay_buffer = deque(maxlen=10000)

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.learning_rate = 1e-4

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        # self.loss = torch.nn.MSELoss()

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.q_net.load_state_dict(torch.load('model.pth'))


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        if test or np.random.rand() > self.epsilon:
            # Convert observation to tensor and use Q-network
            observation = torch.FloatTensor(observation).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(observation)
            action = q_values.argmax().item()
        else:
            # Take a random action
            action = self.env.action_space.sample()

        return action

    def push(self, state, action, reward, next_state, done):

        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to torch tensors for processing in the neural network
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones


        ###########################
        # return


    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #


        ###########################
        for episode in range(1000):  # Example for 1000 episodes
            state = self.env.reset()
            for t in range(500):  # Example for 500 steps per episode
                action = self.make_action(state, test=False)
                next_state, reward, done, _, _ = self.env.step(action)

                # Store transition in replay buffer
                self.push(state, action, reward, next_state, done)

                # Sample a batch and perform training
                batch = self.replay_buffer()
                if batch:
                    states, actions, rewards, next_states, dones = batch

                    # Q-values of the current states
                    q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                    # Target Q-values using the target network for next states
                    next_q_values = self.target_q_net(next_states).max(1)[0]
                    target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

                    # Loss = (predicted Q - target Q)^2
                    loss = F.mse_loss(q_values, target_q_values.detach())

                    # Backpropagation and optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                if done:
                    break

            # Decay epsilon after each episode (epsilon-greedy)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Optionally, update target network every N steps/episodes
            if episode % 10 == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
