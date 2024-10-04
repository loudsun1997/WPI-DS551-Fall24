#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################

    player_score = observation[0]
    if player_score >= 20:
        return 0
    else:
        return 1


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################

    for x in tqdm(range(n_episodes), desc="Processing Episodes"):
        episode_data = []
        state = env.reset()[0]
        done = False

        while not done:
            print(state)
            action = policy(state)  # Pass only the relevant part of the state to the policy
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, reward))  # Append only the relevant part of the state
            state = next_state

        G = 0
        first_visit_states = set()

        for t in reversed(range(len(episode_data))):
            state_t, reward_t = episode_data[t]
            G = gamma * G + reward_t

            if state_t not in first_visit_states:
                first_visit_states.add(state_t)
                returns_sum[state_t] += G
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]

    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state:
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################

    if state not in Q:
        Q[state] = np.zeros(nA)

    if random.random() > epsilon:
        action = np.argmax(Q[state])
    else:
        action = random.choice(range(nA))
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################

    for episode in range(n_episodes):
        episode_data = []
        state = env.reset()[0]
        done = False

        while not done:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)

            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, action, reward)) 
            state = next_state 

        G = 0
        first_visit_pairs = set()

        for t in reversed(range(len(episode_data))):
            state_t, action_t, reward_t = episode_data[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in first_visit_pairs:
                first_visit_pairs.add((state_t, action_t))
                Q[state_t][action_t] += (G - Q[state_t][action_t]) / (returns_count[(state_t, action_t)] + 1)
                returns_count[(state_t, action_t)] += 1
        
        epsilon = max(epsilon - 0.1 / n_episodes, 0.01)
    
    return Q

    return Q
