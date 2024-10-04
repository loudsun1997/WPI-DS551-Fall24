import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

def initial_policy(observation):
    player_score = observation[0]
    if player_score >= 20:
        return 0
    else:
        return 1

class DefaultQ:
    def __init__(self, nA):
        self.nA = nA

    def __call__(self):
        return np.zeros(self.nA)


def mc_prediction_worker(args):
    """Worker function to simulate a batch of episodes and update value function."""
    policy, env, gamma, n_episodes = args
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    for _ in range(n_episodes):
        episode_data = []
        state = env.reset()[0]
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, reward))
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

def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Parallelized Monte Carlo first visit prediction."""
    num_workers = mp.cpu_count()
    episodes_per_worker = n_episodes // num_workers

    # Pool of workers
    with mp.Pool(num_workers) as pool:
        tasks = [(policy, env, gamma, episodes_per_worker) for _ in range(num_workers)]
        results = pool.map(mc_prediction_worker, tasks)

    # Aggregate results from all workers
    final_V = defaultdict(float)
    returns_count = defaultdict(float)

    for worker_V in results:
        for state, value in worker_V.items():
            final_V[state] += value
            returns_count[state] += 1

    # Average across workers
    for state in final_V:
        final_V[state] /= returns_count[state]

    return final_V


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




# Define a regular function to replace the lambda
def init_Q(nA):
    return np.zeros(nA)

def mc_control_worker(args):
    """Worker function for running episodes in parallel for MC control."""
    env, gamma, epsilon, n_episodes = args
    nA = env.action_space.n
    returns_count = defaultdict(float)
    Q = defaultdict(DefaultQ(nA))  # Use an instance of DefaultQ as default_factory

    for _ in range(n_episodes):
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
                returns_count[(state_t, action_t)] += 1
                # Incremental update of Q[state_t][action_t]
                Q[state_t][action_t] += (G - Q[state_t][action_t]) / returns_count[(state_t, action_t)]

    # Remove the default_factory before returning
    Q.default_factory = None
    return Q

def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Parallelized Monte Carlo control with epsilon-greedy policy."""
    num_workers = mp.cpu_count()
    episodes_per_worker = n_episodes // num_workers

    with mp.Pool(num_workers) as pool:
        tasks = [(env, gamma, epsilon, episodes_per_worker) for _ in range(num_workers)]
        results = pool.map(mc_control_worker, tasks)

    # Aggregate results from all workers
    final_Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for worker_Q in results:
        for state, actions in worker_Q.items():
            if state in final_Q:
                final_Q[state] += actions
            else:
                final_Q[state] = actions

    # Normalize final_Q
    for state in final_Q:
        final_Q[state] /= len(results)

    return final_Q