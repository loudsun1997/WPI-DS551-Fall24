import gymnasium as gym
import numpy as np

# Step 1: Set Up the Environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# Extract environment details
nS = env.observation_space.n  # Number of states
nA = env.action_space.n  # Number of actions
P = env.unwrapped.P  # Transition probabilities and rewards from the environment

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy."""
    value_function = np.zeros(nS)
    
    while True:
        delta = 0
        for s in range(nS):
            v = 0
            for a in range(nA):
                action_prob = policy[s, a]
                for probability, next_state, reward, terminal in P[s][a]:
                    v += action_prob * probability * (reward + gamma * value_function[next_state])
            delta = max(delta, abs(v - value_function[s]))
            value_function[s] = v
        if delta < tol:
            break

    return value_function

def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy, improve the policy."""
    new_policy = np.zeros([nS, nA])

    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for probability, next_state, reward, terminal in P[s][a]:
                q_values[a] += probability * (reward + gamma * value_from_policy[next_state])
        
        best_action = np.argmax(q_values)
        new_policy[s] = np.zeros(nA)
        new_policy[s][best_action] = 1.0

    return new_policy

def render_single(env, policy, render=True, n_episodes=1):
    """Render a single episode with the given policy."""
    total_rewards = 0
    for episode in range(n_episodes):
        ob, _ = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = np.argmax(policy[ob])
            next_ob, reward, done, truncated, _ = env.step(action)
            total_rewards += reward
            ob = next_ob
            if done or truncated:
                break

    return total_rewards

def policy_iteration(P, nS, nA, policy, env, gamma=0.9, tol=1e-8):
    """Runs policy iteration with rendering after each policy improvement."""
    new_policy = policy.copy()
    iteration = 0
    
    while True:
        # Policy Evaluation
        V = policy_evaluation(P, nS, nA, new_policy, gamma=gamma, tol=tol)
        
        # Policy Improvement
        improved_policy = policy_improvement(P, nS, nA, V, gamma=gamma)

        # Render the environment using the improved policy
        print(f"\nRendering policy after improvement iteration {iteration}:")
        total_rewards = render_single(env, improved_policy, render=True, n_episodes=1)
        print(f"Total rewards after improvement {iteration}: {total_rewards}\n")
        
        if np.array_equal(improved_policy, new_policy):
            break
        else:
            new_policy = improved_policy
            iteration += 1

    return new_policy, V

def value_iteration(P, nS, nA, V, env, gamma=0.9, tol=1e-8):
    """Value Iteration with rendering after certain iterations."""
    V_new = V.copy()
    policy_new = np.zeros([nS, nA])
    iteration = 0
    
    while True:
        iteration += 1
        delta = 0
        for s in range(nS):
            q_values = np.zeros(nA)
            for a in range(nA):
                for probability, next_state, reward, terminal in P[s][a]:
                    q_values[a] += probability * (reward + gamma * V[next_state])
            V_new[s] = max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        if delta < tol:
            break
        V[:] = V_new

        # Extract policy at certain iterations and render
        if iteration % 2 == 0:
            # Policy Extraction
            for s in range(nS):
                q_values = np.zeros(nA)
                for a in range(nA):
                    for probability, next_state, reward, terminal in P[s][a]:
                        q_values[a] += probability * (reward + gamma * V_new[next_state])
                best_action = np.argmax(q_values)
                policy_new[s] = np.zeros(nA)
                policy_new[s][best_action] = 1.0

            # Render the policy
            print(f"Rendering policy after iteration {iteration}:")
            total_rewards = render_single(env, policy_new, render=True, n_episodes=1)
            print(f"Total rewards: {total_rewards}\n")

    return policy_new, V_new

# Initialize a random or uniform policy
initial_policy = np.ones([nS, nA]) / nA

# Train the model using Policy Iteration with rendering
print("Training with Policy Iteration and Rendering...")
optimal_policy, optimal_value_function = policy_iteration(P, nS, nA, initial_policy, env, gamma=0.9, tol=1e-8)

print("Optimal Policy obtained through Policy Iteration:")
print(optimal_policy)
print("Optimal Value Function obtained through Policy Iteration:")
print(optimal_value_function)

# Alternatively, use Value Iteration with rendering
# V = np.zeros(nS)  # Initialize value function
# print("Training with Value Iteration and Rendering...")
# optimal_policy, optimal_value_function = value_iteration(P, nS, nA, V, env, gamma=0.9, tol=1e-8)

# Close the environment after use
env.close()
