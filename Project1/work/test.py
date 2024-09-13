import gymnasium as gym

# Create the Frozen Lake environment with ANSI rendering mode
env = gym.make('FrozenLake-v1', render_mode='ansi')

# Reset the environment to its initial state
state, info = env.reset()

# Print the initial state
print("Initial State:")
print(env.render())

# Take random actions to explore the environment
for _ in range(10):
    action = env.action_space.sample()  # Select a random action
    next_state, reward, done, truncated, info = env.step(action)  # Take the action
    
    # Print the current state after the action
    print(f"Action taken: {action}")
    print(env.render())  # Render the current state

    if done:
        print("Episode finished after reaching a terminal state.")
        break

# Close the environment
env.close()
