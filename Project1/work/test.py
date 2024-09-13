import gymnasium as gym

# Create the Frozen Lake environment with human rendering mode
env = gym.make('FrozenLake-v1', render_mode='human', desc=None, map_name="4x4", is_slippery=True)

# Reset the environment to its initial state
state, info = env.reset()

# Print the initial state
print("Initial State:")
env.render()

# Define valid actions
valid_actions = [0, 1, 2, 3]
action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

# Take user input to explore the environment
while True:
    # Ask the user for the next action
    action = input("Enter an action (0: Left, 1: Down, 2: Right, 3: Up): ")
    
    # Ensure valid input
    if action.isdigit() and int(action) in valid_actions:
        action = int(action)
    else:
        print("Invalid input! Please enter a number between 0 and 3.")
        continue

    # Take the chosen action
    next_state, reward, done, truncated, info = env.step(action)
    
    # Print the current state after the action
    print(f"Action taken: {action_names[action]}")
    env.render()  # Render the current state

    # Check if the episode is finished
    if done:
        print("Episode finished after reaching a terminal state.")
        break

# Close the environment
env.close()
