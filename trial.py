import random
import time
import numpy as np
from cube_mdp import CubeMDP
from model import PolicyNetwork
from visualizer import Visualizer

# Initialize policy network and load weights
policy = PolicyNetwork(state_size=54, hidden_size=128, action_size=12)
policy.load_weights('final_policy_weights10_2lay.npz')

def action_mapping(index):
    face = index // 2
    direction = '+' if index % 2 == 0 else '-'
    faces = ['F', 'B', 'U', 'D', 'L', 'R']
    return (faces[face], direction)

def flatten_state(state):
    C = ['w', 'y', 'r', 'o', 'b', 'g']
    faces = ['F', 'B', 'U', 'D', 'L', 'R']
    state_vector = []
    color_to_idx = {color: idx for idx, color in enumerate(C)}
    for face in faces:
        flat_face = state[face].flatten()
        encoded_face = [color_to_idx[color] for color in flat_face]
        state_vector.extend(encoded_face)
    return np.array(state_vector)

def run_trial():
    env = CubeMDP()
    env_visualizer = Visualizer()
    n_shuffles = [3, 5, 7, 9]

    # Randomly shuffle the cube state
    env.shuffle_state(n_shuffles[random.randint(0, 3)])
    env_visualizer.config = env.F
    env_visualizer.plot_cube()
    state = flatten_state(env.F)
    done = False
    epochs = 0

    while not done and epochs <= 160:
        action_probs = policy.predict(state)
        action_index = np.random.choice(len(action_probs), p=action_probs)
        action = action_mapping(action_index)
        new_state, reward, done = env.step(action)
        new_state = flatten_state(new_state)
        state = new_state
        epochs += 1
        
        # Visualize each step
        env_visualizer.config = new_state
        env_visualizer.plot_cube()
        time.sleep(1.3)  # Delay to allow visualization to be observed

    if done:
        print("The cube is solved!")
    else:
        print("Failed to solve the cube within the step limit.")

# Call the function to start the trial
run_trial()
