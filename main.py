"""
Policy gradient messing with Rubik's cube
e-mail: alessandro1.barro@mail.polimi.it
"""

########################################################################
#                                MODULES                               #
########################################################################

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from cube import Cube
from model import PolicyNetwork

########################################################################
#                         HYPERPARAMS AND INIT                         #
########################################################################

# Hyperparameters
input_size = 54
hidden_size = 128
output_size = 12
learning_rate = 1e-3
episodes = 3000
max_steps_per_episode = 100
gamma = 0.9
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Initialize Cube, Policy Network, and Optimizer
cube = Cube()
policy_network = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
shuffle_count = 20
epsilon = epsilon_start
solved_count = 0
total_steps_to_solve = 0

########################################################################
#                           HELPER FUNCTIONS                           #
########################################################################

# Function to convert cube state to tensor
def state_to_tensor(cube):
    """
    Convert cube state ("open box") to tensor

    :params cube: Current cube state
    """
    color_to_int = {'W': 0, 'Y': 1, 'R': 2, 'O': 3, 'B': 4, 'G': 5}
    state = []
    for face in cube.faces:
        # Convert each color on the face to its corresponding integer
        face_state = [color_to_int[color] for color in cube.cube[face].flatten()]
        state.extend(face_state)
    # Convert the state list to a tensor, adding a batch dimension
    state_tensor = torch.tensor(state, dtype=torch.float).view(1, -1)
    return state_tensor

def action_mapping(action):
    """
    Map the given action to be compatible to given state

    :params action: Selected action formatted as {FACE, DIRECTION}
    """
    assert 0 <= action < 12, "Action must be between 0 and 11"
    faces = ['F', 'B', 'U', 'D', 'L', 'R']
    directions = ['CW', 'CCW']
    
    face_index = action // 2  # Determines which face to rotate
    direction_index = action % 2  # Determines the direction of rotation (0 for CW, 1 for CCW)

    face = faces[face_index]
    direction = directions[direction_index]

    return face, direction

def calculate_reward(cube):
    """
    Calculate the reward of ending in s' by performing a in s.
    We implement a heuristic to compute distance between the current state
    and the solution

    :params cube: Current cube state
    """

    # Init 'distance' from the solved state
    total_distance = 0
    
    # Solved color for the face
    for face in cube.faces:
        center_color = cube.cube[face][1, 1]
        face_distance = np.sum(cube.cube[face] != center_color)
        total_distance += face_distance
    
    # Subtract 1 for each center sticker since they are always 'solved'
    total_distance -= len(cube.faces)
    
    # Negative reward proportional to the distance
    reward = -1000 / (total_distance + 1)
    
    # Large positive bonus for solving the cube
    if cube.is_solved():
        reward += 10000
    
    return reward

########################################################################
#                               TRAINING                               #
########################################################################

# Initialize metrics
total_rewards = []
epsilons = []
average_steps_to_solve_list = []
success_rates = []

# Training loop
for episode in range(episodes):

    # Epsilon decay strat
    epsilon = max(epsilon_end, epsilon_decay * epsilon)
    
    cube.shuffle(shuffle_count)
    log_probs = []
    rewards = []

    for step in range(max_steps_per_episode):
        state_tensor = state_to_tensor(cube)
        action_probs = policy_network(state_tensor)
        distribution = torch.distributions.Categorical(action_probs)

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = torch.tensor([np.random.choice(output_size)])
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        log_probs.append(log_prob)

        face, direction = action_mapping(action.item())
        cube.rotate(face, direction)

        reward = calculate_reward(cube)
        rewards.append(reward)

        if cube.is_solved():
            solved_count += 1
            total_steps_to_solve += step + 1
            break

    total_rewards.append(sum(rewards))
    epsilons.append(epsilon)

    # Calculate and normalize discounted rewards at the end of each episode
    discounted_rewards = [gamma**i * r for i, r in enumerate(rewards)]
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    # Policy gradient update
    optimizer.zero_grad()
    policy_loss = sum([-log_prob * Gt for log_prob, Gt in zip(log_probs, discounted_rewards)])
    policy_loss.backward()
    optimizer.step()

    if episode % 100 == 0 or episode == episodes - 1:
        print(f"Episode {episode+1}, Epsilon: {epsilon:.2f}, Total Reward: {sum(rewards)}, Solved: {cube.is_solved()}")

# Success Rate and Average Steps to Solve
if solved_count > 0:
    average_steps = total_steps_to_solve / solved_count
    print(f"Average Steps to Solve: {average_steps}, Success Rate: {solved_count/episodes:.2%}")

########################################################################
#                               PLOTTING                               #
########################################################################

fig, axs = plt.subplots(1, 2, figsize=(10, 10))

# Total Reward Per Episode
axs[0].plot(total_rewards)
axs[0].set_title('Total Reward per Episode')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Total Reward')

# Epsilon Values Over Episodes
axs[1].plot(epsilons)
axs[1].set_title('Epsilon Values Over Episodes')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Epsilon')

plt.tight_layout()
plt.show()