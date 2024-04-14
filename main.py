import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from cube_mdp import CubeMDP
from model import PolicyNetwork

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

def reinforce(env, policy, episodes, accuracy, diff, gamma, max_epochs=100, report_interval=100): #avg moves to solve cube is around 70, we add 30 then give up
    episode_rewards = []
    done_log = []
    solved_log = []
    done_log_plot = np.zeros((20, 1))
    done_count = 0
    difficulty_level = 0
    solved_count = 0
    solved_rate = 0

    for episode in range(episodes):
        states, actions, rewards = [], [], []        
        state = env.reset(scramble_moves = 1 + (difficulty_level * 2))
        state = flatten_state(state)
        done = False
        epoch_count = 0
        
        while done == False and epoch_count < max_epochs:
            action_probs = policy.predict(state)
            action_index = np.random.choice(len(action_probs), p=action_probs)
            action = action_mapping(action_index)
            new_state, reward, done = env.step(action)
            new_state = flatten_state(new_state)

            states.append(state)
            actions.append(action_index)
            rewards.append(reward)
            state = new_state
            epoch_count += 1

        done_log.append(done)
        done_log_plot[difficulty_level] += 1
        
        if done == True:
            done_count += 1
            solved_count = 0 # n of times the cube has to be solved in that diff before stepping up diff
            for log in done_log:
                if log == True:
                    solved_count += 1
            solved_rate = solved_count / len(done_log)
            solved_log.append(solved_rate * 100)
            if solved_count >= accuracy: #using solvedrate as accuracy is too demanding for now
                if difficulty_level <= diff:
                    difficulty_level += 1
                    done_log = []

        #print(done_log)

        episode_rewards.append(np.mean(rewards))

        if episode % report_interval == 0 or episode == episodes - 1:
            print('*---------------------------------------------------------------------------------------------*')
            print(f'                                            Episode {episode+1}')
            avg_reward = np.mean(episode_rewards[-report_interval:])
            print(f'End of interval report at current episode:')
            print(f'-  Average Reward per Action = {avg_reward:.2f}')
            print(f'-  Difficulty level (scramble moves): {1 + difficulty_level * 2}')
            print(f'-  Solving rate in current difficulty: {solved_rate * 100}%')

        discounted_rewards = []
        cumulative = 0
        for reward in reversed(rewards):
            cumulative = reward + gamma * cumulative
            discounted_rewards.insert(0, cumulative)

        for state, action_index, reward in zip(states, actions, discounted_rewards):
            advantage = reward
            gradients = policy.get_gradients(state, action_index, advantage)
            policy.update(gradients)

    policy.save_weights("final_policy_weights10_2lay.npz")

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Average Reward per Action')
    plt.title('Training Progress')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward per Action')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(solved_rate * 100, label='Solved count')
    plt.title('Training Progress')
    plt.xlabel('Episodes')
    plt.ylabel('Solving rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = CubeMDP()
    policy = PolicyNetwork(state_size=54, hidden_size=128, action_size=12)
    reinforce(env, policy, episodes=200000, accuracy=100, diff=9, gamma=0.98)
