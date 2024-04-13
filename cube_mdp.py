import random
import numpy as np

class CubeMDP:
    def __init__(self):
        self.K = ['F', 'B', 'U', 'D', 'L', 'R']
        self.C = ['w', 'y', 'r', 'o', 'b', 'g']
        self.d = ['+', '-']
        self.X = {'F': [], 'B': [('h', 2)], 'U': [('v', 1)], 'D': [('v', -1)], 'L': [('h', -1)], 'R': [('h', 1)]}
        self.X_not = {'F': [], 'B': [('h', 2)], 'U': [('v', -1)], 'D': [('v', 1)], 'L': [('h', 1)], 'R': [('h', -1)]}
        self.F = {face: color * np.ones((3, 3), dtype=object) for face, color in zip(self.K, self.C)}
        self.F_T = {face: color * np.ones((3, 3), dtype=object) for face, color in zip(self.K, self.C)}
        self.color_to_int = {color: i for i, color in enumerate(self.C)}

    def switch(self, d):
        if d == 'h': #<-
            temp = self.F['F'].copy()
            self.F['F'] = self.F['R']
            self.F['R'] = self.F['B']
            self.F['B'] = self.F['L']
            self.F['L'] = temp
            self.F['U'] = np.rot90(self.F['U'], -1)
            self.F['D'] = np.rot90(self.F['D'], 1)
        elif d == 'v': #\/
            temp = self.F['F'].copy()
            self.F['F'] = self.F['U']
            self.F['U'] = np.rot90(self.F['B'], 2)
            self.F['B'] = np.rot90(self.F['D'], 2)
            self.F['D'] = temp
            self.F['L'] = np.rot90(self.F['L'], -1)
            self.F['R'] = np.rot90(self.F['R'], 1)
    
    def rotate(self, d):
        if d == '+':
            self.F['F'] = np.rot90(self.F['F'], -1)
            temp = self.F['U'][2, :].copy()
            self.F['U'][2, :] = np.flip(self.F['L'][:, 2])
            self.F['L'][:, 2] = self.F['D'][0, :]
            self.F['D'][0, :] = np.flip(self.F['R'][:, 0])
            self.F['R'][:, 0] = temp
        elif d == '-':
            self.F['F'] = np.rot90(self.F['F'], 1)
            temp = self.F['U'][2, :].copy()
            self.F['U'][2, :] = self.F['R'][:, 0]
            self.F['R'][:, 0] = np.flip(self.F['D'][0, :])
            self.F['D'][0, :] = self.F['L'][:, 2]
            self.F['L'][:, 2] = np.flip(temp)

    def fw_set_front(self, K):
        for d, n in self.X[K]:
            for _ in range(n % 4):
                self.switch(d)

    def bw_set_front(self, K):
        for d, n in self.X_not[K]:
            for _ in range(n % 4):
                self.switch(d)

    def action(self, K, d):
        self.fw_set_front(K)
        self.rotate(d)
        self.bw_set_front(K)

    def get_reward(self, alpha = 8):
        r = 0
        for K in self.F:
            curr_face = self.F.get(K)
            exact_face = self.F_T.get(K)
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    if curr_face[i][j] == exact_face[i][j]:
                        r += 1
        return np.exp(r / alpha)
        
    def step(self, a):
        done = False
        self.action(a[0], a[1]) #tuple
        new_state = self.F
        reward = self.get_reward()
        if self.is_solved_state():
            done = True
        return new_state, reward, done

    def shuffle_state(self, t):
        for _ in range(t):
            K = random.choice(self.K)
            d = random.choice(self.d)
            self.action(K, d)
    
    def reset(self, scramble_moves=15):
        self.F = {face: color * np.ones((3, 3), dtype=object) for face, color in zip(self.K, self.C)}
        self.shuffle_state(scramble_moves)
        return self.F

    def is_solved_state(self):
        for K in self.F.values():
            if not np.all(K == K[0, 0]):
                return False
        return True

    def display_state(self):
        for K in self.K:
            print(f'{K} FACE')
            print(self.F[K])

'''
 *** EXAMPLE USAGE ***

cube = CubeMDP() # Init cube instance

cube.display_state() # Check initial state

# Perform standard RUR'U' algorithm
cube.action('R', '+')
cube.action('U', '+')
cube.action('R', '-')
cube.action('U', '-')

print('*-------------------------*')

cube.display_state() # Check new state
'''