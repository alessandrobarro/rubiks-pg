import random
import numpy as np

class CubeMDP:
    def __init__(self):
        self.K = ['F', 'B', 'U', 'D', 'L', 'R']
        self.C = ['w', 'y', 'r', 'o', 'b', 'g']
        self.d = ['+', '-']
        self.X = {'F': [], 'B': [('h', 2)], 'U': [('v', 1)], 'D': [('v', -1)], 'L': [('h', -1)], 'R': [('h', 1)]}
        self.F = {face: color * np.ones((3, 3), dtype=object) for face, color in zip(self.K, self.C)}
        self.F_T = self.F.copy()

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
        self.F['F'] = np.rot90(self.F['F'], -1 if d == '+' else 1)
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

    def reset_front(self, K):
        for d, n in self.X[K]:
            for _ in range(n % 4):
                self.switch(d)
    
    def action(self, K, d):
        self.reset_front(K)
        self.rotate(d)

    def get_reward(self):
        r = 0
        for K in self.F:
            for i in range(0, 2):
                for j in range(0, 2):
                    if i != 1 and j != 1:
                        if self.F.get(K)[i][j] == self.F_T.get(K)[i][j]:
                            r += 1
        return -np.exp(-r)

    def shuffle_state(self, t):
        for _ in range(t):
            K = random.choice(self.K)
            d = random.choice(self.d)
            self.rotate(K, d)

    def is_solved_state(self):
        for K in self.F.values():
            if not np.all(K == K[0, 0]):
                return False
        return True

    def display_state(self):
        for K in self.K:
            print(f'{K} FACE')
            print(self.F[K])

cube = CubeMDP()

cube.display_state()

cube.action('R', '+')

print('*----------------------------*')

cube.display_state()