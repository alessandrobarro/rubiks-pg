"""
Policy gradient messing with Rubik's cube
e-mail: alessandro1.barro@mail.polimi.it
"""

import random
import numpy as np

class Cube:
    def __init__(self):
        """
        Initialize the cube. We use an "open box" model where the std configuration is {F: W, B: Y, U: R, D: O, L: B, R: G}
        """
        self.faces = ['F', 'B', 'U', 'D', 'L', 'R']
        self.colors = ['W', 'Y', 'R', 'O', 'B', 'G']
        self.directions = ['CW', 'CCW']
        self.cube = {face: color * np.ones((3, 3), dtype=object) for face, color in zip(self.faces, self.colors)}
    
    def print_cube(self):
        """
        Displays current cube state
        """
        for face in self.faces:
            print(f'{face} face')
            print(self.cube[face])

    def swap_pov(self, key):
        """
        Switches the point of view horizontally or vertically
        
        :param key: could be 'h' horizontally, or 'v' vertically

        NOTE: This alters the current cube state
        """
        if key == 'h':
            # {F, B, L, R} are shifted s.t. F' = R
            temp = self.cube['L'].copy()
            self.cube['L'] = self.cube['F']
            self.cube['F'] = self.cube['R']
            self.cube['R'] = self.cube['B']
            self.cube['B'] = temp

            # {U, D} are rotated clockwise
            self.cube['U'] = np.rot90(self.cube['U'], -1)
            self.cube['D'] = np.rot90(self.cube['D'], 1)

        elif key == 'v':
            # {F, B, U, D} are shifted s.t. F' = U
            temp = self.cube['D'].copy()
            self.cube['D'] = self.cube['F']
            self.cube['F'] = self.cube['U']
            self.cube['U'] = np.rot90(self.cube['B'], 2) # When changing plane, prospective is flipped
            self.cube['B'] = np.rot90(temp, 2)

            # {L, R} are rotated clockwise
            self.cube['L'] = np.rot90(self.cube['L'], -1)
            self.cube['R'] = np.rot90(self.cube['R'], 1)

    def bring_to_front(self, face):
        """
        Bring the target face in position F
        
        :param face: The target face to bring in position F
        """
        adj_mappings = {'F': [], 'B': [('h', 2)], 'U': [('v', 1)], 'D': [('v', -1)], 'L': [('h', -1)], 'R': [('h', 1)]}

        for key, times in adj_mappings[face]:
            for _ in range(times % 4):
                self.swap_pov(key)

    def rotate(self, face, key):
        """
        Perform rotations

        :param face: The selected face {F, B, U, D, L, R}
        :param key: The selected direction, 'CW' clockwise, 'CCW' counter-clockwise
        
        NOTE: Rotations are applied only to F, so first state and prospetive has to be modified to bring face to F according to switch_pow() dynamics
        """
        # Bring to F the target face
        self.bring_to_front(face)

        # Rotate the face itself
        self.cube['F'] = np.rot90(self.cube['F'], -1 if key == 'CW' else 1)

        # Perform rotation
        if key == 'CW':
            temp = self.cube['U'][2, :].copy()
            self.cube['U'][2, :] = np.flip(self.cube['L'][:, 2])
            self.cube['L'][:, 2] = self.cube['D'][0, :]
            self.cube['D'][0, :] = np.flip(self.cube['R'][:, 0])
            self.cube['R'][:, 0] = temp

        elif key == 'CCW':
            temp = self.cube['U'][2, :].copy()
            self.cube['U'][2, :] = self.cube['R'][:, 0]
            self.cube['R'][:, 0] = np.flip(self.cube['D'][0, :])
            self.cube['D'][0, :] = self.cube['L'][:, 2]
            self.cube['L'][:, 2] = np.flip(temp)

    def shuffle(self, num_moves):
        """
        Shuffle the cube by performing a series of random moves
        
        :param num_moves: The number of random moves to perform for shuffling
        """
        moves = ['F', 'B', 'U', 'D', 'L', 'R']
        directions = ['CW', 'CCW']
        
        for _ in range(num_moves):
            face = random.choice(moves)
            direction = random.choice(directions)
            self.rotate(face, direction)

    def is_solved(self):
        """
        Check if the cube is solved
        """
        for face in self.cube.values():
            if not np.all(face == face[0, 0]):
                return False
        return True
