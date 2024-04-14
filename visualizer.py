import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class Visualizer:
    def __init__(self):
        self.K = ['F', 'B', 'U', 'D', 'L', 'R']
        self.C = ['w', 'y', 'r', 'o', 'b', 'g']
        self.config = {face: color * np.ones((3, 3), dtype=object) for face, color in zip(self.K, self.C)}

        self.color_mapping = {
            'w': 'white',  # white
            'y': 'yellow', # yellow
            'r': 'red',    # red
            'o': '#FFA500',# orange, using hex since 'orange' is not a single char
            'b': 'blue',   # blue
            'g': 'green'   # green
        }

        
    def draw_cubie(self, ax, position, size, face_config):
        x, y, z = position
        r = size / 2
        corners = np.array([
            [x-r, y-r, z-r],
            [x-r, y-r, z+r],
            [x-r, y+r, z-r],
            [x-r, y+r, z+r],
            [x+r, y-r, z-r],
            [x+r, y-r, z+r],
            [x+r, y+r, z-r],
            [x+r, y+r, z+r],
        ])
        
        faces = [
            [corners[0], corners[1], corners[3], corners[2]], # Left face
            [corners[4], corners[5], corners[7], corners[6]], # Right face
            [corners[0], corners[1], corners[5], corners[4]], # Bottom face
            [corners[2], corners[3], corners[7], corners[6]], # Top face
            [corners[0], corners[2], corners[6], corners[4]], # Back face
            [corners[1], corners[3], corners[7], corners[5]], # Front face
        ]
        
        face_names = ['L', 'R', 'D', 'U', 'B', 'F']
        indices = [(y, x), (y, x), (z, y), (z, y), (z, x), (z, x)]
        
        for i, face in enumerate(faces):
            face_color = face_config[face_names[i]][indices[i]]
            face_color = self.color_mapping[face_color]  # This translates, e.g., 'o' to '#FFA500'
            ax.add_collection3d(Poly3DCollection([face], color=face_color))
        
    def plot_cube(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cubie_size = 0.9
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.draw_cubie(ax, (x, y, z), cubie_size, self.config)
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_zlim(0, 3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(30, 45)
        
        plt.show()

# Create an instance and plot
visualizer = Visualizer()
visualizer.plot_cube()
