import numpy as np
from visualizer import Visualizer

# %% load data

with open('resources/visualization_data.pkl', 'rb') as f:
    data = np.load(f)
    coordinates = data['coordinates']
    faces = data['faces']
    colors = data['colors']

# %% visualize data
vis = Visualizer(background_color='white')
vis.animate(coordinates,
            faces=faces,
            color=colors
            )

# you can alos visualize multiple objects with different colors (specified by rgb values for every coordinate,
# by a string, or by error_values over an color map) and e.g. as points
random_colors = np.random.rand(*coordinates.shape[0:2])
vis.animate([coordinates, coordinates, coordinates],
            faces=[None, faces, faces],
            color=["blue", colors, random_colors],
            shift=True,
            camera_distance=1000,
            rotate_camera=True
            )