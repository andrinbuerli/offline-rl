import os

from PIL import Image

image_files = [
    "pointmaze_open_random-v2_trajectories.png",
    "pointmaze_open_random-v3_trajectories.png",
    "D4RL-pointmaze-open-v2-with-wall_trajectories.png",
    "pointmaze_medium_random-v1_trajectories.png",
    "pointmaze_medium_random-v2_trajectories.png",
    "D4RL-pointmaze-medium-v2-with-wall_trajectories.png",
    "pointmaze_large_random-v1_trajectories.png",
    "pointmaze_large_random-v2_trajectories.png",
    "D4RL-pointmaze-large-v2-with-wall_trajectories.png"
]

input_dir = "assets"
output_dir = "assets/cropped"
os.makedirs(output_dir, exist_ok=True)

for filename in image_files:
    path = os.path.join(input_dir, filename)
    with Image.open(path) as img:
        w, h = img.size
        cropped = img.crop((0, 0, int(2 * w / 3), h))
        cropped.save(os.path.join(output_dir, filename))
