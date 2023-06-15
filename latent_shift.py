import os
import sys

import numpy as np

root = "stylegan2directions/"
files = [f for f in os.listdir(root) if f.endswith(".npy")]
directions = {f[:-4]: np.load(root + f) for f in files}
print("Loaded directions: " + ", ".join(directions.keys()))

arguments = sys.argv[1:]
# get the image path from the command line
latent_path = arguments.pop(0)
latent = np.load(latent_path)["w"]

while len(arguments) > 0:
	direction_name = arguments.pop(0)
	direction = directions[direction_name]

	direction_magnitude = float(arguments.pop(0))

	latent += direction * direction_magnitude


# Save the shifted latent vector
np.savez("shifted", w=latent)
print("Saved shifted.npz")
os.system("python stylegan2/generate.py --outdir=figures --network=ffhq.pkl --projected-w=shifted.npz")
exit(0)