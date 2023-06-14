import numpy as np

latent0 = np.load("latent0/projected_w.npz")
latent1 = np.load("latent1/projected_w.npz")

breaks = 5
alphas = np.linspace(0, 1, breaks)
result = np.zeros((breaks, 18, 512))

for i, alpha in enumerate(alphas):

    between = alpha*latent0['w'] + (1-alpha)*latent1['w']
    result[i] = between

np.savez(f'interpolate/betweens.npz', w = result)

#python stylegan2/generate.py --network=ffhq.pkl --projected-w=interpolate/betweens.npz --outdir=interpolate