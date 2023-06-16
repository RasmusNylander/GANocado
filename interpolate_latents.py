import numpy as np

latent0 = np.load("shift/rasmus.npz")
latent1 = np.load("latent1/projected_w.npz")

breaks = 5
alphas = np.linspace(0, 1, breaks)
result = np.zeros((breaks, 18, 512))

for i, alpha in enumerate(alphas):

    between = alpha*latent0['w'] + (1-alpha)*latent1['w']
    result[i] = between

np.savez(f'interpolate/betweens.npz', w = result)

#python stylegan2/generate.py --network=ffhq.pkl --projected-w=interpolate/betweens.npz --outdir=interpolate



anna = np.load("latent0/projected_w.npz")
james = np.load("latent2/projected_w.npz")
rasmus = latent0
toby = latent1

mix = 0.25*anna['w'] + 0.25*james['w'] + 0.25*rasmus['w'] + 0.25*toby['w']

np.savez("mix/mix.npz", w=mix)
