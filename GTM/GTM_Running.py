# GTM script template - Visualising three dimensional dataset
from GTM import GTM
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# Importing a swiss roll data set from MATLAB
input_structure = sio.loadmat('swiss_roll_data.mat')
input_data = input_structure['X']
input_data = input_data[0:1000, :]
idx = input_structure['t']
idx = idx[:, 0:1000]
idx = np.argsort(idx)
input_data = input_data[idx[0, :], :]
t = np.arange(input_data.shape[0])

# GTM training
test = GTM(input_data, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2, iterations=100)
[w_optimal, beta_optimal, log_likelihood_evolution] = test.gtm_training()

# Mean and mode plots for all samples
fig1 = plt.figure()
means2 = test.gtm_mean(w_optimal, beta_optimal)
modes2 = test.gtm_mode(w_optimal)
plt.scatter(means2[:, 0], means2[:, 1], c=t)
fig2 = plt.figure()
plt.scatter(modes2[:, 0], modes2[:, 1],  c=t)

# Probability distribution plots for all samples
fig3 = plt.figure()
test.gtm_pdf()

# Manifold plot
manifold_data = np.dot(test.fi,w_optimal)
X = manifold_data[:, 0].reshape((60, 60))
Y = manifold_data[:, 1].reshape((60, 60))
Z = manifold_data[:, 2].reshape((60, 60))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.scatter(test.centered_input_data[:, 0], test.centered_input_data[:, 1], test.centered_input_data[:, 2], c='r')
plt.show()


