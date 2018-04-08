# GTM script template - Testing different hyperparameter metrics
from GTM import GTM
from GTM_Indexes import GTMIndexes
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# Importing a swiss roll data set from MATLAB
input_structure = sio.loadmat('swiss_roll_data.mat')
input_data = input_structure['X']
input_data = input_data[0:2000, :]
idx = input_structure['t']
idx = idx[:, 0:2000]
idx = np.argsort(idx)
swiss_input_data = input_data[idx[0, :], :]
input_test_data = input_structure['X']
input_test_data = input_test_data[2000:3000, :]
idx_test = input_structure['t']
idx_test = idx_test[:, 2000:3000]
idx_test = np.argsort(idx_test)
swiss_input_test_data = input_test_data[idx_test[0, :], :]

# Creating midpoint and midpoint + nearest neighbours datasets
swiss = GTMIndexes(input_data=swiss_input_data)
swiss_midpoint = swiss.gtm_midpoint()
swiss_midpoint_neighbors = swiss.gtm_midpoint_neighbors(neighbors=10)

# GTM training
test = GTMIndexes(input_data, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2, iterations=10)
[w_optimal, beta_optimal, log_likelihood_evolution] = test.gtm_training()

# Hyperparameter metrics
r2_distance = test.gtm_distance_index(w_optimal, beta_optimal, swiss_input_data)
r2_neighbors = test.gtm_r2_neighbors(w_optimal, beta_optimal, swiss_input_data)
r2 = test.gtm_r2(w_optimal, beta_optimal, swiss_input_test_data)
r2_midpoint = test.gtm_r2(w_optimal, beta_optimal, swiss_midpoint)
r2_midpoint_neighbors = test.gtm_r2(w_optimal, beta_optimal, swiss_midpoint_neighbors)

print(r2_distance)
print(r2_neighbors)
print(r2)
print(r2_midpoint)
print(r2_midpoint_neighbors)


