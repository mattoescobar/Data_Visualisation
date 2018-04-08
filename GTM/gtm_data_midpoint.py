# Spherical, Cube and Swiss Roll Midpoint Datasets
from GTM_Indexes import GTMIndexes
import numpy as np

np.random.seed(0)

# Importing swiss roll data set from MATLAB
swiss_data = np.load('Swiss_data.npz')
swiss_input_data = swiss_data['arr_0']
swiss_input_test_data = swiss_data['arr_1']
spherical_data = np.load('Spherical_data.npz')
spherical_input_data = spherical_data['arr_0']
spherical_input_test_data = spherical_data['arr_1']
cube_data = np.load('Cube_data.npz')
cube_input_data = cube_data['arr_0']
cube_input_test_data = cube_data['arr_1']

# Creating Midpoint data for each scenario
swiss = GTMIndexes(input_data=swiss_input_data)
swiss_midpoint = swiss.gtm_midpoint()
swiss_midpoint_neighbors = swiss.gtm_midpoint_neighbors(neighbors=10)
np.savez('swiss_midpoint.npz', swiss_midpoint, swiss_midpoint_neighbors)

spherical = GTMIndexes(input_data=spherical_input_data)
spherical_midpoint = spherical.gtm_midpoint()
spherical_midpoint_neighbors = spherical.gtm_midpoint_neighbors(neighbors=10)
np.savez('spherical_midpoint.npz', spherical_midpoint, spherical_midpoint_neighbors)

cube = GTMIndexes(input_data=cube_input_data)
cube_midpoint = cube.gtm_midpoint()
cube_midpoint_neighbors = cube.gtm_midpoint_neighbors(neighbors=10)
np.savez('cube_midpoint.npz', cube_midpoint, cube_midpoint_neighbors)