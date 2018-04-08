# Spherical, Cube and Swiss Roll Data Set procedures used for dimensionality reduction comparison
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

np.random.seed(0)

# Importing swiss roll data set from MATLAB
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

# Creating a spherical data set
phi = np.random.rand(1500, 1)*2*np.pi
costheta1 = np.random.rand(1500, 1)*0.8+0.2
costheta2 = np.random.rand(1500, 1)*0.8-1

theta1 = np.arccos(costheta1)
theta2 = np.arccos(costheta2)
r = 1

x1 = r * np.sin(theta1) * np.cos(phi)
y1 = r * np.sin(theta1) * np.sin(phi)
z1 = r * np.cos(theta1)

x2 = r * np.sin(theta2) * np.cos(phi)
y2 = r * np.sin(theta2) * np.sin(phi)
z2 = r * np.cos(theta2)

spherical_input_data = np.array([np.concatenate([x1[0:1000], x2[0:1000]]), np.concatenate([y1[0:1000], y2[0:1000]]),
                                 np.concatenate([z1[0:1000], z2[0:1000]])])
spherical_input_data = np.transpose(np.reshape(spherical_input_data, (3, 2000)))
spherical_input_test_data = np.array([np.concatenate([x1[1000:1500], x2[1000:1500]]),
                                      np.concatenate([y1[1000:1500], y2[1000:1500]]),
                                      np.concatenate([z1[1000:1500], z2[1000:1500]])])
spherical_input_test_data = np.transpose(np.reshape(spherical_input_test_data, (3, 1000)))

# Creating the cube data set


def spherical_data(number_samples):
    phi = np.random.rand(number_samples, 1)*2*np.pi
    costheta = np.random.rand(number_samples, 1)*2-1
    u = np.random.rand(number_samples, 1)
    theta = np.arccos(costheta)
    r = u**(1./3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

cube_index = [[-1., 1., -1.], [1., 1., -1.], [-1., 1., 1.], [1., 1., 1.], [-1., -1., -1.], [1., -1., -1.],
              [-1., -1., 1.], [1., -1., 1.]]
cube_data_x = []
cube_data_y = []
cube_data_z = []
cube_test_data_x = []
cube_test_data_y = []
cube_test_data_z = []

for i in range(0, 8):
    [x, y, z] = spherical_data(375)
    cube_index_iter = cube_index[i]
    x = (x + cube_index_iter[0])/2.+cube_index_iter[0]/2.
    y = (y + cube_index_iter[1])/2.+cube_index_iter[1]/2.
    z = (z + cube_index_iter[2])/2.+cube_index_iter[2]/2.
    cube_data_x.extend(x[0:250].tolist())
    cube_data_y.extend(y[0:250].tolist())
    cube_data_z.extend(z[0:250].tolist())
    cube_test_data_x.extend(x[250:375].tolist())
    cube_test_data_y.extend(y[250:375].tolist())
    cube_test_data_z.extend(z[250:375].tolist())

cube_input_data = np.array([cube_data_x, cube_data_y, cube_data_z])
cube_input_data = np.transpose(np.reshape(cube_input_data, (3, 2000)))
cube_input_test_data = np.array([cube_test_data_x, cube_test_data_y, cube_test_data_z])
cube_input_test_data = np.transpose(np.reshape(cube_input_test_data, (3, 1000)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(swiss_input_data[:, 0], swiss_input_data[:, 1], swiss_input_data[:, 2], c='r')
# ax.scatter(swiss_input_test_data[:, 0], swiss_input_test_data[:, 1], swiss_input_test_data[:, 2], c='b')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spherical_input_data[:, 0], spherical_input_data[:, 1], spherical_input_data[:, 2], c='r')
# ax.scatter(spherical_input_test_data[:, 0], spherical_input_test_data[:, 1], spherical_input_test_data[:, 2], c='b')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cube_input_data[:, 0], cube_input_data[:, 1], cube_input_data[:, 2], c='r')
# ax.scatter(cube_input_test_data[:, 0], cube_input_test_data[:, 1], cube_input_test_data[:, 2], c='b')
plt.show()

np.savez('Swiss_data.npz', swiss_input_data, swiss_input_test_data)
np.savez('Spherical_data.npz', spherical_input_data, spherical_input_test_data)
np.savez('Cube_data.npz', cube_input_data, cube_input_test_data)

