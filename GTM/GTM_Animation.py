import numpy as np
import mayavi.mlab as mlab
import moviepy.editor as mpy
from GTM import GTM
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from moviepy.video.io.bindings import mplfig_to_npimage

# Creating a spherical data set
phi = np.random.rand(1000, 1)*2*np.pi
costheta1 = np.random.rand(1000, 1)*0.8+0.2
costheta2 = np.random.rand(1000, 1)*0.8-1

theta1 = np.arccos(costheta1)
theta2 = np.arccos(costheta2)
r = 1

x1 = r * np.sin(theta1) * np.cos(phi)
y1 = r * np.sin(theta1) * np.sin(phi)
z1 = r * np.cos(theta1)

x2 = r * np.sin(theta2) * np.cos(phi)
y2 = r * np.sin(theta2) * np.sin(phi)
z2 = r * np.cos(theta2)

input_data = np.array([np.concatenate([x1, x2]), np.concatenate([y1, y2]), np.concatenate([z1, z2])])
input_data = np.transpose(np.reshape(input_data, (3, 2000)))

xx = np.concatenate([x1, x2])
yy = np.concatenate([y1, y2])
zz = np.concatenate([z1, z2])

color_vector = np.array([np.repeat("#0000FF", 1000), np.repeat("#00FF00", 1000)])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, c=color_vector)

# GTM training
test = GTM(input_data, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2, iterations=100)
[w2, beta2, log_likelihood_evol2] = test.gtm_training()

# Animated Manifold Learning
duration = 4.0
fig_myv = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
fps = 25


def x_reconstructed(d):
    x = test.manifold[:, 0, d].reshape((60, 60))
    return x


def y_reconstructed(d):
    y = test.manifold[:, 1, d].reshape((60, 60))
    return y


def z_reconstructed(d):
    z = test.manifold[:, 2, d].reshape((60, 60))
    return z

xx = test.centered_input_data[:, 0]
yy = test.centered_input_data[:, 1]
zz = test.centered_input_data[:, 2]


def make_frame(t):
    mlab.clf() # clear the figure (to reset the colors)
    mlab.mesh(y_reconstructed(test.iterations * t / duration), x_reconstructed(test.iterations * t / duration),
              z_reconstructed(test.iterations * t / duration), figure=fig_myv, transparent=True,
              representation='wireframe', color=(0, 1, 0), line_width=1)
    mlab.points3d(yy[0:1000], xx[0:1000], zz[0:1000], scale_factor=0.05, color=(1, 0, 0))
    mlab.points3d(yy[1000:2000], xx[1000:2000], zz[1000:2000], scale_factor=0.05, color=(0, 0, 1))
    return mlab.screenshot(antialiased=True)

animation_manifold = mpy.VideoClip(make_frame, duration=duration)

# Animated Map Training

fig_mpl, ax = plt.subplots(1, figsize=(4, 4), facecolor='white')
color_vector = np.array([np.linspace(1, 1, 1000), np.linspace(2, 2, 1000)])


def make_mean_plot(t):
    ax.clear()
    mean_iter = test.gtm_mean(test.w_evolution[:, :, test.iterations*t/duration],
                              test.beta_evolution[test.iterations*t/duration, 0])
    x = mean_iter[:, 0]
    y = mean_iter[:, 1]
    ax.scatter(x[0:1000], y[0:1000], c='r')
    ax.scatter(x[1000:2000], y[1000:2000], c='b')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return mplfig_to_npimage(fig_mpl)

animation_mean = mpy.VideoClip(make_mean_plot, duration=duration)

animation_manifold.write_gif("gtm_manifold.gif", fps=fps)
animation_mean.write_gif("gtm_mean.gif", fps=fps)

clip_mean = mpy.VideoFileClip("gtm_mean.gif")
clip_manifold = mpy.VideoFileClip("gtm_manifold.gif").resize(height=clip_mean.h)
clips = mpy.clips_array([[clip_manifold, clip_mean]])
clips.write_gif("gtm_evolution.gif", fps=fps)
