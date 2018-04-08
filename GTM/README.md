# Generative Topographic Mapping (GTM)
This section presents a broad account of GTM and its derivative codes.

## Prerequisites
GTM / GTM_Indexes
* Scikit-learn

gtm_animation
* mayavi
* moviepy

The remaining codes rely on GTM / GTM_Indexes and, therefore, have the same prerequisites. 

## Usage
### GTM training - gtm_template.py:
GTM.py is the core code behind GTM, where all its training algorithm is depicted. gtm_template.py shows how to use GTM to fit a given dataset.
```python
from GTM import GTM
# GTM Training
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
```

### GTM hyperparameter assessment - gtm_hyperparameter_template.py:
GTM_Indexes.py provides different hyperparameter criteria to GTM, allowing different optimisation paths. gtm_hyperparameter_template.py shows how to use GTM to fit a given dataset whilst verifying different hyperparameter criteria. Please check this repository's 
wiki for a more detailed explanation on which metrics are being used. 
```python
from GTM_Indexes import GTMIndexes
# Creating midpoint and midpoint + nearest neighbours datasets
mid_data = GTMIndexes(input_data=data)
input_midpoint = mid_data.gtm_midpoint()
input_midpoint_neighbors = mid_data.gtm_midpoint_neighbors(neighbors=10)

# GTM training
test = GTMIndexes(data, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2, iterations=10)
[w_optimal, beta_optimal, log_likelihood_evolution] = test.gtm_training()

# Hyperparameter metrics
r2_distance = test.gtm_distance_index(w_optimal, beta_optimal, data)
r2_neighbors = test.gtm_r2_neighbors(w_optimal, beta_optimal, data)
r2 = test.gtm_r2(w_optimal, beta_optimal, test_data)
r2_midpoint = test.gtm_r2(w_optimal, beta_optimal, input_midpoint)
r2_midpoint_neighbors = test.gtm_r2(w_optimal, beta_optimal, input_midpoint_neighbors)

print(r2_distance)
print(r2_neighbors)
print(r2)
print(r2_midpoint)
print(r2_midpoint_neighbors)
```

gtm_data.py, gtm_data_midpoint.py, gtm_hyperparameter_grid_search.py, gtm_hyperparameter_grid_search_midpoint.py, and gtm_hyperparameter_optimal.py are all supplementary files that help in the assessment of the sensitivity of each hyperparameter metric.

These codes were used on three simulation datasets representing potential nonlinearities encountered in real datasets:
* Spherical dataset
* Swiss roll dataset
* Cube dataset
By investigating how each hyperparameter behaved, the metric with the best performance was found. For more details, please check this repository's wiki. 

### Manifold evolution - gtm_animation.py
gtm_animation.py uses a spherical dataset as base for GTM training and then introduces animation showing how the three dimensional 
manifold created by GTM is flattened on a 2D surface. 
```python
import mayavi.mlab as mlab
import moviepy.editor as mpy
from GTM import GTM
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
```



