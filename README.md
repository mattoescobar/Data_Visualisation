# Data Visualisation Experiments
This repository's goal is to provide examples of how to use different data visualisation techniques, focusing however on Generative Topographic Mapping (GTM) compared to other methodologies. GTM is a probabilistic manifold based data visualsation and dimenisonality reduction tool. 

This repository also includes many analyses conducted on Tennessee Eastman Process (TEP), an industrial dataset commonly used for fault detection assessment.

## Prerequisites
GTM
* Scikit-learn

AutoEncoders (AE)
* TensorFlow framework
* Keras API

Self-Organising Maps (SOM)
* MiniSom module - JustGlowing's MiniSom (https://github.com/JustGlowing/minisom)

## Generative Topographic Mapping (GTM)
This repository contains a detailed GTM implementation from scratch in GTM.py, which relies on commonly available python packages.

### Usage
#### GTM training - gtm_template.py:
```python
from GTM import GTM
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
print(r2_midpoint_neighbors)```
For more details on this implementation and different GTM applications with extra features, please check GTM sub-directory's README.md and this repository's wiki. 

#### GTM, SOM and AE comparison - DR_Comparison.py
DR_Comparison is using a three-dimensional scattered spherical dataset divided in two categories to show the nonlinear discriminatory capabilities of GTM, SOM, and AE. The code currently used can easily be modified to support other datasets. 

## Tennessee Eastman Process (TEP)
TEP folder presents GTM and AE related applications to Tennessee Eastman Process (TEP) industrial chemical dataset. Please check this repository's wiki for more information on TEP and the README.md on its correspondent sub-directory. 

# License
Refer to LICENSE file on the root of this repository.
