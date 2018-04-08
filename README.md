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
#### GTM training:
```python
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
For more details on different GTM applications, please check GTM sub-directory's README.md and this repository's wiki. 

#### GTM, SOM and AE comparison - DR_Comparison.py
DR_Comparison is using a three-dimensional scattered spherical dataset divided in two categories to show the nonlinear discriminatory capabilities of GTM, SOM, and AE. The code currently used can easily be modified to support other datasets. 

## Tennessee Eastman Process (TEP)
TEP folder presents GTM and AE related applications to Tennessee Eastman Process (TEP) industrial chemical dataset. Please check this repository's wiki for more information on TEP and the README.md on its correspondent sub-directory. 

# License
Refer to LICENSE file on the root of this repository.
