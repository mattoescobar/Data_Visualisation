# Description

To present a broad account of GTM and its derivative codes.

# Installation
All files except GTM_Animation rely on commonly available python packages. GTM_Animation relies on mayavi and moviepy packages for its 
successful execution.

# Usage

GTM.py is the core code behind GTM, where all its training algorithm is depicted. <br/>
GTM_Indexes.py provides different hyperparameter criteria to GTM, allowing different optimisation paths. Please check this repository's 
wiki for a more detailed explanation. <br/>
GTM_Running.py is a simple implementation that trains GTM on a swiss roll dataset. <br/>
GTM_Animation.py uses a spherical dataset as base for GTM training and then introduces animation showing how the three dimensional 
manifold created by GTM is flattened on a 2D surface. 
