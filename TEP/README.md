# Tennessee Eastman Process (TEP)
This section shows different applications of data visualisation and dimensionality reduction applied to Tennessee Eastman Process (TEP). More information on the TEP dataset can be found in this repository's wiki.

## Prerequisites
Same as the ones presented in the GTM subdirectory's README.md

## Usage 
### Hyperparameter criteria assessment
TEP_data.py, TEP_data_midpoint.py, gtm_tep_hyperparameter_grid_search.py, gtm_tep_hyperparameter_grid_search_midpoint.py, and gtm_tep_hyperparameter_optimal.py are derivative of codes with similar names on the GTM subdirectory. 

Investigating how hyperparameters behave with the TEP dataset was fundamental to corroborate the results presented in the GTM subdirectory analysis. For more details, please check this repository's wiki. 

### Comparison between GTM and AE
TEP's complex high dimensionality was used to see whether GTM would outperform AE when reducing data dimensionality to a 2D latent map. 
```python
from GTM_Indexes import GTMIndexes
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler

test = GTMIndexes(dataset_train, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2,
                  iterations=50)
[w_optimal, beta_optimal, log_likelihood_evolution] = test.gtm_training()

# Visualising GTM
fig1 = plt.figure()
means2 = test.gtm_mean(w_optimal, beta_optimal)
plt.scatter(means2[:, 0], means2[:, 1], c=y_target_train)
plt.show()

# Training AE
encoding_dim = 2
input_data = Input(shape=(dataset_train.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input_data)
decoded = Dense(dataset_train.shape[1], activation='sigmoid')(encoded)
autoencoder = Model(input_data, decoded)
encoder = Model(input_data, encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=1000, batch_size=30, shuffle=False, validation_data=(X_train, X_train))

# Visualising AE encoding map
encoded_imgs = encoder.predict(X_train)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_target_train)
plt.show()
```
