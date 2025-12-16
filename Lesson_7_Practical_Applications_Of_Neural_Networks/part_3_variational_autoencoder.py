from tensorflow.keras.layers import Input, Layer, Lambda
from tensorflow.keras import backend as K

# Define the encoder
input_img = Input(shape=(28 * 28,))
h = Dense(128, activation='relu')(input_img)
z_mean = Dense(2)(h)
z_log_var = Dense(2)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Latent space
z = Lambda(sampling)([z_mean, z_log_var])

# Define the decoder
decoder_h = Dense(128, activation='relu')
decoder_mean = Dense(28 * 28, activation='sigmoid')

h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Create the VAE model
vae = Model(input_img, x_decoded_mean)
vae.compile(optimizer='adam', loss='binary_crossentropy')
