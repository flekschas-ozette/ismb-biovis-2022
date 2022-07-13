import os
import sys
import tensorflow as tf

# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Reshape, UpSampling2D, Layer, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean

sys.stderr = stderr

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=-1, keepdims=True
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def create(num_markers):
    encoder_input = Input(shape=(num_markers,), name='encoder_input')

    x = Dense(1024, activation='relu', name='dense1')(encoder_input)
    x = Dense(256, activation='relu', name='dense2')(x)
    x = Dense(16, activation='relu', name='dense3')(x)

    # Latent Space
    z_mean = Dense(2, name='z_mean')(x)
    z_log_var = Dense(2, name='z_log_var')(x)
    z = Sampling(name='z')([z_mean, z_log_var])
    
    latent_input = Input(shape=(2,), name='latent_input')

    x = Dense(16, activation='relu', name='undense3')(latent_input)
    x = Dense(256, activation='relu', name='undense2')(x)
    x = Dense(1024, activation='relu', name='undense1')(x)
    decoded = Dense(num_markers, activation='sigmoid', name='output')(x)
    
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    decoder = keras.Model(latent_input, decoded, name='decoder')
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    return vae
