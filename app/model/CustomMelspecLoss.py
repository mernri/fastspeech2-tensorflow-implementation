import tensorflow as tf
from app.params import *

class CustomMelspecLoss(tf.keras.losses.Loss):
    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, y_true, y_pred):
        melspec_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        return self.beta * melspec_loss
