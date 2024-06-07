import tensorflow as tf
from tensorflow.keras import layers
from app.params import *
import numpy as np

'''
FastSpeech2 (https://arxiv.org/pdf/2006.04558.pdf)
The duration, pitch and energy predictors share similar model structure (but different
model parameters), which consists of a 2-layer 1D-convolutional network with ReLU activation,
each followed by the layer normalization and the dropout layer, and an extra linear layer to project
the hidden states into the output sequence.
'''

class VarianceAdaptor(layers.Layer):
    def __init__(self, conv_filters, conv_kernel_size, rate):
        super().__init__()
        self.conv1 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
    
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self.duration_predictor = tf.keras.layers.Dense(1, activation='linear')
            
    def call(self, encoder_output, phone_durations_input, training):
        conv1_output = self.conv1(encoder_output)
        conv1_output = self.dropout1(conv1_output, training=training)
        conv1_output = self.layernorm1(conv1_output + encoder_output)
        
        conv2_output = self.conv2(conv1_output)
        conv2_output = self.dropout2(conv2_output, training=training)
        conv2_output = self.layernorm2(conv2_output + conv1_output)
  
        regulated_output, updated_masks = self.regulate_length(encoder_output, phone_durations_input)
        
        duration_predictions = self.duration_predictor(conv2_output)
        duration_predictions = tf.squeeze(duration_predictions, axis=-1)

        return regulated_output, duration_predictions, updated_masks


    def regulate_length(self, encoder_output, phone_durations_input):
        clipped_durations = tf.clip_by_value(phone_durations_input, MIN_DURATION, MAX_DURATION)
        int_durations = tf.cast(tf.round(clipped_durations), tf.int32)
        
        def regulate_single_sequence(args):
            sequence, repeat_factors = args
            regulated_sequence = tf.repeat(sequence, repeat_factors, axis=0)

            mask = tf.repeat(tf.ones_like(sequence, dtype=tf.float32)[:, 0], repeat_factors, axis=0)

            padding_size = TARGET_LENGTH - tf.shape(regulated_sequence)[0]

            paddings = tf.ones((padding_size, tf.shape(sequence)[-1]), dtype=tf.float32) * MEL_SPEC_PADDING_VALUE

            mask_paddings = tf.ones((padding_size,), dtype=tf.float32) * MEL_SPEC_PADDING_VALUE  

            regulated_single_sentence = tf.concat([regulated_sequence, paddings], axis=0)
            updated_mask = tf.concat([mask, mask_paddings], axis=0)
                        
            return regulated_single_sentence, updated_mask

        regulated_sequences_and_masks = tf.map_fn(regulate_single_sequence, (encoder_output, int_durations), dtype=(tf.float32, tf.float32))
        regulated_sequences = regulated_sequences_and_masks[0]
        updated_masks = regulated_sequences_and_masks[1]
        
        
        return regulated_sequences, updated_masks

