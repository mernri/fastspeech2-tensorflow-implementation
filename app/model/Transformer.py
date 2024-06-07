from tensorflow.keras import Model, layers
import tensorflow as tf
import numpy as np
from app.model import *
from app.params import *

class Transformer(Model):
    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                 conv_kernel_size, conv_filters, rate, 
                 var_conv_filters, var_conv_kernel_size, var_rate):
        
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.encoder = Encoder(num_layers, embedding_dim, num_heads, dff, 
                               conv_kernel_size, conv_filters, rate)

        self.variance_adaptor = VarianceAdaptor(var_conv_filters, var_conv_kernel_size, var_rate)
        
        self.decoder = Decoder(num_layers, embedding_dim, num_heads, dff, 
                               conv_kernel_size, conv_filters, rate)

        self.final_layer = layers.Dense(N_MELS)
    
    def call(self, inputs, training=False):
        tokens_input = inputs['tokens_input']
        phone_durations_input = inputs['phone_durations_input']

        # EMBEDDING & POSITIONAL ENCODING
        tokens_padding_mask = self.create_tokens_padding_mask(tokens_input)
        
        embedding_output = self.embedding(tokens_input) 
        embedding_output = embedding_output * tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))

        seq_length = tf.shape(embedding_output)[1]
        pos_encoding = self.positional_encoding(seq_length, self.embedding_dim)
        
        embedding_and_pos_output = embedding_output + pos_encoding[:, :seq_length, :]
        
        # ENCODER
        encoder_output = self.encoder(embedding_and_pos_output, training, tokens_padding_mask)
        
        # VARIANCE ADAPTOR
        regulated_output, duration_predictions, updated_masks = self.variance_adaptor(encoder_output, phone_durations_input, training)
        updated_masks = updated_masks[:, tf.newaxis, tf.newaxis, :]

        # POSITIONAL ENCODING FOR DURATION REGULATED TOKENS
        regulated_seq_length = tf.shape(regulated_output)[1] 
        pos_encoding_regulated = self.positional_encoding(regulated_seq_length, self.embedding_dim)
  
        duration_regulation_and_pos_output = regulated_output + pos_encoding_regulated

        # DECODER
        decoder_output = self.decoder(duration_regulation_and_pos_output, training, updated_masks)        

        # FINAL LAYER
        final_output = self.final_layer(decoder_output)
        melspec_output = tf.transpose(final_output, perm=[0, 2, 1])
        
        # # Phonem Duration loss
        # duration_loss = tf.keras.losses.MeanAbsoluteError()(phone_durations_input, duration_predictions)
        # print("[TRANFORMER] duration_loss", duration_loss)
        
        # # Phonem Duration loss
        # tf.print('[TRANSFORMER DURATION LOSS]', duration_loss)
        
        
        mae_loss = tf.keras.losses.MeanAbsoluteError()
        phoneme_duration_loss = mae_loss(y_true=phone_durations_input, y_pred=duration_predictions)
        alpha = 0.5
        self.add_loss(alpha * phoneme_duration_loss)

        return melspec_output
        
                    
    def create_tokens_padding_mask(self, inputs):
        mask = tf.cast(tf.math.equal(inputs, TOKEN_PADDING_VALUE), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]
    
    def positional_encoding(self, position, embedding_dim):
        def get_angles(pos, i, embedding_dim):
            angle_rates = 1 / tf.math.pow(tf.cast(10000, tf.float32), (2 * (i // 2)) / tf.cast(embedding_dim, tf.float32))
            return pos * angle_rates
        
        pos = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(embedding_dim, dtype=tf.float32)[tf.newaxis, :]
        
        angle_rads = get_angles(pos, i, embedding_dim)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        
        pos_encoding = tf.reshape(pos_encoding, [position, embedding_dim])
        
        pos_encoding = pos_encoding[tf.newaxis, ...]    
            
        return tf.cast(pos_encoding, dtype=tf.float32)