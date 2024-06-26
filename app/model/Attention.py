import tensorflow as tf
from tensorflow.keras import layers

class Attention(layers.Layer):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def call(self, query, key, value, mask=None):
        # Queries x Keys
        dot_product = tf.matmul(query, key, transpose_b=True, name="dot_product_q_k")
        
        # Scale
        scaler = tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        scaled_dot_product = dot_product / scaler
        
        if mask is not None:
            scaled_dot_product += (mask * -1e9)

        # Softmax
        softmaxed_attention_weights = tf.nn.softmax(scaled_dot_product, axis=-1, name="apply_softmax")

        # Multiply by Values
        attention_output = tf.matmul(softmaxed_attention_weights, value,  name="multiply_scores_w_value")
        
        return attention_output
    
