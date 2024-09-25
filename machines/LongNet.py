# Since LongNet is originally given in pytorch , I'm trying to recreate in TensorFlow to see if I can replicate
# what GigaPath from Microsoft did in using it to analyze gigabyte size images, specifically oncology images

########################
# IMPORTS
########################

# We have to build this from the Layer, Dense and Dropout layers in Keras.

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

########################
# DILATED ATTENTION
########################

# PURPOSE: To create a dilatedAttention Layer for Tensorflow, since one is not available from keras
# DETAILS:

#   Dilated Attention is a form of multiheaded attention, which is an important component of transformers.

#   A self attention mechanism finds the relevance of one element in a vector to another element in the vector
#   For example, in a jpeg, it could find the relevance of one pixel to another.

#   Multihead attention layers do self attention in parallel multiple times with different linear projections (learned each time)
#   each projection is a head

#   
class DilatedAttention(Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate, **kwargs):
        super(DilatedAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        return attn_output

########################
# ENCODER/DECODER LAYERS
########################

class EncoderLayer(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.attention = DilatedAttention(embed_dim, num_heads, dropout_rate)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class DecoderLayer(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.attention1 = DilatedAttention(embed_dim, num_heads, dropout_rate)
        self.attention2 = DilatedAttention(embed_dim, num_heads, dropout_rate)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
    
    def call(self, inputs, enc_output, training=False):
        attn_output1 = self.attention1(inputs, training=training)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(inputs + attn_output1)
        attn_output2 = self.attention2(out1, enc_output, training=training)
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(out1 + attn_output2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

########################
# ENCODER / DECODER
########################

class Encoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.enc_layers = [EncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        return self.layernorm(x)

class Decoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dec_layers = [DecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, enc_output, training=False):
        x = inputs
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training=training)
        return self.layernorm(x)

########################
# LONGNET
########################

class LongNet(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(LongNet, self).__init__(**kwargs)
        self.encoder = Encoder(num_layers, embed_dim, num_heads, ff_dim, dropout_rate)
        self.decoder = Decoder(num_layers, embed_dim, num_heads, ff_dim, dropout_rate)
        self.final_layer = Dense(embed_dim)
    
    def call(self, enc_inputs, dec_inputs, training=False):
        enc_output = self.encoder(enc_inputs, training=training)
        dec_output = self.decoder(dec_inputs, enc_output, training=training)
        final_output = self.final_layer(dec_output)
        return final_output

########################
# EXAMPLE USAGE
########################