
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LayerNormalization, Embedding)

def scaled_dot_product_attention(
    q, k, v, mask=None, neg_infty=-1.0e9):
    # Head dimension. #
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    lq = tf.shape(q)[2]
    lk = tf.shape(k)[2]
    
    # Multiplicative Attention. #
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Scale multiplicative attention mechanism. #
    attn_logits = matmul_qk * tf.math.rsqrt(dk)
    
    # Add the mask to the attention mechanism. #
    if mask is not None:
        attn_mask = (mask * neg_infty)
    else:
        attn_mask = tf.zeros([lq, lk])
    attn_logits += attn_mask
    
    attn_weights = tf.nn.softmax(attn_logits, axis=-1)
    attn_outputs = tf.matmul(attn_weights, v)
    return attn_outputs, attn_weights

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(d_model / n_heads)
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def call(self, v, k, q, mask=None):
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        
        attn_tuple = scaled_dot_product_attention(
            q, k, v, mask=mask)
        
        attn_wgt = attn_tuple[1]
        attn_out = self.combine_heads(attn_tuple[0])
        attn_out = self.wc(attn_out)
        return attn_out, attn_wgt
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        return self.ffwd_2(self.ffwd_1(x))

# GPT Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, rate1=0.1, rate2=0.1):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(d_model, n_heads)
        
        self.lnorm_1 = LayerNormalization(epsilon=1.0e-6)
        self.lnorm_2 = LayerNormalization(epsilon=1.0e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_enc, x_pos, training=True, mask=None):
        x_embed = x_enc + x_pos
        attn_self_tuple = self.attn_self(
            x_embed, x_embed, x_embed, mask=mask)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = self.dropout_1(
            attn_self_tuple[0], training=training)
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_output))
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return ffwd_self_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        tmp_pos_embed = []
        for n_layer in range(n_layers):
            tmp_pos_embed.append(
                Embedding(max_seq_length, d_model))
        
        self.pos_embed = tmp_pos_embed
        self.dec_embed = Embedding(vocab_size, d_model)
        del tmp_pos_embed
        
        # Decoder Layers. #
        tmp_dec_layers = []
        for n_layer in range(n_layers):
            tmp_dec_layers.append(DecoderLayer(
                d_model, n_heads, d_ffwd, rate1, rate2))
        
        self.dec_layers = tmp_dec_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
        del tmp_dec_layers
    
    def call(self, x, training=True):
        seq_length = tf.shape(x)[1]
        input_mask = tf.linalg.band_part(
            tf.ones([seq_length, seq_length]), -1, 0)
        input_mask = 1.0 - input_mask
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            layer_output = self.dec_layers[m](
                layer_input, x_pos_embed, 
                training=training, mask=input_mask)
            layer_input  = layer_output
        
        # The final layer's output is the decoder output. #
        dec_outputs = layer_output

        # Extract the embedding matrix. #
        x_vocab = tf.range(self.vocab_size)
        w_embed = self.dec_embed(x_vocab)
        
        # Return the vocab logits. #
        dec_logits = tf.matmul(
            dec_outputs, w_embed, transpose_b=True)
        return dec_logits, dec_outputs

class GPTDecoder(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, d_model, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(GPTDecoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.vocab_size = vocab_size
        
        # GPT Model. #
        self.gpt_model = Decoder(
            n_layers, d_model, 
            n_heads, d_ffwd, vocab_size, 
            max_seq_length, rate1=rate1, rate2=rate2)
    
    def call(self, x, training=True):
        output_tuple = self.gpt_model(x, training=training)
        
        # Extract the model's outputs. #
        dec_logits = output_tuple[0]
        dec_embed  = output_tuple[1]
        return dec_logits, dec_embed
    
    def infer(self, x):
        """
        To be depreciated. Use self.gen_text() instead.
        """
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        for step in range(self.seq_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_logits = self.call(tmp_inputs, training=False)
            
            tmp_logit = tmp_logits[:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
    
    def gen_text(
        self, x, gen_len=None, sample=True):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        if gen_len is None:
            gen_len = self.seq_len
        
        for step in range(gen_len):
            tmp_inputs  = tf.concat(infer_ids, axis=1)
            tmp_outputs = self.call(
                tmp_inputs[:, -self.seq_len:], training=False)
            
            tmp_logit = tmp_outputs[0][:, -1, :]
            if sample:
                tmp_probs  = tf.nn.softmax(
                    tmp_logit, axis=1).numpy()[0, :]
                tmp_sample = np.random.choice(
                    self.vocab_size, p=tmp_probs)
                tmp_sample = tf.expand_dims(tf.constant(
                    tmp_sample, dtype=tf.int32), axis=0)
            else:
                tmp_sample = tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32)
            
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], lambda: tmp_sample)
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)

