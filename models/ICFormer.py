import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------- Patch Embedding -------------------------------
class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        """
        num_patches: number of patches == number of cycles
        projection_dim: dimension of the embedding
        """
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        """
        Applies positional embedding to know the cycle order
        patch: [batch_size, num_patches, seq_len]
        
        returns: [batch_size, num_patches, projection_dim]
        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
# ----------------------------------------------------------------------------

# ---------------------------- Transformer Encoder ---------------------------
class EncoderLayer(layers.Layer):
    def __init__(self, num_heads, head_size, dff, n_features, rate=0.1):
        """
        num_heads: number of heads in the multi-head attention layer
        head_size: dimension of the head embedding
        dff: dimension of the feed forward network
        rate: dropout rate
        """
        super(EncoderLayer, self).__init__()

        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=rate)
        self.dropout1 = layers.Dropout(rate)
        self.ffn = keras.Sequential([
                                layers.LayerNormalization(epsilon=1e-6),
                                layers.Conv1D(filters=dff, kernel_size=1, activation="relu"),
                                layers.Conv1D(filters=n_features, kernel_size=1)
                            ])

    def call(self, inputs, training, mask):
        """
        inputs: [batch_size, num_patches, d_model]
        training: boolean, specifies whether the dropout layer should behave in training mode (adding dropout) or in inference mode (doing nothing)
        mask: padding mask in the multi-head attention layer
        
        returns: [batch_size, num_patches, d_model]
        """
        x = self.layernorm(inputs)
        attn_output, att_weights = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=True, training=training)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        res = attn_output + inputs
        ffn_output = self.ffn(res)  # (batch_size, input_seq_len, d_model)

        return ffn_output + res, attn_output, att_weights, res

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, head_size, dff, input_dim, rate=0.1):
        """
        num_layers: number of encoder layers
        num_heads: number of heads in the multi-head attention layer
        head_size: dimension of the head embedding
        dff: dimension of the feed forward network
        input_dim: dimension of the input
        """
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(num_heads, head_size, dff, input_dim, rate) for _ in range(num_layers)]

    def call(self, x, training, mask):

        """
        x: [batch_size, num_patches, d_model]
        training: boolean, specifies whether the dropout layer should behave in training mode (adding dropout) or in inference mode (doing nothing)
        mask: padding mask in the multi-head attention layer
        
        return: [batch_size, num_patches, d_model]
        """

        attention_weights = {}
        attention_outputs = {}
        attention_outputs_sum = {}

        for i in range(self.num_layers):
            x, att_output, att_weights, att_output_sum = self.enc_layers[i](x, training, mask)
            attention_outputs[f'encoder_layer{i+1}'] = att_output
            attention_weights[f'encoder_layer{i+1}'] = att_weights
            attention_outputs_sum[f'encoder_layer{i+1}'] = att_output_sum

        return x, attention_outputs, attention_weights, attention_outputs_sum
# ----------------------------------------------------------------------------

# ------------------------------- Transformer --------------------------------
class ICFormer(tf.keras.Model):
    def __init__(self, look_back=12, n_features=128, num_transformer_blocks=2, num_heads=2, head_size=32, ff_dim=32, mlp_units=128, mlp_dropout=0.0, dropout=0.0):
        """
        """
        super().__init__()
        self.input_layer = tf.keras.layers.Input(shape=(look_back, n_features))
        self.patch_embedding = PatchEmbedding(look_back, mlp_units)
        self.encoder = Encoder(num_transformer_blocks, num_heads, head_size, ff_dim, mlp_units, rate=dropout)
                                    
        #self.decoder = Decoder(num_layers, d_model, num_heads, dff,
        #						num_patches_decoder, rate)

        self.mlp_layer =  keras.Sequential([
                                layers.GlobalAveragePooling1D(data_format="channels_first"),
                                layers.Dense(mlp_units, activation="relu"),
                            ])
        
        self.dropout = layers.Dropout(mlp_dropout)
        
        self.regression_layer = tf.keras.layers.Dense(3*look_back, activation='linear', name='regression_output')
        self.classification_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='classification_output')

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.regression_tracker = tf.keras.metrics.MeanSquaredError(name='regression')
        self.classification_tracker = tf.keras.metrics.BinaryCrossentropy(name='classification')
        self.regression_mae_tracker = tf.keras.metrics.MeanAbsoluteError(name='regression_mae')
        self.classification_acc_tracker = tf.keras.metrics.BinaryAccuracy(name='classification')
    
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        self.val_regression_tracker = tf.keras.metrics.MeanSquaredError(name='val_regression')
        self.val_classification_tracker = tf.keras.metrics.BinaryCrossentropy(name='val_classification')
        self.val_regression_mae_tracker = tf.keras.metrics.MeanAbsoluteError(name='val_regression_mae')
        self.val_classification_acc_tracker = tf.keras.metrics.BinaryAccuracy(name='val_classification_accuracy')
       
	
    # Loss and metrics
    def loss_function(self, y_reg, y_clf, regression_output, classification_output):

        loss_reg = tf.keras.losses.MeanSquaredError()
        loss_clf = tf.keras.losses.BinaryCrossentropy()
        loss_ = loss_reg(y_reg, regression_output) + loss_clf(y_clf, classification_output)*10
        
        return tf.reduce_mean(loss_)

    @property
    def metrics(self):
        # The 'Metric' objects is listed here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of 'evaluate()'.
        #return [self.loss_tracker, self.mape_tracker]
        return [self.loss_tracker, self.regression_tracker, self.classification_tracker, self.regression_mae_tracker, self.classification_acc_tracker,
                self.val_loss_tracker, self.val_regression_tracker, self.val_classification_tracker, self.val_regression_mae_tracker, self.val_classification_acc_tracker]

    @tf.function
    def train_step(self, data):
        # Unpack the data
        x, y = data
        y_reg = y['y_regression']
        y_clf = y['y_classification']

        with tf.GradientTape() as tape:
            regression_output, classification_output, _, _, _ = self.forward_pass(x)
            loss = self.loss_function(y_reg, y_clf, regression_output, classification_output)

        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update training metric
        self.loss_tracker.update_state(loss)
        self.regression_tracker.update_state(y_reg, regression_output)
        self.classification_tracker.update_state(y_clf, classification_output)
        self.regression_mae_tracker.update_state(y_reg, regression_output)
        self.classification_acc_tracker.update_state(y_clf, classification_output)
        return {"loss": self.loss_tracker.result(), "regression": self.regression_tracker.result(), "classification": self.classification_tracker.result(), "regression_mae": self.regression_mae_tracker.result(), "classification_acc": self.classification_acc_tracker.result()}

    @tf.function
    def test_step(self, data):
        x, y = data

        y_reg = y['y_regression']
        y_clf = y['y_classification']
        regression_output, classification_output, _, _, _ = self.forward_pass(x)
        loss = self.loss_function(y_reg, y_clf, regression_output, classification_output)
        
        # Update val metrics
        self.val_loss_tracker.update_state(loss)

        self.val_regression_tracker.update_state(y_reg, regression_output)
        self.val_classification_tracker.update_state(y_clf, classification_output)
        self.val_regression_mae_tracker.update_state(y_reg, regression_output)
        self.val_classification_acc_tracker.update_state(y_clf, classification_output)

        return {"loss": self.val_loss_tracker.result(), "regression": self.val_regression_tracker.result(), "classification": self.val_classification_tracker.result(), "regression_mae": self.val_regression_mae_tracker.result(), "classification_acc": self.val_classification_acc_tracker.result()}

    def call(self, x):
        return self.forward_pass(x, training=False)

    def forward_pass(self, x, training=True):
        encoder_inputs = self.patch_embedding(x) # patch embedding
        encoder_outputs, attention_outputs, attention_weights, attention_outputs_sum = self.encoder(encoder_inputs, training=training, mask=None) # transformer encoder
        mlp_output = self.mlp_layer(encoder_outputs)
        mlp_output = self.dropout(mlp_output, training=training)

        regression_output = self.regression_layer(mlp_output) # regression layer
        classification_output = self.classification_layer(mlp_output) # classification layer
        return regression_output, classification_output, attention_outputs, attention_weights, attention_outputs_sum
# -----------------------------------------------------------------------