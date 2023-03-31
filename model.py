import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.callbacks import TensorBoard,ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold, train_test_split
from MHA_custom import *



def _build_proj_equation(free_dims, bound_dims, output_dims):
    _CHR_IDX = string.ascii_lowercase
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)

def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))
        # self.flatten_fe = layers.Reshape(target_shape=(int(input_shape[0]/patch_size[0]),
        #                                             int(input_shape[1]/patch_size[1])*int(input_shape[2]/patch_size[2]), embed_dim))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projection': self.projection,
            'flatten': self.flatten,
            #'flatten_fe': self.flatten_fe,
        })
        return config
    
    def call(self, videos, vivit_model):
        projected_patches = self.projection(videos)
        if vivit_model == 'FEE':
            flattened_patches = self.flatten_fe(projected_patches)
        else:
            flattened_patches = self.flatten(projected_patches)        
        return flattened_patches
    
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config
    
    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
    
    
def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    vivit_model,
    input_shape=None,
    patch_size=None,
    transformer_layers=None,
    num_heads=None,
    embed_dim=None,
    layer_norm_eps=None,
    num_classes=None,
    ):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)

    if vivit_model == 'STA':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
    
        # Encode patches.
        encoded_patches = positional_encoder(patches)

        
        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization and MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)
            print("attention_output :",attention_output.shape)
            # Skip connection
            x2 = layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])        
        
    elif vivit_model == 'FE':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
        encoded_patches = positional_encoder(patches) 
        #print("pt_encoded_patches :",encoded_patches.shape)

        for t in range(int(input_shape[0]/patch_size[0])):
            n_t = int(input_shape[0]/patch_size[0])
            n_w = int(input_shape[1]/patch_size[1])
            n_h = int(input_shape[2]/patch_size[2])
            p_encoded_patches = layers.Reshape(target_shape=(n_t, n_w*n_h, embed_dim))(encoded_patches)
            # MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(p_encoded_patches[:,t])
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)

            # Skip connection
            x2 = layers.Add()([attention_output, p_encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)
            
            # Skip connection
            encoded_patch = layers.Add()([x3, x2])
            
            if t == 0: t_encoded_patches = encoded_patch
            else: t_encoded_patches = layers.Concatenate(axis=1)([t_encoded_patches, encoded_patch])
        
        for _ in range(transformer_layers):
            # MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(t_encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)

            # Skip connection
            x2 = layers.Add()([attention_output, t_encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])

    elif vivit_model == 'FSA':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
    
        # Encode patches.
        encoded_patches = positional_encoder(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            n_t = int(input_shape[0]/patch_size[0])
            n_w = int(input_shape[1]/patch_size[1])
            n_h = int(input_shape[2]/patch_size[2])
            # Spatial MHSA
            x1 = layers.Reshape(target_shape=(n_t, n_w*n_h, embed_dim))(encoded_patches)
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(2,3)
            )(x1, x1)
            x2 = layers.Add()([attention_output, x1])

#             x2 = layers.Reshape(target_shape=(n_t, n_w, n_h, embed_dim))(x2)
#             x2 = layers.Permute((2,3,1,4))(x2)

            # Temporal MHSA
#             x2 = layers.Reshape(target_shape=(-1, n_t*embed_dim))(x2)
            x2 = layers.LayerNormalization(epsilon=1e-6)(x2)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(1,3)
            )(x2, x2)
            x3 = layers.Add()([attention_output, x2])              
            
#             x3 = layers.Reshape(target_shape=(n_w, n_h, n_t, embed_dim))(x3)
#             x3 = layers.Permute((3,1,2,4))(x3)
            
            x3 = layers.Reshape(target_shape=(-1, embed_dim))(x3)
   
            # Layer Normalization and MLP
            x4 = layers.LayerNormalization(epsilon=1e-6)(x3)
            x4 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x4)

            # Skip connection
            encoded_patches = layers.Add()([x4, x3])
        
    elif vivit_model == 'FDPA':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
        # Encode patches.
        encoded_patches = positional_encoder(patches)
        
        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            n_t = int(input_shape[0]/patch_size[0])
            n_w = int(input_shape[1]/patch_size[1])
            n_h = int(input_shape[2]/patch_size[2])
            
            x1 = layers.Reshape(target_shape=(n_t, n_w*n_h, embed_dim))(encoded_patches)
            
            # Layer normalization and MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
            sp_attention_output = MultiHeadAttention_nonlinear( #, sp_attention_weights
                num_heads=num_heads//2, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(2,3)
            )(x1, x1) #, return_attention_scores=True
            tm_attention_output = MultiHeadAttention_nonlinear( #, tm_attention_weights
                num_heads=num_heads//2, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(1,3)
            )(x1, x1) #, return_attention_scores=True
                        
            #print("sp_attention_output :",sp_attention_output.shape)
            #print("sp_attention_weights :",sp_attention_weights.shape)
            #print("tm_attention_output :",tm_attention_output.shape)
            #print("tm_attention_weights :",tm_attention_weights.shape)
                    
            attention_output = layers.Concatenate(axis=-1)([sp_attention_output, tm_attention_output])
            #print("attention_output_1 :",attention_output.shape)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(3, 2, 1)
            
            #print(einsum_equation)
            print(output_rank)
            print(bias_axes)
            
            attention_output = layers.EinsumDense(einsum_equation,
                                                  output_shape=_get_output_shape(output_rank - 1,[embed_dim])
                                                  ,bias_axes=bias_axes)(attention_output)
            
            attention_output = layers.Reshape(target_shape=(-1, embed_dim))(attention_output)            
            # Skip connection
            x2 = layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])
    
    else: raise Exception(f'ViViT model method is not defined correctly : \'{vivit_model}\'. Please define a model method as following. [\'STA\', \'FE\', \'FSA\', \'FDPA\']')
        
    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    
    # Classify outputs.
    if num_classes == 1:
        outputs = layers.Dense(units=num_classes, activation="sigmoid")(representation)
    else:
        outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model