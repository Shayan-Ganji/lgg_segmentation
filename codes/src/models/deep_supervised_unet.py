import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import CFG

class CustomGroupNormalization(layers.Layer):
    """Batch-size independent Group Normalization layer engineered for v3."""
    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super(CustomGroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],),
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],),
                                    initializer='zeros', trainable=True)
        super(CustomGroupNormalization, self).build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        x_reshaped = tf.reshape(x, [N, H, W, self.groups, C // self.groups])
        mean, var = tf.nn.moments(x_reshaped, axes=[1, 2, 4], keepdims=True)
        x_normalized = (x_reshaped - mean) / tf.sqrt(var + self.epsilon)
        x_normalized = tf.reshape(x_normalized, input_shape)
        return self.gamma * x_normalized + self.beta

def build_deep_supervised_unet(input_shape=(CFG["IMG_SIZE"], CFG["IMG_SIZE"], 1)) -> Model:
    """nnU-Net Style Deeply Supervised Network with multi-resolution auxiliary outputs."""
    inputs = layers.Input(input_shape)
    
    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = CustomGroupNormalization(groups=CFG["GROUP_NORM_G"])(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        
        x = layers.Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = CustomGroupNormalization(groups=CFG["GROUP_NORM_G"])(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        return x
        
    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)
    
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)
    
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)
    
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D()(c4)
    
    # Bottleneck
    b = conv_block(p4, 1024)
    
    # Decoder with Deep Supervision
    u4 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(b)
    u4 = layers.Concatenate()([u4, e4 := c4])
    d4 = conv_block(u4, 512)
    out4 = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    out4 = layers.UpSampling2D(size=(8, 8), interpolation='bilinear', name='out4')(out4)
    
    u3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(d4)
    u3 = layers.Concatenate()([u3, e3 := c3])
    d3 = conv_block(u3, 256)
    out3 = layers.Conv2D(1, 1, activation='sigmoid')(d3)
    out3 = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='out3')(out3)
    
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d3)
    u2 = layers.Concatenate()([u2, e2 := c2])
    d2 = conv_block(u2, 128)
    out2 = layers.Conv2D(1, 1, activation='sigmoid')(d2)
    out2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='out2')(out2)
    
    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d2)
    u1 = layers.Concatenate()([u1, e1 := c1])
    d1 = conv_block(u1, 64)
    out1 = layers.Conv2D(1, 1, activation='sigmoid', name='out1')(d1)
    
    return Model(inputs, [out1, out2, out3, out4], name="DeepSupervised_UNet")