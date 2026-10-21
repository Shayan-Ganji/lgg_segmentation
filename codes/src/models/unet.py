import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import CFG

def build_vanilla_unet(input_shape=(CFG["IMG_SIZE"], CFG["IMG_SIZE"], 1)) -> Model:
    """
    Standard 2D U-Net Architecture (~31M parameters).
    Used as the baseline model in v1 and v2 optimization experiments.
    """
    inputs = layers.Input(input_shape)
    
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    # Encoder (Contracting Path)
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = conv_block(p4, 1024)
    
    # Decoder (Expanding Path)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 64)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    return Model(inputs=[inputs], outputs=[outputs], name="Vanilla_UNet")