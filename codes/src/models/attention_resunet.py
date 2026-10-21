import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from src.config import CFG

class GroupNorm(layers.Layer):
    """Group Normalization layer (Wu & He 2018)."""
    def __init__(self, groups: int = 8, eps: float = 1e-5, **kw):
        super().__init__(**kw)
        self.groups = groups
        self.eps = eps

    def build(self, shape):
        C = int(shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, C), initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(1, 1, 1, C), initializer="zeros")

    def call(self, x):
        B, H, W, C = tf.unstack(tf.shape(x))
        G = self.groups
        xr = tf.reshape(x, [B, H, W, G, C // G])
        mu, va = tf.nn.moments(xr, axes=[1, 2, 4], keepdims=True)
        xr = (xr - mu) / tf.sqrt(va + self.eps)
        return tf.reshape(xr, [B, H, W, C]) * self.gamma + self.beta

    def get_config(self):
        c = super().get_config()
        c.update({"groups": self.groups, "eps": self.eps})
        return c

class StochasticDepth(layers.Layer):
    """Stochastic Depth / DropPath (Huang et al. 2016)."""
    def __init__(self, drop_prob: float = 0.10, **kw):
        super().__init__(**kw)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        B = tf.shape(x)[0]
        mask = tf.floor(keep + tf.random.uniform([B, 1, 1, 1], dtype=x.dtype))
        return x / keep * mask

    def get_config(self):
        c = super().get_config()
        c.update({"drop_prob": self.drop_prob})
        return c

def res_block(x, filters: int, groups: int = 8, drop_path: float = 0.0) -> tf.Tensor:
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        shortcut = GroupNorm(groups)(shortcut)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = GroupNorm(groups)(x)
    x = layers.Activation("gelu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = GroupNorm(groups)(x)

    if drop_path > 0:
        x = StochasticDepth(drop_path)(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("gelu")(x)
    return x

def attn_gate(g: tf.Tensor, s: tf.Tensor, f: int) -> tf.Tensor:
    """Additive attention gate (Oktay et al. 2018)."""
    g1 = layers.Conv2D(f, 1, padding="same")(g)
    s1 = layers.Conv2D(f, 1, padding="same")(s)
    psi = layers.Activation("relu")(layers.Add()([g1, s1]))
    psi = layers.Conv2D(1, 1, padding="same")(psi)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([s, psi])

def ds_head(x, target: int, name: str):
    x = layers.Conv2D(1, 1, padding="same", dtype="float32")(x)
    logits = layers.Resizing(target, target, interpolation="bilinear", name=name + "_logits")(x)
    probs = layers.Activation("sigmoid", dtype="float32", name=name)(logits)
    return probs

def build_attention_res_unet(img_size=CFG["IMG_SIZE"], in_ch=CFG["INPUT_CHANNELS"],
                             base_f=CFG["BASE_FILTERS"], groups=CFG["GROUP_NORM_G"],
                             dropout=CFG["DROPOUT"]):
    """
    Builds the production Attention Res-UNet.
    Returns:
      train_model: outputs [main_prob, ds_half, ds_quarter] for loss computation
      inference_model: outputs [main_prob, main_logits] for calibration & export
    """
    f = [base_f * (2 ** i) for i in range(5)]
    dp = [0.00, 0.05, 0.10, 0.15, 0.20]

    inp = Input(shape=(img_size, img_size, in_ch), name="image")

    # Encoder
    e1 = res_block(inp, f[0], groups, dp[0])
    p1 = layers.MaxPooling2D(2)(e1)

    e2 = res_block(p1, f[1], groups, dp[1])
    p2 = layers.MaxPooling2D(2)(e2)

    e3 = res_block(p2, f[2], groups, dp[2])
    p3 = layers.MaxPooling2D(2)(e3)

    e4 = res_block(p3, f[3], groups, dp[3])
    p4 = layers.MaxPooling2D(2)(e4)

    # Bottleneck
    bn = res_block(p4, f[4], groups, dp[4])
    bn = layers.SpatialDropout2D(dropout)(bn)

    # Decoder
    def dec(x, skip, filters, sdp):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        skip = attn_gate(g=x, s=skip, f=filters // 2)
        x = layers.Concatenate()([x, skip])
        return res_block(x, filters, groups, sdp)

    d4 = dec(bn, e4, f[3], dp[3])
    d3 = dec(d4, e3, f[2], dp[2])
    d2 = dec(d3, e2, f[1], dp[1])
    d1 = dec(d2, e1, f[0], dp[0])

    # Deep supervision heads (1/4 and 1/2 scale)
    ds_quarter = ds_head(d3, img_size, name="ds_quarter")
    ds_half = ds_head(d2, img_size, name="ds_half")

    # Primary full-resolution head: extract raw logits and probabilities
    main_logits = layers.Conv2D(1, 1, padding="same", dtype="float32", name="main_logits")(d1)
    main_prob = layers.Activation("sigmoid", dtype="float32", name="main")(main_logits)

    train_model = Model(inputs=inp, outputs=[main_prob, ds_half, ds_quarter], name="Attention_ResUNet_Train")
    inference_model = Model(inputs=inp, outputs=[main_prob, main_logits], name="Attention_ResUNet_Inference")

    return train_model, inference_model