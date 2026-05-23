import os
import math
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from src.config import CFG
from src.dataset import build_dataset
from src.models.attention_resunet import build_attention_res_unet
from src.losses import compound_production_loss
from src.metrics import soft_dice_coef

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for warmup_epochs, then cosine decay down to lr_min."""
    def __init__(self, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        wu_lr = self.lr_max * (step / self.warmup_steps)
        prog = tf.clip_by_value((step - self.warmup_steps) / (self.total_steps - self.warmup_steps), 0.0, 1.0)
        cos_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(math.pi * prog))
        return tf.where(step < self.warmup_steps, wu_lr, cos_lr)

    def get_config(self):
        return dict(lr_max=self.lr_max, lr_min=self.lr_min,
                    warmup_steps=self.warmup_steps, total_steps=self.total_steps)

def train_fold(fold: int, df_tr, df_vl, steps_per_epoch: int) -> str:
    """Trains a single cross-validation fold using the production Attention Res-UNet pipeline."""
    print(f"\n{'='*65}")
    print(f"  TRAINING FOLD {fold+1}/{CFG['N_FOLDS']} | Train Slices: {len(df_tr)} | Val Slices: {len(df_vl)}")
    print(f"{'='*65}")

    ckpt_path = os.path.join(CFG["OUTPUT_DIR"], f"fold{fold+1}_best.keras")

    ds_tr = build_dataset(df_tr, training=True, batch_size=CFG["BATCH_SIZE"])
    ds_vl = build_dataset(df_vl, training=False, batch_size=CFG["BATCH_SIZE"])

    model = build_attention_res_unet()

    total_steps = CFG["EPOCHS"] * steps_per_epoch
    warmup_steps = CFG["WARMUP_EPOCHS"] * steps_per_epoch
    schedule = WarmupCosineDecay(
        lr_max=CFG["LR_INITIAL"], lr_min=CFG["LR_MIN"],
        warmup_steps=warmup_steps, total_steps=total_steps
    )

    opt = tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=CFG["WEIGHT_DECAY"],
        clipnorm=CFG["GRAD_CLIP"]
    )

    model.compile(
        optimizer=opt,
        loss={
            "main": compound_production_loss,
            "ds_half_act": compound_production_loss,
            "ds_quarter_act": compound_production_loss,
        },
        loss_weights={
            "main": 1.0,
            "ds_half_act": CFG["DS_WEIGHT"],
            "ds_quarter_act": CFG["DS_WEIGHT"],
        },
        metrics={"main": [soft_dice_coef]},
    )

    def wrap_targets(img, msk):
        return img, {"main": msk, "ds_half_act": msk, "ds_quarter_act": msk}

    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_main_soft_dice_coef", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_main_soft_dice_coef", mode="max", patience=CFG["ES_PATIENCE"], restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(CFG["OUTPUT_DIR"], f"fold{fold+1}_log.csv"))
    ]

    model.fit(
        ds_tr.map(wrap_targets),
        validation_data=ds_vl.map(wrap_targets),
        epochs=CFG["EPOCHS"],
        callbacks=callbacks,
        verbose=1
    )
    
    return ckpt_path