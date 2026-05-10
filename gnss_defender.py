import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        # Rzutowanie y_true do formatu One-Hot
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[1])

        pt = tf.where(tf.equal(y_true_one_hot, 1), y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - pt, self.gamma)

        loss = -focal_weight * tf.math.log(pt)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


def build_transformer_defender(window_size: int, num_features: int, num_classes: int = 3) -> models.Model:
    inputs = layers.Input(shape=(window_size, num_features))

    # 1. Adversarial Robustness (Gaussian Noise Injection)
    # Zabezpiecza przed model evasion attacks poprzez dodanie szumu termicznego w trakcie treningu
    x = layers.GaussianNoise(0.01)(inputs)

    # 2. Temporal Feature Extraction
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.2)(x)  # Regularyzacja na poziomie cech (usuwa całe kanały)

    # 3. Transformer Encoder Block (Zastępuje przestarzałe BiLSTM)
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = models.Sequential([
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(32)
    ])
    ffn_output = ffn(x)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # 4. Uncertainty / Classification Head
    # BYŁO: x = layers.GlobalAveragePooling1D()(x)

    # NOWE: Pobieramy wynik z ostatniego kroku czasowego (t=60).
    # Transformer już zakodował w nim kontekst z całego okna.
    x = x[:, -1, :]

    x = layers.Dropout(0.4)(x)
    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="GNSS_Transformer_Edge")

    # Przechodzimy na stabilną funkcję straty + obniżamy Learning Rate
    # Zbalansowanie klas załatwimy samym argumentem class_weights w main.py
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),  # Obniżony z 1e-3
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]