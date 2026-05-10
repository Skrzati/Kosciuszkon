import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import RobustScaler


class GNSSDataPipeline:
    def __init__(self, window_size=10, batch_size=32):
        self.window_size = window_size
        self.batch_size = batch_size
        # RobustScaler jest odporny na gigantyczne skoki (outliery) wywołane spoofingiem
        self.scaler = RobustScaler()

    def prepare_dataset(self, features, labels, is_training=True):
        """Przygotowuje chronologiczny potok danych (Sliding Window) bez wycieków."""
        # 1. Bezpieczne skalowanie
        if is_training:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        # 2. Tworzenie okien (zoptymalizowana pętla)
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i: i + self.window_size])
            # Etykieta odpowiada stanowi na końcu okna czasowego
            y.append(labels[i + self.window_size - 1])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # 3. Konwersja do TensorFlow Dataset dla wydajności
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if is_training:
            # Tasujemy tylko całe okna, nie pojedyncze próbki!
            dataset = dataset.shuffle(10000)

        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


def build_spatiotemporal_model(window_size: int, num_features: int, num_classes: int = 3) -> models.Model:
    """Hybrydowa sieć CNN + BiLSTM + Attention do wykrywania anomalii czasoprzestrzennych."""
    inputs = layers.Input(shape=(window_size, num_features))

    # Spatial Feature Extraction (korelacje między antenami)
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)

    # Temporal Dynamics (dynamika sygnału w czasie)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    # Temporal Self-Attention (skupienie sieci na nagłych skokach w oknie)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attention])  # Połączenie rezydualne
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="GNSS_Defender_Flat")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model