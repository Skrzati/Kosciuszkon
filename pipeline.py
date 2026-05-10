import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit


class GNSSDataPipeline:
    def __init__(self, window_size=10, batch_size=32):
        self.window_size = window_size
        self.batch_size = batch_size
        self.scaler = RobustScaler()  # Odporny na anomalie (spoofing spikes)

    def prepare_streaming_dataset(self, features: np.ndarray, labels: np.ndarray, is_training: bool = True):
        """Tworzy bezwyciekowy tf.data.Dataset z oknami przesuwnymi."""

        # Fit skalera TYLKO na danych treningowych
        if is_training:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        # Generowanie okien bez alokowania nadmiarowej pamięci RAM
        dataset = dataset.window(self.window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size + 1))

        def split_window(window):
            return window[:-1], window[-1][-1]  # Cechy, Label

        dataset = dataset.map(split_window, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            # Shuffle odbywa się NA OKNACH, nie na surowych próbkach czasowych
            dataset = dataset.shuffle(buffer_size=10000)

        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def walk_forward_validation(self, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_index, test_index in tscv.split(X):
            yield train_index, test_index