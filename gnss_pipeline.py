import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler


class SpatiotemporalPipeline:
    def __init__(self, window_size=10, batch_size=64):
        self.window_size = window_size
        self.batch_size = batch_size
        self.scaler = RobustScaler()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ekstrakcja przestrzennych wektorów innowacji (Double Differences)."""
        ref_ant = 'N'
        features = pd.DataFrame()

        for ant in ['S', 'E', 'W']:
            # Single Difference (eliminuje błąd zegara satelity i jonosferę)
            features[f'SD_PR_{ant}'] = df[f'{ant}_pseudorange'] - df[f'{ref_ant}_pseudorange']
            # Różnica SNR - kluczowa dla ataku punktowego (spoofer) vs orbita
            features[f'SD_SNR_{ant}'] = df[f'{ant}_snr'] - df[f'{ref_ant}_snr']
            # Rezydua Dopplera (wykrywanie nienaturalnych przyspieszeń fazy)
            features[f'Doppler_Res_{ant}'] = df[f'{ant}_doppler'] - df[f'{ant}_pseudorange'].diff()

        features['label'] = df['label']
        return features.fillna(0)

    def create_walk_forward_dataset(self, X: np.ndarray, y: np.ndarray, is_training: bool):
        """Bezwyciekowe generowanie okien czasowych przy użyciu natywnego API Keras."""

        # 1. Skalowanie na chronologicznym rozkładzie
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # 2. Wyrównanie etykiet
        # Okno zawiera próbki od X[i] do X[i + window_size - 1].
        # Pragniemy, aby etykieta dla tego okna pochodziła z momentu końcowego okna.
        # Ucinamy początkowe etykiety, aby indeks 0 w wektorze 'targets'
        # odpowiadał końcowi pierwszego wygenerowanego okna.
        targets = y[self.window_size - 1:]

        # 3. Zoptymalizowany generator C++ pod spodem (zastępuje window + flat_map)
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=X,
            targets=targets,
            sequence_length=self.window_size,
            sequence_stride=1,
            shuffle=is_training,
            batch_size=self.batch_size
        )

        # 4. Prefetch do RAM/VRAM dla maksymalnej wydajności I/O
        return dataset.prefetch(tf.data.AUTOTUNE)