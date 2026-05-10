import pandas as pd


def compute_double_differences(df: pd.DataFrame, ref_ant: str = 'N') -> pd.DataFrame:
    """
    Oblicza Single Differences (SD) oraz rezydua Dopplera.
    Eliminuje to zjawisko uczenia się przez model konkretnej trajektorii satelity,
    zmuszając go do szukania anomalii w korelacji fazy między antenami.
    """
    df_features = pd.DataFrame()
    other_ants = [ant for ant in ['S', 'E', 'W'] if ant != ref_ant]

    for ant in other_ants:
        # Różnica pseudoodległości między antenami (eliminuje błąd zegara satelity)
        sd_col = f'SD_{ant}_{ref_ant}'
        df_features[sd_col] = df[f'{ant}_pseudorange'] - df[f'{ref_ant}_pseudorange']

        # Rezydua Dopplera (wykrywanie nienaturalnych skoków prędkości w sygnale)
        # Pochodna z pseudoodległości powinna z grubsza zgadzać się z Dopplerem
        df_features[f'Residual_Doppler_{ant}'] = df[f'{ant}_doppler'] - df[f'{ant}_pseudorange'].diff()

        # Różnica w SNR
        df_features[f'SNR_diff_{ant}_{ref_ant}'] = df[f'{ant}_snr'] - df[f'{ref_ant}_snr']

    # Dodajemy zmienne środowiskowe, które mogą maskować ataki
    for col in ['kp_index', 'temperature', 'pressure', 'humidity']:
        if col in df.columns:
            df_features[col] = df[col]

    # Label kopiujemy na koniec
    df_features['label'] = df['label']

    # Usunięcie NaN powstałych po operacji diff()
    return df_features.fillna(0)