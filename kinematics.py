import pandas as pd
import numpy as np


class GNSSEngine:
    def __init__(self, reference_antenna='N'):
        self.ref_ant = reference_antenna

    def compute_double_differences(self, df: pd.DataFrame, sat_id='PRN01') -> pd.DataFrame:
        """
        Oblicza Double-Differenced Pseudoranges eliminując błąd zegara.
        Wymaga danych z wielu satelitów (tutaj uproszczenie dla demonstracji koncepcji przestrzennej).
        """
        other_ants = [ant for ant in ['S', 'E', 'W'] if ant != self.ref_ant]

        for ant in other_ants:
            # Single Difference (pomiędzy antenami dla tego samego satelity)
            # Eliminuje błąd zegara satelity
            sd_col = f'SD_{ant}_{self.ref_ant}'
            df[sd_col] = df[f'{ant}_pseudorange'] - df[f'{self.ref_ant}_pseudorange']

            # W pełnym modelu tutaj następuje odjęcie SD satelity referencyjnego
            # tworząc Double Difference (DD), co eliminuje błąd zegara odbiornika.

            # Dodajemy pochodną w czasie (Doppler-derived residual)
            df[f'Residual_Doppler_{ant}'] = df[f'{ant}_doppler'] - df[f'{ant}_pseudorange'].diff()

        return df.fillna(0)