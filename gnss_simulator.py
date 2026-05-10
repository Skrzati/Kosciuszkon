import numpy as np
import pandas as pd


class RealisticGNSSSimulator:
    def __init__(self, samples=86400):
        self.samples = samples
        self.antennas = ['N', 'S', 'E', 'W']

    def _generate_multipath_fading(self):
        """Symulacja zaników Rayleigha (środowisko miejskie/niskie pułapy)."""
        rayleigh = np.random.rayleigh(scale=1.0, size=self.samples)
        return rayleigh * 2.0  # Wpływ na SNR

    def _inject_seamless_takeover(self, df: pd.DataFrame, start_idx: int, duration: int, attack_type: int):
        """Wstrzykuje koherentny atak z uwzględnieniem fizyki Angle of Arrival (AoA) i Code-Carrier Divergence."""
        end_idx = min(start_idx + duration, self.samples)
        actual_duration = end_idx - start_idx

        if actual_duration <= 0: return

        # 1. Kinematyka ataku (powolny dryf)
        accel_drift = np.random.uniform(0.1, 0.5) if attack_type == 1 else np.random.uniform(0.5, 2.0)
        t = np.arange(actual_duration)
        drift_distance = 0.5 * accel_drift * (t ** 2)
        doppler_shift = accel_drift * t

        # 2. Szum fazowy taniego oscylatora (SDR Spoofer fingerprint)
        spoofer_phase_noise = np.random.normal(0, 0.8, actual_duration)

        for ant in self.antennas:
            # KLUCZOWE: Angle of Arrival (AoA) Bias
            # Ponieważ spoofer nadaje z jednego punktu (np. z ziemi), rzut wektora fali na matrycę anten
            # jest asymetryczny. Wprowadzamy stałe przesunięcie fazowe per antena.
            aoa_bias = np.random.uniform(-2.0, 2.0)

            # KLUCZOWE: Code-Carrier Divergence (Rozbieżność pętli DLL i PLL)
            # W tanich radiach faza nośna (Doppler) i kod (Pseudorange) rozjeżdżają się w czasie.
            ccd_drift = np.linspace(0, np.random.uniform(1.0, 4.0), actual_duration)

            # Aplikacja zniekształceń do pomiarów
            df.loc[start_idx:end_idx - 1, f'{ant}_pseudorange'] += (drift_distance + aoa_bias + spoofer_phase_noise)

            # Doppler otrzymuje CCD Drift, co sprawi, że pochodna Pseudorange przestanie do niego pasować!
            df.loc[start_idx:end_idx - 1, f'{ant}_doppler'] += (doppler_shift + ccd_drift)

            # Charakterystyka siły sygnału
            snr_boost = np.linspace(0, 15, actual_duration) if attack_type == 1 else np.linspace(0, 5, actual_duration)
            df.loc[start_idx:end_idx - 1, f'{ant}_snr'] += snr_boost

        df.loc[start_idx:end_idx - 1, 'label'] = attack_type

    def generate(self, filename='blue_shield_synthetic.csv'):
        print("Generowanie realistycznej dystrybucji RF i ataków stochastycznych (Stratified)...")

        t_seconds = np.arange(self.samples)
        df_data = {'timestamp': t_seconds, 'label': np.zeros(self.samples, dtype=int)}

        # Bazowe parametry GNSS
        for ant in self.antennas:
            base_distance = 20000000 + np.sin(t_seconds / 3600) * 500000
            df_data[f'{ant}_pseudorange'] = base_distance + np.random.normal(0, 1.5, self.samples)
            df_data[f'{ant}_snr'] = 45.0 - self._generate_multipath_fading()
            df_data[f'{ant}_doppler'] = np.cos(t_seconds / 3600) * 800 + np.random.normal(0, 0.5, self.samples)

        df = pd.DataFrame(df_data)

        # GWARANCJA KLAS W OBU ZBIORACH (Stratified Random Injection)
        # Dzielimy dobę na 4 równe bloki czasowe.
        # Skoro chronologiczny split to 70%, pierwsze 3 bloki wejdą do Treningu, a ostatni do Testu.

        blocks = [
            (3600, 20000),  # Blok 1 (Trening)
            (25000, 40000),  # Blok 2 (Trening)
            (45000, 60000),  # Blok 3 (Trening)
            (65000, 82000)  # Blok 4 (Test)
        ]

        print("Wstrzykiwanie ataków z gwarancją reprezentacji w czasie...")
        num_attacks = 0
        for start_limit, end_limit in blocks:
            # W każdym bloku losujemy moment rozpoczęcia ataku naziemnego
            idx_ground = np.random.randint(start_limit, end_limit - 3000)
            dur_ground = int(np.random.normal(1200, 200))  # Ok. 20 minut
            self._inject_seamless_takeover(df, idx_ground, dur_ground, 1)

            # W każdym bloku losujemy moment rozpoczęcia ataku powietrznego
            idx_air = np.random.randint(start_limit, end_limit - 3000)
            dur_air = int(np.random.normal(1500, 300))  # Ok. 25 minut
            self._inject_seamless_takeover(df, idx_air, dur_air, 2)

            num_attacks += 2

        print(f"✅ Z powodzeniem wstrzyknięto {num_attacks} zbalansowanych czasowo ataków.")

        df.to_csv(filename, index=False)
        print("✅ Generacja zakończona.")


if __name__ == "__main__":
    RealisticGNSSSimulator().generate()