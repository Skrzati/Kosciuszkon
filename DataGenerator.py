import pandas as pd
import numpy as np
import requests
from skyfield.api import load, wgs84


# ZMIANA: Zmniejszono domyślną ilość próbek do 3600 (1 godzina symulacji)
def generate_meteo_blue_shield(samples=3600, filename='blue_shield_meteo_data_small.csv'):
    print(f"Tworzenie symulacji ({samples} sekund)...")
    print("1. Łączenie z serwerem Celestrak...")

    tle_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle'
    local_tle_file = 'gps_real_data.txt'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    try:
        response = requests.get(tle_url, headers=headers)
        if response.status_code == 200:
            content = response.text
            if "<html" in content.lower():
                raise ConnectionError("Serwer zwrócił stronę HTML. Spróbuj za chwilę.")
            with open(local_tle_file, 'w') as f:
                f.write(content)
        else:
            raise ConnectionError("Błąd pobierania TLE.")
    except Exception as e:
        print(f"Błąd sieci: {e}. Używam istniejącego pliku TLE (jeśli istnieje).")

    ts = load.timescale()
    satellites = load.tle_file(local_tle_file)
    sat = list({s.name: s for s in satellites}.values())[0]
    print(f"Wybrano satelitę: {sat.name}")

    t_seconds = np.arange(samples)
    t = ts.utc(2026, 5, 10, 0, 0, t_seconds)

    antennas = {
        'N': wgs84.latlon(54.40, 18.64), 'S': wgs84.latlon(54.30, 18.64),
        'E': wgs84.latlon(54.35, 18.70), 'W': wgs84.latlon(54.35, 18.58)
    }

    print("2. Generowanie realistycznej pogody...")
    t_hours = t_seconds / 3600.0

    temp_c = 15 + 8 * np.sin(2 * np.pi * (t_hours - 9) / 24) + np.random.normal(0, 0.2, samples)
    humidity = 70 + 20 * np.cos(2 * np.pi * (t_hours - 9) / 24) + np.random.normal(0, 1.5, samples)
    humidity = np.clip(humidity, 10, 100)
    pressure = 1013 + np.cumsum(np.random.normal(0, 0.05, samples))
    pressure = np.clip(pressure, 990, 1030)

    kp_index = np.random.choice([1, 2, 3, 5], samples, p=[0.7, 0.2, 0.08, 0.02])

    df_data = {
        'timestamp': t_seconds,
        'temperature': temp_c,
        'pressure': pressure,
        'humidity': humidity,
        'kp_index': kp_index
    }

    print("3. Obliczanie fizyki orbitalnej i opóźnień troposferycznych...")
    for name, location in antennas.items():
        difference = sat - location
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()

        alt_degrees = np.maximum(alt.degrees, 5.0)

        e = (humidity / 100.0) * 6.11 * np.exp((17.27 * temp_c) / (237.3 + temp_c))
        zhd = 0.0022768 * pressure
        zwd = 0.002277 * (1255 / (temp_c + 273.15) + 0.05) * e
        tropo_delay = (zhd + zwd) / np.sin(np.radians(alt_degrees))

        base_noise = np.random.normal(0, 0.5, samples) * (1 + kp_index / 5)
        multipath = np.random.exponential(scale=2.5, size=samples)

        df_data[f'{name}_pseudorange'] = distance.m + tropo_delay + base_noise + multipath

        base_snr = 30 + (alt_degrees / 90.0) * 20
        df_data[f'{name}_snr'] = base_snr - (humidity / 100.0 * 2) + np.random.normal(0, 1, samples)

        range_rate = np.gradient(distance.m)
        doppler_shift = -(range_rate / 299792458.0) * 1575.42e6
        df_data[f'{name}_doppler'] = doppler_shift + np.random.normal(0, 1.5, samples)

    labels = np.zeros(samples)

    print("4. Wstrzykiwanie Ataków...")

    # ZMIANA: Czas trwania ataku jest teraz dynamiczny (np. 10% długości całej symulacji)
    len_g = max(10, int(samples * 0.10))
    start_g = int(samples * 0.15)

    for name in antennas.keys():
        bias = np.random.uniform(500, 2000) if name in ['N', 'E'] else np.random.uniform(-50, 50)
        df_data[f'{name}_pseudorange'][start_g:start_g + len_g] += bias
        df_data[f'{name}_snr'][start_g:start_g + len_g] += 30
    labels[start_g:start_g + len_g] = 1

    # ZMIANA: Atak powietrzny zajmuje 15% długości symulacji
    len_a = max(10, int(samples * 0.15))
    start_a = int(samples * 0.70)

    for name in antennas.keys():
        df_data[f'{name}_pseudorange'][start_a:start_a + len_a] -= np.linspace(0, 400, len_a)
        df_data[f'{name}_snr'][start_a:start_a + len_a] += 15
        df_data[f'{name}_doppler'][start_a:start_a + len_a] += 200
    labels[start_a:start_a + len_a] = 2

    df_data['label'] = labels
    df = pd.DataFrame(df_data)

    df = df.iloc[10:-10].reset_index(drop=True)

    df.to_csv(filename, index=False)
    print(f"✅ SUKCES: Wygenerowano zredukowaną paczkę danych ({filename})")


if __name__ == '__main__':
    # Jeśli chcesz np. wygenerować równe 5000 próbek, wystarczy że podasz to tu:
    generate_meteo_blue_shield(samples=3600)