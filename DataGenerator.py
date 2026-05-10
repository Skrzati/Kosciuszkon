import pandas as pd
import numpy as np
import requests
from skyfield.api import load, wgs84


def get_real_weather_data(samples=86400):
    print("   -> Pobieranie rzeczywistych danych meteo (Open-Meteo dla Gdańska)...")

    # Darmowe API Open-Meteo (Gdańsk: 54.35N, 18.64E) - pobieramy minione 24h
    url = "https://api.open-meteo.com/v1/forecast?latitude=54.35&longitude=18.64&hourly=temperature_2m,relative_humidity_2m,surface_pressure&past_days=1&forecast_days=0"

    try:
        response = requests.get(url).json()

        # Pobieramy równe 24 godziny danych (24 punkty)
        hourly_temp = response['hourly']['temperature_2m'][:24]
        hourly_hum = response['hourly']['relative_humidity_2m'][:24]
        hourly_press = response['hourly']['surface_pressure'][:24]

        # Interpolacja liniowa (rozciągnięcie 24 punktów na 86 400 sekund)
        t_hourly = np.linspace(0, samples, 24)
        t_seconds = np.arange(samples)

        temp_c = np.interp(t_seconds, t_hourly, hourly_temp)
        humidity = np.interp(t_seconds, t_hourly, hourly_hum)
        pressure = np.interp(t_seconds, t_hourly, hourly_press)

        # Dodajemy mikroszumy sekundowe (np. nagłe podmuchy wiatru, wahania czujnika)
        temp_c += np.random.normal(0, 0.05, samples)
        humidity += np.random.normal(0, 0.2, samples)
        pressure += np.random.normal(0, 0.02, samples)

        print("   ✅ Pomyślnie pobrano i wyinterpolowano rzeczywistą pogodę!")
        return temp_c, humidity, pressure

    except Exception as e:
        print(f"   ❌ Błąd pobierania pogody: {e}. Przełączam na tryb sztuczny.")
        # Fallback - jeśli brak internetu, użyj starej metody z sin/cos
        t_hours = np.arange(samples) / 3600.0
        temp_c = 15 + 8 * np.sin(2 * np.pi * (t_hours - 9) / 24) + np.random.normal(0, 0.2, samples)
        humidity = 70 + 20 * np.cos(2 * np.pi * (t_hours - 9) / 24) + np.random.normal(0, 1.5, samples)
        humidity = np.clip(humidity, 10, 100)
        pressure = 1013 + np.cumsum(np.random.normal(0, 0.05, samples))
        pressure = np.clip(pressure, 990, 1030)
        return temp_c, humidity, pressure


# Powrót do 86400 sekund (pełne 24 godziny) i oryginalnej nazwy pliku
def generate_meteo_blue_shield(samples=86400, filename='blue_shield_meteo_data.csv'):
    print(f"Tworzenie dobowej symulacji ({samples} sekund)...")
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

    print("2. Generowanie realistycznej pogody (wywołanie API)...")

    # Zastąpiona logika pogodowa - teraz korzystamy z funkcji wyżej
    temp_c, humidity, pressure = get_real_weather_data(samples)

    # Kp-index pozostaje bez zmian (to pogoda kosmiczna, trudniejsza do prostego wpięcia z API)
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

    # Atak naziemny (1 godzina = 3600 sekund)
    start_g = int(samples * 0.15)
    len_g = min(3600, int(samples * 0.8))  # Zabezpieczenie na wypadek bardzo małej próbki

    for name in antennas.keys():
        bias = np.random.uniform(500, 2000) if name in ['N', 'E'] else np.random.uniform(-50, 50)
        df_data[f'{name}_pseudorange'][start_g:start_g + len_g] += bias
        df_data[f'{name}_snr'][start_g:start_g + len_g] += 30
    labels[start_g:start_g + len_g] = 1

    # Atak powietrzny (1.5 godziny = 5400 sekund)
    start_a = int(samples * 0.70)
    len_a = min(5400, int(samples * 0.25))

    for name in antennas.keys():
        df_data[f'{name}_pseudorange'][start_a:start_a + len_a] -= np.linspace(0, 400, len_a)
        df_data[f'{name}_snr'][start_a:start_a + len_a] += 15
        df_data[f'{name}_doppler'][start_a:start_a + len_a] += 200
    labels[start_a:start_a + len_a] = 2

    df_data['label'] = labels
    df = pd.DataFrame(df_data)

    # Wycinamy marginesy po funkcji gradient
    df = df.iloc[10:-10].reset_index(drop=True)

    df.to_csv(filename, index=False)
    print(f"✅ SUKCES: Wygenerowano dobę danych ze zjawiskami pogody ({filename})")


if __name__ == '__main__':
    # Generujemy z powrotem pełne 86 400
    generate_meteo_blue_shield(samples=86400)