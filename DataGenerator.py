import pandas as pd
import numpy as np
import requests
from skyfield.api import load, wgs84


def get_real_weather_data(samples=86400, day_offset=0):
    t_h = np.arange(samples) / 3600.0
    temp = (15 + day_offset) + 8 * np.sin(2 * np.pi * (t_h - 9) / 24) + np.random.normal(0, 0.5, samples)
    hum = (70 - day_offset * 2) + 20 * np.cos(2 * np.pi * (t_h - 9) / 24) + np.random.normal(0, 1.5, samples)
    press = (1013 + day_offset) + np.cumsum(np.random.normal(0, 0.05, samples))
    return temp, np.clip(hum, 10, 100), np.clip(press, 980, 1040)


def generate_national_blue_shield(filename, day_offset):
    samples = 86400
    ts = load.timescale()
    satellites = load.tle_file('gps_real_data.txt')
    sat = list({s.name: s for s in satellites}.values())[0]
    t_seconds = np.arange(samples)
    t = ts.utc(2026, 5, 10 + day_offset, 0, 0, t_seconds)

    # 9 GŁÓWNYCH STACJI REFERENCYJNYCH W POLSCE
    miasta = {
        'WAW': (52.23, 21.01), 'GDN': (54.35, 18.64), 'KRK': (50.06, 19.94),
        'WRO': (51.10, 17.03), 'POZ': (52.40, 16.92), 'BIA': (53.13, 23.16),
        'SZC': (53.42, 14.55), 'LUB': (51.24, 22.56), 'RZE': (50.04, 21.99)
    }
    antennas = {name: wgs84.latlon(lat, lon) for name, (lat, lon) in miasta.items()}

    temp, hum, press = get_real_weather_data(samples, day_offset)
    kp_index = np.random.choice([1, 2, 3, 7, 8], samples, p=[0.75, 0.1, 0.1, 0.03, 0.02])

    df_data = {'timestamp': t_seconds, 'temperature': temp, 'pressure': press, 'humidity': hum, 'kp_index': kp_index}

    # Symulacja satelitarna dla całej Polski
    for name, loc in antennas.items():
        diff = sat - loc
        topo = diff.at(t)
        alt, az, dist = topo.altaz()
        alt_deg = np.maximum(alt.degrees, 5.0)
        noise = np.random.normal(0, 1, samples) * (kp_index / 2.5)  # Szum burz słonecznych
        df_data[f'{name}_pseudorange'] = dist.m + noise + np.random.exponential(2.5, samples)
        df_data[f'{name}_snr'] = (30 + (alt_deg / 90.0) * 20) - noise
        df_data[f'{name}_doppler'] = -(np.gradient(dist.m) / 299792458.0) * 1575.42e6 + np.random.normal(0, 2, samples)

    labels = np.zeros(samples)

    def profile(l, mv, tp):
        tm = np.linspace(0, 1, l)
        if tp == 'linear': return mv * tm
        if tp == 'pulse': return np.where(np.sin(tm * 40 * np.pi) > 0, mv, 0)
        if tp == 'sine': return mv * np.sin(tm * 6 * np.pi)
        return np.full(l, mv) + np.random.normal(0, abs(mv) * 0.05, l)

    # NOWOŚĆ: EPICENTRA ATAKÓW
    def get_affected_cities(radius_km=200):
        # Losujemy punkt w Polsce
        atk_lat = np.random.uniform(50.0, 54.5)
        atk_lon = np.random.uniform(14.5, 23.5)
        affected = []
        for name, (lat, lon) in miasta.items():
            # Szybkie przybliżenie euklidesowe odległości w km
            dist = np.sqrt(((lat - atk_lat) * 111) ** 2 + ((lon - atk_lon) * 70) ** 2)
            if dist <= radius_km:
                affected.append(name)
        # Jeśli atak spadł w las i nikogo nie trafił, wymuszamy 1 miasto
        return affected if affected else [list(miasta.keys())[np.random.randint(0, 9)]]

    print(f" -> Generowanie ataków i anomalii (Dzień {day_offset})...")
    # 1. Ataki Naziemne (Zagłuszacze - mały zasięg 150 km)
    for _ in range(np.random.randint(2, 5)):
        l, s = np.random.randint(1800, 5000), np.random.randint(0, 80000)
        cele = get_affected_cities(radius_km=150)
        for n in cele: df_data[f'{n}_snr'][s:s + l] -= profile(l, 25, 'pulse')
        labels[s:s + l] = 1

    # 2. Ataki Powietrzne (Drony - średni zasięg 250 km)
    for _ in range(np.random.randint(1, 4)):
        l, s = np.random.randint(3600, 9000), np.random.randint(0, 75000)
        cele = get_affected_cities(radius_km=250)
        for n in cele: df_data[f'{n}_doppler'][s:s + l] += profile(l, 300, 'linear')
        labels[s:s + l] = np.maximum(labels[s:s + l], 2)

    # 3. Atak Replay (Szeroki zasięg)
    for _ in range(np.random.randint(1, 3)):
        l, s = np.random.randint(2000, 6000), np.random.randint(0, 80000)
        cele = get_affected_cities(radius_km=300)
        for n in cele: df_data[f'{n}_pseudorange'][s:s + l] += profile(l, 1000, 'sine')
        labels[s:s + l] = np.maximum(labels[s:s + l], 3)

    df_data['label'] = labels
    pd.DataFrame(df_data).iloc[10:-10].to_csv(filename, index=False)
    print(f"✅ ZAPISANO: {filename}")


if __name__ == '__main__':
    print("🚀 URUCHAMIANIE: KRAJOWY GENERATOR ZAGROŻEŃ GNSS")
    # Generujemy epicką ilość danych (5 Dni Treningu, 3 Dni Egzaminu)
    for i in range(1, 6): generate_national_blue_shield(f'blue_shield_train_day_{i}.csv', day_offset=i)
    for i in range(1, 4): generate_national_blue_shield(f'blue_shield_test_day_{i}.csv', day_offset=i + 5)
    print("🏆 Baza danych wygenerowana! Czas na trening SI.")