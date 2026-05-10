import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Rozpoczęcie Eksploracyjnej Analizy Danych (EDA) ---")

# 1. Wczytanie danych (Używamy zbioru testowego z Dnia 2)
plik_danych = 'blue_shield_test.csv'
try:
    df = pd.read_csv(plik_danych)
    print(f"Załadowano zbiór danych '{plik_danych}': {df.shape[0]} wierszy, {df.shape[1]} kolumn.\n")
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku '{plik_danych}'. Najpierw uruchom DataGenerator.py!")
    exit()

# 2. Sprawdzenie jakości danych (Missing values)
print("--- Kontrola Jakości Danych ---")
missing_values = df.isnull().sum().sum()
if missing_values == 0:
    print("✅ Brak pustych wartości (NaN) - zbiór jest kompletny.")
else:
    print(f"❌ Znaleziono {missing_values} pustych wartości!")

# 3. Podstawowe statystyki (Mean, Min, Max)
print("\n--- Podstawowe Statystyki Wybranych Cech ---")
# Wybieramy kilka kluczowych kolumn, by nie zalać ekranu tekstem
cols_to_show = ['N_snr', 'N_pseudorange', 'temperature', 'kp_index']
print(df[cols_to_show].describe().loc[['mean', 'min', 'max', 'std']])

# 4. Wizualizacje

# A. Rozkład Klas (Normalny vs Anomalie)
plt.figure(figsize=(8, 5))
# Dodano hue='label' i legend=False, aby naprawić błąd FutureWarning z biblioteki Seaborn
sns.countplot(data=df, x='label', hue='label', palette='Set2', legend=False)
plt.title('Dystrybucja klas w zbiorze danych (Dzień Testowy)')
plt.xlabel('Klasa (0: Normalny, 1: Atak Naziemny, 2: Atak Powietrzny)')
plt.ylabel('Liczba próbek')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('eda_class_distribution.png')
plt.show()

# B. Szereg Czasowy (Time-Series) - Pokazanie ataku na wykresie
# DYNAMICZNE SZUKANIE ATAKU (bo ataki są teraz w losowych momentach!)
atak_indeksy = df[df['label'] > 0].index

if not atak_indeksy.empty:
    pierwszy_atak = atak_indeksy[0]
    # Wybieramy okno: 3000 sekund przed atakiem i 8000 po nim, żeby ładnie to uchwycić
    start_idx = max(0, pierwszy_atak - 3000)
    end_idx = min(len(df), pierwszy_atak + 8000)
    print(f"\nZnaleziono atak w sekundzie {pierwszy_atak}. Rysowanie wykresu od {start_idx} do {end_idx}...")
else:
    print("\nUwaga: W tym zbiorze nie wylosowano żadnych ataków! Rysuję domyślny fragment.")
    start_idx = 10000
    end_idx = 20000

plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'][start_idx:end_idx], df['N_snr'][start_idx:end_idx], label='SNR Anteny N', color='blue', alpha=0.6)

# Zaznaczenie obszarów ataku (gdzie label > 0)
attack_indices = df.index[(df['label'] > 0) & (df.index >= start_idx) & (df.index < end_idx)]
if not attack_indices.empty:
    for idx in attack_indices:
        plt.axvline(x=df['timestamp'][idx], color='red', alpha=0.05)

plt.title('Szereg Czasowy: Stosunek Sygnału do Szumu (SNR) w czasie z widocznymi atakami (Ramp Attack)')
plt.xlabel('Czas (sekundy)')
plt.ylabel('Wartość SNR')
plt.legend()
plt.tight_layout()
plt.savefig('eda_time_series.png')
plt.show()

print("\n✅ Analiza zakończona. Wygenerowano i zapisano wykresy: 'eda_class_distribution.png' oraz 'eda_time_series.png'.")