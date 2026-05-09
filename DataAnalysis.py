import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Rozpoczęcie Eksploracyjnej Analizy Danych (EDA) ---")

# 1. Wczytanie danych
df = pd.read_csv('blue_shield_meteo_data.csv')
print(f"Załadowano zbiór danych: {df.shape[0]} wierszy, {df.shape[1]} kolumn.\n")

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
sns.countplot(data=df, x='label', palette='Set2')
plt.title('Dystrybucja klas w zbiorze danych')
plt.xlabel('Klasa (0: Normalny, 1: Atak Naziemny, 2: Atak Powietrzny)')
plt.ylabel('Liczba próbek')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('eda_class_distribution.png')
plt.show()

# B. Szereg Czasowy (Time-Series) - Pokazanie ataku na wykresie
# Wybieramy wycinek czasu (np. od sekundy 10000 do 20000), żeby złapać moment ataku
start_idx = 10000
end_idx = 20000

plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'][start_idx:end_idx], df['N_snr'][start_idx:end_idx], label='SNR Anteny N', color='blue', alpha=0.6)

# Zaznaczenie obszarów ataku (gdzie label > 0)
attack_indices = df.index[(df['label'] > 0) & (df.index >= start_idx) & (df.index < end_idx)]
if not attack_indices.empty:
    for idx in attack_indices:
        plt.axvline(x=df['timestamp'][idx], color='red', alpha=0.05)

plt.title('Szereg Czasowy: Stosunek Sygnału do Szumu (SNR) w czasie z widocznymi atakami')
plt.xlabel('Czas (sekundy)')
plt.ylabel('Wartość SNR')
plt.legend()
plt.tight_layout()
plt.savefig('eda_time_series.png')
plt.show()

print("\n✅ Analiza zakończona. Wygenerowano i zapisano wykresy: 'eda_class_distribution.png' oraz 'eda_time_series.png'.")