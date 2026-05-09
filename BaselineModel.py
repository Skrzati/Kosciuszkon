import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("--- Trening Modelu Bazowego (Random Forest) ---")

# 1. Wczytanie danych
df = pd.read_csv('blue_shield_meteo_data.csv')

# Definicja cech
features = [
    'N_pseudorange', 'N_snr', 'N_doppler',
    'S_pseudorange', 'S_snr', 'S_doppler',
    'E_pseudorange', 'E_snr', 'E_doppler',
    'W_pseudorange', 'W_snr', 'W_doppler',
    'kp_index', 'temperature', 'pressure', 'humidity'
]
target = 'label'

# 2. Skalowanie
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target].values

# W odróżnieniu od LSTM, Random Forest nie dostaje historii (sliding window),
# tylko ocenia sygnał z danej sekundy - to klasyczne podejście Machine Learning.

# 3. Podział danych (80% trening, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Trening modelu referencyjnego (Baseline)
print("Trenowanie lasu losowego (może to zająć kilkanaście sekund)...")
baseline_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
baseline_model.fit(X_train, y_train)

# 5. Predykcje i Ewaluacja
y_pred = baseline_model.predict(X_test)

print("\n--- RAPORT KLASYFIKACJI (MODEL BAZOWY) ---")
print(classification_report(y_test, y_pred, target_names=['Normalny', 'Atak Naziemny', 'Atak Powietrzny']))

# 6. Macierz Pomyłek
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normalny (0)', 'Naziemny (1)', 'Powietrzny (2)'],
            yticklabels=['Normalny (0)', 'Naziemny (1)', 'Powietrzny (2)'])
plt.xlabel('Przewidziane przez Random Forest')
plt.ylabel('Rzeczywistość')
plt.title('Macierz Pomyłek - Model Bazowy (Random Forest)')
plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png')
plt.show()

print("✅ Model bazowy wytrenowany. Wykres zapisano jako 'baseline_confusion_matrix.png'.")