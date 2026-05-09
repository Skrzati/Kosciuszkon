import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. Wczytanie i przygotowanie danych
print("1. Wczytywanie całodobowych danych uwzględniających zjawiska meteo...")
# ZMIANA: Czytamy nowy, wielki plik
df = pd.read_csv('blue_shield_meteo_data.csv')
print(f"Załadowano {len(df)} próbek z pliku!")

# ZMIANA: Definiujemy rozszerzone cechy (Dodana troposfera)
features = [
    'N_pseudorange', 'N_snr', 'N_doppler',
    'S_pseudorange', 'S_snr', 'S_doppler',
    'E_pseudorange', 'E_snr', 'E_doppler',
    'W_pseudorange', 'W_snr', 'W_doppler',
    'kp_index',
    'temperature', 'pressure', 'humidity' # <--- Nowe czujniki meteorologiczne
]
target = 'label'

# Skalowanie danych
print("2. Skalowanie danych (normalizacja matematyki orbitalnej)...")
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 2. Tworzenie "okien" (sliding window) - LSTM patrzy 10 sekund wstecz
print("3. Tworzenie okien czasowych...")
def create_windows(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:(i + window_size)][features].values)
        y.append(data.iloc[i + window_size][target])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_windows(df, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 3. ZAKTUALIZOWANA BUDOWA MODELU LSTM (METEO)
# =====================================================================
print("4. Budowa modelu Błękitnej Tarczy v2.0 (Sieć Rekurencyjna)...")
model = tf.keras.Sequential([
    # Warstwa wejściowa automatycznie dostosuje się do 16 wymiarów
    tf.keras.layers.Input(shape=(window_size, len(features))),

    # Warstwa ukryta LSTM
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.2),  # Zapobieganie overfittingowi
    tf.keras.layers.Dense(32, activation='relu'),

    # 3 klasy wyjściowe i funkcja softmax
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Trening
print("\n--- ROZPOCZYNAMY TRENING AI ---")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64, # ZMIANA: Zwiększony batch_size przyspieszy uczenie 86 tys. próbek
    validation_data=(X_test, y_test),
    verbose=1
)

# 5. Wykresy wyników treningu
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu (Z danych meteo)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Strata modelu')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 6. ANALIZA BŁĘDÓW I ZAPIS MODELU
# ==========================================
print("\n--- GENEROWANIE RAPORTU SKUTECZNOŚCI ---")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Konwersja do klas 0, 1, 2

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normalny (0)', 'Naziemny (1)', 'Powietrzny (2)'],
            yticklabels=['Normalny (0)', 'Naziemny (1)', 'Powietrzny (2)'])
plt.xlabel('Przewidziane przez AI')
plt.ylabel('Rzeczywistość')
plt.title('Macierz Pomyłek - Błękitna Tarcza (Meteo)')
plt.show()

print("\n--- SZCZEGÓŁOWY RAPORT KLASYFIKACJI ---")
print(classification_report(y_test, y_pred, target_names=['Normalny', 'Atak Naziemny', 'Atak Powietrzny']))

# ZAPIS GOTOWEGO MODELU DO PLIKU
model_filename = 'blekitna_tarcza_meteo_model.keras'
model.save(model_filename)
print(f"\n✅ SUKCES! Zaawansowany model został zapisany jako '{model_filename}'")