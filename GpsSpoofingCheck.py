import sys
import os

# 1. Wyłączenie logów C++ i wymuszenie CPU na poziomie systemu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['ABSL_FLAGS_logging_verbosity'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 2. OSTATECZNE ODCIĘCIE GPU DLA MACA (tensorflow-metal bypass)
try:
    # Zmuszamy TensorFlow, aby "nie widział" żadnego urządzenia GPU, w tym układów Apple M1/M2/M3
    tf.config.set_visible_devices([], 'GPU')
    print("Wymuszono tryb czystego CPU. Wtyczka Apple Metal wyłączona.")
except Exception as e:
    print(f"Uwaga przy konfiguracji urządzeń: {e}")

# ==========================================
# 1. Wczytanie i przygotowanie danych
print("\n1. Wczytywanie zredukowanych danych uwzględniających zjawiska meteo...")
df = pd.read_csv('blue_shield_meteo_data.csv')
print(f"Załadowano {len(df)} próbek z pliku!")

features = [
    'N_pseudorange', 'N_snr', 'N_doppler',
    'S_pseudorange', 'S_snr', 'S_doppler',
    'E_pseudorange', 'E_snr', 'E_doppler',
    'W_pseudorange', 'W_snr', 'W_doppler',
    'kp_index',
    'temperature', 'pressure', 'humidity'
]
target = 'label'

# Skalowanie danych
print("2. Skalowanie danych (normalizacja matematyki orbitalnej)...")
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 2. Tworzenie "okien" (sliding window) - ZOPTYMALIZOWANE!
print("3. Tworzenie okien czasowych (szybka metoda NumPy)...")

def create_windows(data, window_size=10):
    # KRYTYCZNA POPRAWKA: Konwersja do NumPy PRZED pętlą.
    data_features = data[features].values
    data_target = data[target].values

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data_features[i:(i + window_size)])
        y.append(data_target[i + window_size])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

window_size = 10
X, y = create_windows(df, window_size)

# Zwalnianie RAM-u po oryginalnym DataFrame (bardzo ważne przy dużych plikach!)
import gc
del df
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 3. ZAKTUALIZOWANA BUDOWA MODELU LSTM (METEO)
# =====================================================================
print("4. Budowa modelu Błękitnej Tarczy v2.0 (Sieć Rekurencyjna)...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, len(features))),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Trening
print("\n--- ROZPOCZYNAMY TRENING AI ---")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
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
y_pred = np.argmax(y_pred_probs, axis=1)

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

# ZMIANA 2: Delikatnie inna nazwa modelu wyjściowego
model_filename = 'blekitna_tarcza_meteo_model.keras'
model.save(model_filename)
print(f"\n✅ SUKCES! Zaawansowany model został zapisany jako '{model_filename}'")