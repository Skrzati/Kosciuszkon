import sys
import os
import glob
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

print("\n🚀 KRAJOWY SYSTEM BŁĘKITNA TARCZA - INICJALIZACJA")

miasta = ['WAW', 'GDN', 'KRK', 'WRO', 'POZ', 'BIA', 'SZC', 'LUB', 'RZE']
features = []
for m in miasta:
    features.extend([f'{m}_pseudorange', f'{m}_snr', f'{m}_doppler'])
features.extend(['kp_index', 'temperature', 'pressure', 'humidity', 'snr_std', 'doppler_std'])
target = 'label'
window_size = 60
step = 5


def load_and_engineer_features(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        print(f"❌ BŁĄD: Brak plików '{file_pattern}'. Uruchom najpierw generator!")
        sys.exit()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # KRAJOWA INŻYNIERIA CECH - Odchylenie na obszarze całej Polski!
        snr_cols = [f'{m}_snr' for m in miasta]
        doppler_cols = [f'{m}_doppler' for m in miasta]
        df['snr_std'] = df[snr_cols].std(axis=1)
        df['doppler_std'] = df[doppler_cols].std(axis=1)
        dfs.append(df)
    return dfs


print("1. Pobieranie danych z radarów...")
train_dfs = load_and_engineer_features('blue_shield_train_day_*.csv')
test_dfs = load_and_engineer_features('blue_shield_test_day_*.csv')

print("2. Normalizacja środowiska...")
scaler = StandardScaler()
combined_train = pd.concat(train_dfs, ignore_index=True)
scaler.fit(combined_train[features])
del combined_train
gc.collect()

print("3. Analiza topologiczna okien czasowych...")


def create_windows_from_dfs(dfs):
    X, y = [], []
    for df in dfs:
        df[features] = scaler.transform(df[features])
        data_f = df[features].values
        data_t = df[target].values
        for i in range(0, len(df) - window_size, step):
            X.append(data_f[i:(i + window_size)])
            y.append(data_t[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


X_train, y_train = create_windows_from_dfs(train_dfs)
X_test, y_test = create_windows_from_dfs(test_dfs)
print(f" -> Pakiety Treningowe: {len(X_train)} | Pakiety Bojowe (Test): {len(X_test)}")
del train_dfs, test_dfs;
gc.collect()

print("\n4. Uzbrajanie Głębokiej Sieci Neuronowej (LSTM 128)...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, len(features))),
    tf.keras.layers.LSTM(128, return_sequences=False),  # Wzmocnione serce dla 9 miast
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- TRENING BOJOWY ROZPOCZĘTY ---")
wagi_klas = {0: 1.0, 1: 8.0, 2: 5.0, 3: 8.0}

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,  # Większy batch dla stabilności pamięci
    validation_data=(X_test, y_test),
    class_weight=wagi_klas,
    verbose=1
)

# EWALUACJA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja (Egzamin)')
plt.title('Skuteczność Ochrony GNSS')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Strata Algorytmiczna')
plt.legend()
plt.show()

print("\n--- RAPORT Z LINII FRONTU ---")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
etykiety = ['Normalny (0)', 'Jammer Naziemny (1)', 'Dron Spoofujący (2)', 'Atak Replay (3)']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etykiety, yticklabels=etykiety)
plt.xlabel('Identyfikacja AI')
plt.ylabel('Prawdziwe Zagrożenie')
plt.title('Macierz Pomyłek - Narodowy System Wczesnego Ostrzegania')
plt.show()

print("\n--- SZCZEGÓŁOWA ANALIZA TAKTYCZNA ---")
print(classification_report(y_test, y_pred, target_names=['Normalny', 'Jammer', 'Dron', 'Replay']))

model.save('krajowa_blekitna_tarcza.keras')
print("\n✅ KRAJOWA BŁĘKITNA TARCZA JEST ONLINE. Model zapisany.")