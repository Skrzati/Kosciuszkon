import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from gnss_simulator import RealisticGNSSSimulator
from gnss_pipeline import SpatiotemporalPipeline
from gnss_defender import build_transformer_defender, get_callbacks
from sklearn.utils.class_weight import compute_class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_production_pipeline():
    print("🛡️ Inicjalizacja systemu Błękitna Tarcza v3.0 (Production-Grade)")

    # 1. Generowanie realistycznych danych (Seamless Takeover + Randomization)
    data_file = 'blue_shield_synthetic.csv'
    sim = RealisticGNSSSimulator(samples=86400)
    sim.generate(filename=data_file)
    df = pd.read_csv(data_file)

    # 2. Pipeline i ekstrakcja (Double Differences)
    pipeline = SpatiotemporalPipeline(window_size=60, batch_size=128)
    df_features = pipeline.extract_features(df)

    features = df_features.drop('label', axis=1).values
    labels = df_features['label'].values

    # 3. Restrykcyjny Chronologiczny Podział (Train: 70%, Test: 30%)
    split_idx = int(len(features) * 0.70)
    X_train, y_train = features[:split_idx], labels[:split_idx]
    X_test, y_test = features[split_idx:], labels[split_idx:]

    print(f"📊 Rozkład klas w treningu: {np.bincount(y_train.astype(int))}")
    print(f"📊 Rozkład klas w testach: {np.bincount(y_test.astype(int))}")

    # Ręczna stabilizacja (Soft Balancing) zapobiegająca eksplozji False Positives
    class_weight_dict = {
        0: 1.0,  # Normalny lot - traktujemy jako twardą bazę (1.0)
        1: 3.5,  # Atak naziemny (kara 3.5x większa za przeoczenie)
        2: 3.5  # Atak powietrzny (kara 3.5x większa za przeoczenie)
    }
    print(f"⚖️ Zaimplementowane zoptymalizowane wagi klas: {class_weight_dict}")

    train_ds = pipeline.create_walk_forward_dataset(X_train, y_train, is_training=True)
    test_ds = pipeline.create_walk_forward_dataset(X_test, y_test, is_training=False)

    # 4. Architektura i trening z zabezpieczeniami
    model = build_transformer_defender(window_size=60, num_features=X_train.shape[1])

    print("\n🚀 Rozpoczęcie Treningu (Zabezpieczone przez EarlyStopping i Focal Loss)...")
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=15,  # Większa liczba, ale EarlyStopping go przerwie przy przeuczeniu
        class_weight=class_weight_dict,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 5. Ewaluacja (Benchmarki)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    print("\n--- 📈 RAPORT EWALUACJI PRODUKCYJNEJ ---")
    # DODANO: labels=[0, 1, 2] aby wymusić raport dla 3 klas
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=['Normalny', 'Atak Naziemny', 'Atak Powietrzny'],
        zero_division=0
    ))

if __name__ == "__main__":
    run_production_pipeline()