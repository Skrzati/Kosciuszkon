import os

# 1. ŚCISŁA KONFIGURACJA ŚRODOWISKA (Musi być na samej górze!)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler

# Konfiguracja strony
st.set_page_config(page_title="Błękitna Tarcza - Dashboard", layout="wide")
st.title("🛡️ Błękitna Tarcza - System Detekcji Spoofingu GPS")

# Definicja cech
features = [
    'N_pseudorange', 'N_snr', 'N_doppler',
    'S_pseudorange', 'S_snr', 'S_doppler',
    'E_pseudorange', 'E_snr', 'E_doppler',
    'W_pseudorange', 'W_snr', 'W_doppler',
    'kp_index', 'temperature', 'pressure', 'humidity'
]


# 2. FUNKCJE ŁADUJĄCE (Z LENIWYM IMPORTEM TENSORFLOW)
@st.cache_resource
def load_model():
    # Importujemy TF tylko tutaj, aby uniknąć blokady mutexa przy starcie aplikacji
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')  # Twarde wyłączenie GPU
    except:
        pass
    return tf.keras.models.load_model('blekitna_tarcza_meteo_model.keras')


@st.cache_data
def load_data_and_scaler():
    df = pd.read_csv('blue_shield_meteo_data.csv')
    scaler = StandardScaler()
    scaler.fit(df[features])
    return df, scaler


# Inicjalizacja danych
with st.spinner("Inicjalizacja systemu... Proszę czekać (ładowanie silnika AI może zająć chwilę)"):
    df, scaler = load_data_and_scaler()
    # Model ładujemy dopiero, gdy dane są gotowe
    model = load_model()

# 3. INTERFEJS UŻYTKOWNIKA
time_step = st.slider("Oś czasu (Sekunda symulacji)", 10, len(df) - 1, 10000)

if time_step >= 10:
    # Wycinanie okna danych
    current_window = df.iloc[time_step - 10:time_step]
    stan_rzeczywisty = df.iloc[time_step]['label']

    # --- PREDYKCJA AI ---
    # Skalowanie i przygotowanie kształtu (1, 10, 16)
    scaled_window = scaler.transform(current_window[features])
    model_input = np.expand_dims(scaled_window, axis=0)

    # Wykonanie predykcji
    prediction_probs = model.predict(model_input, verbose=0)
    ai_decision = np.argmax(prediction_probs)
    ai_certainty = np.max(prediction_probs) * 100

    # --- WIZUALIZACJA ---
    st.subheader("Bieżąca Telemetria Meteo i Sygnału")

    m1, m2, m3 = st.columns(3)
    m1.metric("Temperatura", f"{df.iloc[time_step]['temperature']:.1f} °C")
    m2.metric("Ciśnienie", f"{df.iloc[time_step]['pressure']:.1f} hPa")
    m3.metric("Wilgotność", f"{df.iloc[time_step]['humidity']:.1f} %")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig_snr = px.line(current_window, y=['N_snr', 'S_snr', 'E_snr', 'W_snr'], title="Moc Sygnału (SNR)")
        st.plotly_chart(fig_snr, use_container_width=True)
    with c2:
        fig_dop = px.line(current_window, y=['N_doppler', 'S_doppler', 'E_doppler', 'W_doppler'],
                          title="Przesunięcie Dopplera")
        st.plotly_chart(fig_dop, use_container_width=True)

    # Status i Werdykt
    status_map = {
        0: ("✅ SYGNAŁ CZYSTY", "green"),
        1: ("🚨 WYKRYTO ATAK NAZIEMNY", "red"),
        2: ("⚠️ WYKRYTO ATAK POWIETRZNY", "orange")
    }

    ai_tekst, ai_kolor = status_map[ai_decision]
    real_tekst, _ = status_map[stan_rzeczywisty]

    st.divider()
    st.markdown(f"<h1 style='text-align: center; color: {ai_kolor};'>WERDYKT AI: {ai_tekst}</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: gray;'>Pewność sieci neuronowej: {ai_certainty:.2f}%</h4>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #555;'>Prawda obiektywna: {real_tekst}</p>",
                unsafe_allow_html=True)

else:
    st.info("Przesuń suwak, aby rozpocząć analizę.")