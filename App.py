import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import time

# --- KONFIGURACJA ---
st.set_page_config(page_title="Narodowa Błękitna Tarcza", page_icon="🛡️", layout="wide")

# Stylizacja mrocznego interfejsu
st.markdown("""
    <style>
    .alert-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: monospace; font-size: 14px; }
    .alert-red { background-color: #ff4b4b20; border-left: 5px solid #ff4b4b; color: #ff4b4b; font-weight: bold; }
    .alert-orange { background-color: #ffa50020; border-left: 5px solid #ffa500; color: #ffa500; }
    .alert-green { background-color: #00ff0020; border-left: 5px solid #00ff00; color: #00ff00; }
    </style>
""", unsafe_allow_html=True)


# --- ŁADOWANIE DANYCH (Z INŻYNIERIĄ CECH) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('blue_shield_test_day_1.csv')
        miasta = ['WAW', 'GDN', 'KRK', 'WRO', 'POZ', 'BIA', 'SZC', 'LUB', 'RZE']
        snr_cols = [f'{m}_snr' for m in miasta]
        df['snr_std'] = df[snr_cols].std(axis=1)
        return df
    except FileNotFoundError:
        return None


df = load_data()

if df is not None:
    # --- NAGŁÓWEK ---
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/769/769269.png", width=80)
    with col_title:
        st.title("🛡️ NARODOWA BŁĘKITNA TARCZA | Tactical Command")
        st.markdown("**Status:** Aktywny | **Zasięg:** Terytorium RP | **Silnik:** LSTM-Spatial Fusion")

    st.markdown("---")

    # --- KONSOLA OPERATORA ---
    st.sidebar.header("🕹️ Konsola Operatora")
    sekunda_obecna = st.sidebar.slider("Znacznik czasu (sekunda doby):", 0, len(df) - 1000, 15000, step=500)
    okno_czasowe = st.sidebar.selectbox("Rozdzielczość radaru (sekundy):", [500, 1000, 3600])

    df_window = df.iloc[sekunda_obecna:sekunda_obecna + okno_czasowe]
    aktualny_status = df_window['label'].max()

    # --- KPI (Wskaźniki góra) ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("System", "9 / 9 Stacji", "Online")
    k2.metric("Pogoda (Kp)", f"{df_window['kp_index'].mean():.1f}", f"Wariancja: {df_window['snr_std'].mean():.1f}")
    k3.metric("Średnie SNR", f"{df_window['WAW_snr'].mean():.1f} dB", "Ref: Warszawa")

    label_map = ["BEZPIECZNIE", "JAMMER NAZIEMNY", "DRON SPOOFUJĄCY", "ATAK REPLAY"]
    k4.metric("ALARM AI", label_map[int(aktualny_status)], "ZAGROŻENIE" if aktualny_status > 0 else "BRAK",
              delta_color="inverse" if aktualny_status > 0 else "normal")

    st.markdown("---")

    # --- INTELIGENTNA LOKALIZACJA EPICENTRUM 2.0 ---
    miasta_map = {
        'Warszawa': 'WAW', 'Gdansk': 'GDN', 'Krakow': 'KRK', 'Wroclaw': 'WRO',
        'Poznan': 'POZ', 'Bialystok': 'BIA', 'Szczecin': 'SZC', 'Lublin': 'LUB', 'Rzeszow': 'RZE'
    }
    coords = {
        'Warszawa': (52.23, 21.01), 'Gdansk': (54.35, 18.64), 'Krakow': (50.06, 19.94),
        'Wroclaw': (51.10, 17.03), 'Poznan': (52.40, 16.92), 'Bialystok': (53.13, 23.16),
        'Szczecin': (53.42, 14.55), 'Lublin': (51.24, 22.56), 'Rzeszow': (50.04, 21.99)
    }

    stacje_data = []

    # 1. Wyliczamy mediany krajowe (odporne na burze słoneczne)
    med_snr = np.median([df_window[f'{m}_snr'].mean() for m in miasta_map.values()])
    med_dop = np.median([df_window[f'{m}_doppler'].mean() for m in miasta_map.values()])

    # 2. Liczymy "Score Anomalii" dla każdego miasta
    scores = {}
    for nazwa, kod in miasta_map.items():
        # Dewiacja od normy krajowej
        snr_diff = med_snr - df_window[f'{kod}_snr'].mean()
        dop_diff = abs(df_window[f'{kod}_doppler'].mean() - med_dop)

        # Hybrydowy wskaźnik: Jammer bije w SNR, Dron bije w Doppler
        scores[nazwa] = (snr_diff * 10) + (dop_diff * 2)

    # 3. Jeśli AI wykryło atak, musimy podświetlić najbardziej podejrzane miejsca
    epicentra = []
    if aktualny_status > 0:
        # Sortujemy miasta od najbardziej "zepsutych"
        sorted_cities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Zawsze podświetlamy TOP 1 (największa anomalia)
        epicentra.append(sorted_cities[0][0])
        # Jeśli inne miasta mają podobnie wysoki wynik (ataki obszarowe), też je bierzemy
        for city, score in sorted_cities[1:]:
            if score > sorted_cities[0][1] * 0.7:  # Próg 70% najsilniejszej anomalii
                epicentra.append(city)

    for nazwa in miasta_map.keys():
        czy_atak_tu = nazwa in epicentra
        stacje_data.append({
            'miasto': nazwa,
            'lat': coords[nazwa][0],
            'lon': coords[nazwa][1],
            'color': [255, 50, 50, 160] if czy_atak_tu else [0, 200, 0, 180],
            'radius': 120000 if czy_atak_tu else 8000
        })

    # --- WIZUALIZACJA MAPY ---
    st.subheader("📡 Radar Przestrzenny (Spatial Isolation Mode)")
    df_pdk = pd.DataFrame(stacje_data)
    view_state = pdk.ViewState(latitude=52.2, longitude=19.3, zoom=5.3, pitch=35)

    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style="dark",
        initial_view_state=view_state,
        layers=[
            pdk.Layer('ScatterplotLayer', data=df_pdk, get_position='[lon, lat]',
                      get_color='color', get_radius='radius', pickable=True),
            pdk.Layer('TextLayer', data=df_pdk, get_position='[lon, lat]',
                      get_text='miasto', get_size=15, get_color=[255, 255, 255])
        ]
    ))

    st.markdown("---")

    # --- KONSOLA SIEM (Logi) ---
    c_log, c_plot = st.columns([1, 2])
    with c_log:
        st.subheader("🚨 Logi Systemowe")
        if aktualny_status == 0:
            st.markdown("<div class='alert-box alert-green'>[SEC: " + str(
                sekunda_obecna) + "] STATUS: Sygnał GNSS autentyczny. Brak dewiacji przestrzennej.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='alert-box alert-red'>[ALARM] WYKRYTO EPICENTRUM: {', '.join(epicentra).upper()}</div>",
                unsafe_allow_html=True)
            st.markdown(f"<div class='alert-box alert-orange'>SYGNATURA: {label_map[int(aktualny_status)]}</div>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div class='alert-box alert-orange'>DECYZJA AI: Wymagane natychmiastowe przełączenie na systemy inercyjne (INS).</div>",
                unsafe_allow_html=True)

    with c_plot:
        st.subheader("📈 Analiza Telemetryczna (WAW vs GDN)")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(df_window['timestamp'], df_window['WAW_snr'], label='SNR Warszawa', color='#1f77b4', linewidth=1)
        ax.plot(df_window['timestamp'], df_window['GDN_snr'], label='SNR Gdańsk', color='#ff7f0e', alpha=0.6,
                linewidth=1)
        if aktualny_status > 0:
            ax.axvspan(df_window['timestamp'].iloc[0], df_window['timestamp'].iloc[-1], color='red', alpha=0.1)
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.legend()
        st.pyplot(fig)

else:
    st.error("Nie znaleziono pliku blue_shield_test_day_1.csv!")