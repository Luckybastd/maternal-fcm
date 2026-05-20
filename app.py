import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Risiko Kehamilan", page_icon="🤰", layout="centered")

# --- 2. LOAD MODEL & SCALER ---
@st.cache_resource
def load_models():
    with open('fcm_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model_data, scaler

model_data, scaler = load_models()
V = model_data['V']
w = model_data['w']
c = model_data['c']
mapping = model_data['mapping']

# --- 3. UI JUDUL WEB ---
st.title("🤰 Sistem Deteksi Dini Risiko Kehamilan")
st.markdown("""
Aplikasi ini mengimplementasikan algoritma **Fuzzy C-Means (FCM)** untuk memprediksi tingkat risiko kehamilan berdasarkan parameter fisiologis medis. 
Silakan masukkan data pasien pada formulir di bawah ini.
""")

# --- 4. FORMULIR INPUT USER ---
st.header("📝 Masukkan Data Pasien")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia (Age)", min_value=10, max_value=70, value=25)
    systolic = st.number_input("Tekanan Darah Sistolik (SystolicBP)", min_value=70, max_value=200, value=120)
    diastolic = st.number_input("Tekanan Darah Diastolik (DiastolicBP)", min_value=40, max_value=130, value=80)

with col2:
    bs = st.number_input("Kadar Gula Darah (BS)", min_value=4.0, max_value=30.0, value=7.0, step=0.1)
    body_temp = st.number_input("Suhu Tubuh (BodyTemp - Fahrenheit)", min_value=90.0, max_value=105.0, value=98.0, step=0.1)
    heart_rate = st.number_input("Detak Jantung (HeartRate)", min_value=50, max_value=150, value=70)

# --- 5. TOMBOL & PROSES PREDIKSI ---
st.write("---")
if st.button("🔍 Analisis Risiko Pasien", type="primary", use_container_width=True):
    
    # Bungkus input ke DataFrame agar sesuai dengan format model
    input_df = pd.DataFrame([[age, systolic, diastolic, bs, body_temp, heart_rate]],
                            columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
    
    # Normalisasi Input
    input_scaled = scaler.transform(input_df)
    
    # Replikasi Perhitungan Prediksi Fuzzy C-Means
    d = np.zeros((1, c))
    for k in range(c):
        d[:, k] = np.linalg.norm(input_scaled - V[k], axis=1)
    
    d_fixed = np.fmax(d, np.finfo(np.float64).eps) # Hindari pembagian dengan nol
    d_inv = d_fixed ** (-2 / (w - 1))
    U_new = d_inv / np.sum(d_inv, axis=1)[:, np.newaxis]
    
    # Ambil keputusan Klaster Maksimal
    klaster_prediksi = np.argmax(U_new, axis=1)[0] + 1
    label_prediksi = mapping.get(klaster_prediksi, "Tidak Diketahui")
    
    # --- 6. MENAMPILKAN HASIL ---
    st.subheader("📊 Hasil Analisis Medis")
    
    # Warna peringatan berbeda sesuai level risiko
    if label_prediksi == 'high risk':
        st.error(f"⚠️ Kategori Pasien: **TINGGI ({label_prediksi.upper()})** (Masuk ke Klaster {klaster_prediksi})")
    elif label_prediksi == 'mid risk':
        st.warning(f"🔔 Kategori Pasien: **SEDANG ({label_prediksi.upper()})** (Masuk ke Klaster {klaster_prediksi})")
    else:
        st.success(f"✅ Kategori Pasien: **RENDAH ({label_prediksi.upper()})** (Masuk ke Klaster {klaster_prediksi})")
    
    st.write("**Rincian Derajat Keanggotaan (Probabilitas Fuzzy):**")
    for k in range(c):
        nama_label = mapping.get(k+1, 'Unknown')
        persentase = U_new[0][k] * 100
        st.write(f"- Klaster {k+1} ({nama_label.upper()}): **{persentase:.2f}%**")