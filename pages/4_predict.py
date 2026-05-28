import streamlit as st
import pandas as pd
import pickle
import os

st.header("🏠 Pipeline Prediksi Harga Rumah")
st.write("Unggah berkas data properti Anda untuk mendapatkan estimasi nilai jual secara instan.")

# 1. Pilihan interaktif model untuk melakukan inferensi prediksi
selected_model_name = st.selectbox(
    "Pilih Arsitektur Model Pemodelan:",
    ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]
)

# Struktur pemetaan pilihan menu ke file biner .pkl yang sebenarnya di folder models/
model_files = {
    "Linear Regression": "models/linreg_model.pkl",
    "Random Forest Regressor": "models/rf_model.pkl",
    "XGBoost Regressor": "models/xgb_model.pkl"
}

# 2. Widget Pengunggahan File CSV Data Uji (seperti test.csv)
uploaded_file = st.file_uploader("Pilih file CSV data rumah baru:", type=["csv"])

if uploaded_file is not None:
    # Membaca file CSV yang diunggah
    df_input = pd.read_csv(uploaded_file)
    st.write("📌 **Pratinjau Data yang Diunggah:**")
    st.dataframe(df_input.head())
    
    # 3. Tombol untuk memicu jalannya pipeline inferensi
    trigger_prediction = st.button("🚀 Jalankan Pipeline Prediksi")
    
    if trigger_prediction:
        # Menampilkan animasi loading interaktif
        with st.spinner(f"Model {selected_model_name} sedang memproses prediksi data..."):
            try:
                # 3a. Memuat file model .pkl yang dipilih secara dinamis
                model_path = model_files[selected_model_name]
                
                with open(model_path, "rb") as file:
                    loaded_model = pickle.load(file)
                
                # 3b. Menyelaraskan kolom input dengan 5 fitur utama saat proses pelatihan model
                features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "1stFlrSF"]
                
                # Memastikan kolom yang dibutuhkan ada di file yang diunggah
                missing_cols = [col for col in features if col not in df_input.columns]
                if missing_cols:
                    st.error(f"Gagal memproses! File CSV kekurangan kolom fitur berikut: {missing_cols}")
                else:
                    # Filter data dan tangani data kosong (NaN) dengan imputasi nilai 0
                    X_input = df_input[features].fillna(0)
                    
                    # 3c. Menjalankan proses prediksi sesungguhnya berbasis model terpilih
                    predictions = loaded_model.predict(X_input)
                    
                    st.success(f"Pipeline prediksi menggunakan {selected_model_name} berhasil dieksekusi!")
                    
                    # 3d. Menggabungkan hasil prediksi ke dataframe hasil akhir
                    df_result = df_input.copy()
                    df_result['Predicted_SalePrice'] = predictions
                    
                    # 4. Menampilkan hasil keluaran prediksi kepada pengguna (menampilkan Id jika ada)
                    st.write("### 🔑 Hasil Prediksi Model:")
                    display_cols = ['Id', 'Predicted_SalePrice'] if 'Id' in df_result.columns else ['Predicted_SalePrice']
                    st.dataframe(df_result[display_cols].head(10))
                    
                    # 5. Opsi pengunduhan berkas hasil prediksi utuh (.CSV)
                    csv_download = df_result.to_csv(index=False).encode('utf-8')
                    file_output_name = f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv"
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi Lengkap (.CSV)",
                        data=csv_download,
                        file_name=file_output_name,
                        mime="text/csv"
                    )
                
            except FileNotFoundError:
                st.error(f"Berkas model `{model_path}` tidak ditemukan di folder `models/`. Silakan jalankan `train_model.py` terlebih dahulu di VS Code.")
            except Exception as e:
                st.error(f"Terjadi kesalahan sistem: {e}")

else:
    st.info("Silakan unggah berkas CSV pengujian komersial (seperti berkas `test.csv`) untuk memulai alur inferensi.")