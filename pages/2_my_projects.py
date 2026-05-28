import streamlit as st

st.header("📊 Daftar Proyek Saya")
st.write("Berikut adalah beberapa proyek unggulan berbasis Artificial Intelligence dan Data Science yang telah saya kembangkan:")

# Menggunakan kolom untuk membagi tampilan proyek agar responsif dan rapi
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Proyek 1: Recommendation System")
    st.markdown("""
    * **Algoritma/Model:** *Collaborative Filtering (SVD)* & *Cosine Similarity*
    * **Deskripsi:** Sistem rekomendasi pintar yang dirancang untuk memberikan saran produk atau konten yang dipersonalisasi berdasarkan riwayat preferensi dan kesamaan perilaku antar-pengguna.
    """)
    # Integrasi tautan langsung ke Google Drive
    st.link_button("📂 Buka Folder Proyek", "https://drive.google.com/drive/folders/12oGoJ8xCk_9D4daiJQ83PrujPnzjx7zO?usp=sharing")

with col2:
    st.subheader("Proyek 2: CNN Learning Mini Project")
    st.markdown("""
    * **Algoritma/Model:** *Convolutional Neural Network (CNN)*
    * **Deskripsi:** Implementasi arsitektur *Deep Learning* (CNN) untuk klasifikasi citra/gambar otomatis, mencakup tahapan augmentasi data, ekstraksi fitur spasial, dan evaluasi *loss-accuracy curve*.
    """)
    # Integrasi tautan langsung ke Google Colab
    st.link_button("💻 Buka Notebook Colab", "https://colab.research.google.com/drive/1Dak4nZqH7zj07x0r3ws6nkpAwNqmmECL?usp=sharing")

with col3:
    st.subheader("Proyek 3: Case Study Feature Importance Analysis & Model Interpretation")
    st.markdown("""
    * **Algoritma/Model:** *XGBoost Regressor* & *SHAP (SHapley Additive exPlanations)*
    * **Deskripsi:** Proyek khusus untuk membedah keputusan model kompleks (*Black-Box Model*) menggunakan SHAP values guna memahami kontribusi serta tingkat signifikansi setiap fitur secara transparan.
    """)
    # Integrasi tautan langsung ke Google Colab
    st.link_button("💻 Buka Notebook Colab", "https://colab.research.google.com/drive/1_mK0hOslWBFUgPPtP8S-NTBm0jp3ze08?usp=sharing")