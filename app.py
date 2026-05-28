import streamlit as st

# 1. SET KONFIGURASI HALAMAN GLOBAL (Harus dipanggil pertama kali)
st.set_page_config(
    page_title="Shalehuddin Zaki - Portfolio App",
    page_icon="🚀",
    layout="wide"
)

# 2. INISIALISASI HALAMAN-HALAMAN MODULAR
about_me_page = st.Page("pages/1_about_me.py", title="Tentang Saya", icon="👤")
my_projects_page = st.Page("pages/2_my_projects.py", title="Proyek Saya", icon="📊")
visualization_page = st.Page("pages/3_data_visualization.py", title="Visualisasi Data & Model", icon="📈")
predict_page = st.Page("pages/4_predict.py", title="Prediksi Harga Rumah", icon="🏠")

# 3. MENGONFIGURASI NAVIGASI SIDEBAR
pg = st.navigation([about_me_page, my_projects_page, visualization_page, predict_page])

# 4. TAMPILAN UI UTAMA (LANDING PAGE BRANDING)
# Menggunakan struktur kolom untuk memberikan ruang kosong atau tata letak hero banner yang seimbang
main_col, side_space = st.columns([3, 1])

with main_col:
    st.title("🚀 Data Science & MLOps Portfolio")
    st.subheader("Selamat Datang di Portofolio Interaktif Saya")
    
    st.markdown(
        """
        Aplikasi web interaktif ini dirancang khusus untuk mendemonstrasikan keahlian 
        **End-to-End Machine Learning Engineering (MLOps)**. Di sini, saya mengintegrasikan analisis data, 
        eksplorasi fitur, visualisasi evaluasi performa model, hingga deployment *pipeline* prediksi real-time 
        menggunakan dataset komersial terkemuka: **Kaggle House Prices: Advanced Regression Techniques**.
        """
    )

st.markdown("---")

# 5. MENAMPILKAN RINGKASAN PROYEK/FITUR UTAMA (MEMPERCANTIK UI)
st.markdown("### 🛠️ Fitur Utama Aplikasi Ini")

# Membuat 4 kartu ringkasan visual untuk merepresentasikan setiap halaman
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.info("### 👤\n**Tentang Saya**\n\nLatar belakang akademik, kompetensi teknis utama, dan tautan jejaring profesional.")

with c2:
    st.info("### 📊\n**Proyek Saya**\n\nKumpulan pameran riwayat pengerjaan proyek data analitik dan rekayasa fitur pilihan.")

with c3:
    st.info("### 📈\n**Visualisasi**\n\nAnalisis grafik eksploratif dataset asli (EDA) serta pemantauan kurva metrik performa model.")

with c4:
    st.info("### 🏠\n**Prediksi**\n\nSimulator inferensi model riil untuk memproses file CSV eksternal dan mengunduh hasil estimasi harga.")

st.markdown("---")

# 6. BANNER PETUNJUK NAVIGASI UNTUK PENGGUNA
st.caption("💡 **Petunjuk Penggunaan:** Silakan gunakan menu **Sidebar Navigasi** di sebelah kiri untuk menjelajahi fungsionalitas dan detail dari setiap modul aplikasi.")

# 7. MENJALANKAN HALAMAN YANG DIPILIH USER DARI SIDEBAR
pg.run()