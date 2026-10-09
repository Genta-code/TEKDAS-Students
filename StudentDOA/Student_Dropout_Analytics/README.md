# 🎓 Student Dropout Analytics: Data -> BI -> ML -> LLM

Dashboard pembelajaran interaktif berbasis Streamlit menggunakan dataset mahasiswa UCI (**Predict Students' Dropout and Academic Success**) di folder `StudentDOA`.

Arsitektur aplikasi ini mengikuti pola yang sama dengan folder `simulation/`:
1. **Data**: Memahami karakteristik, struktur, dan validasi data mahasiswa.
2. **Business Intelligence (BI)**: Menemukan pola deskriptif (pengaruh SPP, beasiswa, dan capaian SKS semester 1 & 2 terhadap kelulusan).
3. **Machine Learning (ML)**: Melatih pipeline Random Forest Classifier untuk memprediksi probabilitas risiko putus studi (*dropout*) mahasiswa.
4. **LLM Academic Advisor**: Menggunakan AI / Ollama untuk menghasilkan rencana aksi intervensi taktis bagi dosen wali dan bagian kemahasiswaan berbasis data (*evidence-based*).

---

## 🚀 Cara Menjalankan

Buka terminal di folder proyek (`TEKDAS-Students`) dan aktifkan virtual environment jika ada:

```bash
# Masuk ke folder StudentDOA
cd StudentDOA

# (Opsional) Persiapkan data dan latih model
python data_prep.py
python ml_model.py

# Jalankan dashboard Streamlit
streamlit run app.py
```

> **Catatan**: Jika model belum dilatih sebelumnya, Anda juga dapat menekan tombol **"Latih Model Random Forest Sekarang"** langsung dari tab **Machine Learning** di dalam dashboard Streamlit!
