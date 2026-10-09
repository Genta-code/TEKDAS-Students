# 🛒 E-Commerce Customer Behavior: Data -> BI -> ML -> LLM

Dashboard pembelajaran interaktif berbasis Streamlit menggunakan dataset transaksi pelanggan di folder `StudentDOA/E-Commerce_Customer_Behavior`.

Arsitektur aplikasi ini mengikuti pola yang sama dengan folder `simulation/`:
1. **Data**: Memahami karakteristik, struktur, dan validasi data pelanggan & pesanan (`customers.csv`, `orders.csv`, `product_summary.csv`, `monthly_revenue.csv`).
2. **Business Intelligence (BI)**: Menemukan pola deskriptif tren revenue bulanan, revenue per kategori produk, dan riwayat pesanan per pelanggan.
3. **Machine Learning (ML)**: Melatih pipeline Random Forest Classifier untuk memprediksi probabilitas customer churn.
4. **LLM Analyst**: Menggunakan AI / Ollama untuk menyusun rencana aksi retensi pelanggan berbasis data (*evidence-based*).

---

## 🚀 Cara Menjalankan

Buka terminal di folder `E-Commerce_Customer_Behavior`:

```bash
# Masuk ke folder proyek
cd "e:\TekDas\TEKDAS-Students\StudentDOA\E-Commerce_Customer_Behavior"

# (Opsional) Jalankan persiapan data & pelatihan model secara manual via CLI:
python data_prep.py
python ml_model.py

# Jalankan dashboard Streamlit:
streamlit run app.py
```

> **Catatan**: Jika model belum dilatih atau dihapus, Anda juga dapat menekan tombol **"Latih Model Random Forest Sekarang"** langsung dari tab **Machine Learning** di dalam dashboard Streamlit!
