# 🎮 VCT 2025 Esports Intelligence: Data -> BI -> ML -> LLM

Dashboard pembelajaran interaktif berbasis Streamlit menggunakan dataset turnamen profesional **Valorant Champions Tour (VCT) 2025** di folder `StudentDOA/VCT`.

Arsitektur aplikasi ini mengikuti pola yang sama dengan folder `simulation/`:
1. **Data**: Memahami karakteristik, struktur, dan validasi data performa pemain dan histori match (`players_stats.csv`, `scores.csv`, `maps_scores.csv`, `agents_pick_rates.csv`).
2. **Esports Intelligence (BI)**: Menemukan pola deskriptif tim dengan kemenangan terbanyak, leaderboard ACS & Rating pemain, serta histori match per pemain.
3. **Machine Learning (ML)**: Melatih pipeline Random Forest Classifier untuk memprediksi probabilitas performa *High-Impact* (Rating ≥ 1.05) bagi pro player.
4. **LLM Esports Coach & Analyst**: Menggunakan AI / Ollama untuk menyusun rekomendasi latihan dan arahan taktis berbasis data (*evidence-based*).

---

## 🚀 Cara Menjalankan

Buka terminal di folder `VCT`:

```bash
# Masuk ke folder proyek
cd "e:\TekDas\TEKDAS-Students\StudentDOA\VCT"

# (Opsional) Persiapan data dan latih model via CLI:
python data_prep.py
python ml_model.py

# Jalankan dashboard Streamlit:
streamlit run app.py
```

> **Catatan**: Jika model belum dilatih atau di-reset, Anda juga dapat menekan tombol **"Latih Model Random Forest Sekarang"** langsung dari tab **Machine Learning** di dalam dashboard Streamlit!
