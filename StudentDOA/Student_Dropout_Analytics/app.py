"""Streamlit Academic Success Dashboard: Student Dropout Analytics (Data -> BI -> ML -> LLM).

Menjalankan dashboard:
    streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from data_prep import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    RAW_TARGET,
    TARGET,
    engineer_features,
    load_raw_dataset,
    prepare_student_features,
)
from advisor import DEFAULT_MODEL, build_prompt, generate_student_strategy

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Student Academic Intelligence: ML + LLM",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Success Intelligence — dari Data ke ML dan LLM")
st.caption(
    "Proyek pembelajaran: Eksplorasi data akademik mahasiswa, prediksi risiko Dropout dengan Machine Learning, "
    "dan gunakan LLM untuk menyusun rencana intervensi akademik berbasis bukti."
)


@st.cache_data
def load_data():
    raw_df = load_raw_dataset(BASE_DIR)
    processed_df = engineer_features(raw_df)
    return raw_df, processed_df


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "student_dropout_pipeline.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_optional_outputs():
    fi_path = BASE_DIR / "feature_importance.csv"
    metrics_path = BASE_DIR / "model_metrics.json"
    fi = pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return fi, metrics


raw_students, students = load_data()
model = load_model()
feature_importance, model_metrics = load_optional_outputs()

# Headline Academic Metrics
total_students = len(students)
dropout_count = int(students[TARGET].sum())
dropout_rate = students[TARGET].mean()
graduate_count = int((students[RAW_TARGET] == "Graduate").sum())
tuition_good_rate = (students["Tuition fees up to date"] == 1).mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Mahasiswa", f"{total_students:,}")
col2.metric("Tingkat Dropout (Observed)", f"{dropout_rate:.1%}", f"{dropout_count:,} mhs", delta_color="inverse")
col3.metric("Tingkat Kelulusan (Graduate)", f"{(graduate_count / total_students):.1%}", f"{graduate_count:,} mhs")
col4.metric("Kepatuhan SPP (Tuition Up-to-Date)", f"{tuition_good_rate:.1%}")

# Sidebar controls
st.sidebar.header("Pilih Mahasiswa")
filter_status = st.sidebar.selectbox(
    "Filter status di dropdown:",
    ["Semua", "Dropout", "Graduate", "Enrolled"],
)

if filter_status == "Semua":
    student_list = students[ID_COLUMN].tolist()
else:
    student_list = students.loc[students[RAW_TARGET] == filter_status, ID_COLUMN].tolist()

selected_id = st.sidebar.selectbox("Student ID", student_list)
decision_threshold = st.sidebar.slider("ML Decision Threshold (Risiko Dropout)", 0.10, 0.90, 0.50, 0.05)

# Ekstrak baris mahasiswa terpilih
student_row = students.loc[students[ID_COLUMN].eq(selected_id)].iloc[0]
X_student = pd.DataFrame([student_row])[FEATURE_COLUMNS]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Ringkasan Cepat:**")
st.sidebar.write(f"- **Umur saat Masuk:** {int(student_row['Age at enrollment'])} tahun")
st.sidebar.write(f"- **Status SPP:** {'Lunas ✅' if student_row['Tuition fees up to date'] == 1 else 'Menunggak ⚠️'}")
st.sidebar.write(f"- **Beasiswa:** {'Penerima Beasiswa' if student_row['Scholarship holder'] == 1 else 'Bukan'}")
st.sidebar.write(f"- **Lulus Matkul Sem 1:** {int(student_row['Curricular units 1st sem (approved)'])} unit")
st.sidebar.write(f"- **Lulus Matkul Sem 2:** {int(student_row['Curricular units 2nd sem (approved)'])} unit")

tab_data, tab_bi, tab_ml, tab_llm = st.tabs([
    "1 · Dataset Mahasiswa",
    "2 · Academic Intelligence (BI)",
    "3 · Machine Learning",
    "4 · LLM Academic Advisor",
])

# TAB 1: DATASET MAHASISWA
with tab_data:
    st.subheader("Data Pendidikan Mahasiswa (UCI Student Dataset)")
    st.markdown(
        "- **dataset.csv** — Berisi data demografis, latar belakang sosial-ekonomi, dan rekam jejak akademik semester 1 & 2.\n"
        "- **Target** — Status akhir mahasiswa: `Graduate` (Lulus), `Dropout` (Putus studi), atau `Enrolled` (Masih aktif).\n"
        "- **Feature Engineering** — Ditambahkan metrik `total_approved_units`, rasio kelulusan per semester, dan selisih peningkatan nilai."
    )

    col_view1, col_view2 = st.columns([2, 1])
    with col_view1:
        st.write(f"Ukuran Dataset: **{students.shape[0]:,} baris × {students.shape[1]} kolom**")
    with col_view2:
        search_query = st.text_input("Cari Student ID (contoh: STD_0010):", "")

    df_display = students
    if search_query:
        df_display = df_display[df_display[ID_COLUMN].str.contains(search_query.strip(), case=False, na=False)]

    st.dataframe(df_display.head(100), use_container_width=True)

# TAB 2: BUSINESS / ACADEMIC INTELLIGENCE (BI)
with tab_bi:
    st.subheader("Visualisasi Deskriptif: Faktor Kunci Akademik & Finansial")

    bi_col1, bi_col2 = st.columns(2)
    with bi_col1:
        st.markdown("**Distribusi Status Akhir Mahasiswa (Target)**")
        target_counts = students[RAW_TARGET].value_counts()
        st.bar_chart(target_counts)

    with bi_col2:
        st.markdown("**Tingkat Dropout berdasarkan Status Pelunasan SPP**")
        tuition_group = (
            students.groupby("Tuition fees up to date")[TARGET]
            .mean()
            .rename(index={0: "SPP Menunggak", 1: "SPP Lunas"})
        )
        st.bar_chart(tuition_group)

    st.markdown("---")
    bi_col3, bi_col4 = st.columns(2)
    with bi_col3:
        st.markdown("**Rata-rata Unit Matkul Lulus (Sem 1 vs Sem 2)**")
        sem_compare = pd.DataFrame({
            "Semester 1": [students["Curricular units 1st sem (approved)"].mean()],
            "Semester 2": [students["Curricular units 2nd sem (approved)"].mean()],
        }, index=["Rata-rata Unit Lulus"])
        st.bar_chart(sem_compare.T)

    with bi_col4:
        st.markdown("**Tingkat Dropout berdasarkan Status Beasiswa**")
        scholarship_group = (
            students.groupby("Scholarship holder")[TARGET]
            .mean()
            .rename(index={0: "Non-Beasiswa", 1: "Penerima Beasiswa"})
        )
        st.bar_chart(scholarship_group)

    st.markdown(f"**Profil Detail Mahasiswa Terpilih ({selected_id})**")
    detail_cols = [
        ID_COLUMN, "Age at enrollment", "Tuition fees up to date", "Debtor", "Scholarship holder",
        "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
        "total_approved_units", RAW_TARGET
    ]
    st.dataframe(pd.DataFrame([student_row[detail_cols]]), use_container_width=True)

# TAB 3: MACHINE LEARNING
with tab_ml:
    st.subheader("Prediksi Risiko Dropout Mahasiswa")

    if model is None:
        st.error("Model Machine Learning belum ditemukan (`student_dropout_pipeline.pkl`).")
        if st.button("Latih Model Random Forest Sekarang"):
            with st.spinner("Sedang melatih model..."):
                from ml_model import train_and_save_model
                trained_pipe, metrics_dict, fi_data = train_and_save_model()
                st.success("Model berhasil dilatih dan disimpan!")
                st.rerun()
    else:
        # Prediksi probabilitas
        prob_array = model.predict_proba(X_student)[0]
        dropout_prob = float(prob_array[1])
        is_pred_dropout = int(dropout_prob >= decision_threshold)
        actual_is_dropout = int(student_row[TARGET])

        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Probabilitas Dropout",
            f"{dropout_prob:.1%}",
            delta=f"{dropout_prob - decision_threshold:+.1%} vs threshold",
            delta_color="inverse",
        )
        p2.metric(
            "Hasil Prediksi",
            "🚨 Risiko Tinggi Dropout" if is_pred_dropout else "✅ Aman / Berpotensi Lulus",
            help=f"Decision Threshold saat ini: {decision_threshold:.2f}",
        )
        p3.metric(
            "Status Nyata (Dataset)",
            f"{student_row[RAW_TARGET]}",
            help="Status aktual yang tercatat dalam dataset",
        )

        st.info(
            f"Threshold saat ini: **{decision_threshold:.2f}**. Mahasiswa dengan skor probabilitas ≥ threshold "
            "diklasifikasikan memerlukan intervensi dini untuk mencegah putus studi."
        )

        # Metrik evaluasi model
        if model_metrics:
            st.markdown("**Kinerja Model (Evaluasi pada Test Set 20%):**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.3f}")
            m2.metric("Recall Dropout", f"{model_metrics.get('recall_dropout', 0):.3f}")
            m3.metric("F1-Score Dropout", f"{model_metrics.get('f1_dropout', 0):.3f}")
            m4.metric("ROC-AUC Score", f"{model_metrics.get('roc_auc', 0):.3f}")

        # Feature Importance
        if not feature_importance.empty:
            st.markdown("**Top 15 Faktor Terpenting dalam Model (Global Feature Importance):**")
            top_fi = feature_importance.head(15).set_index("feature")
            st.bar_chart(top_fi["importance"])

# TAB 4: LLM ACADEMIC ADVISOR
with tab_llm:
    st.subheader("LLM Academic Advisor — Asisten Rekomendasi Terstruktur")
    st.caption(f"Model Ollama: {DEFAULT_MODEL} (Dapat disesuaikan lewat environment variable OLLAMA_MODEL)")

    if model is None:
        st.warning("Silakan latih model terlebih dahulu di Tab 3 untuk menghasilkan skor probabilitas.")
    else:
        prob_array = model.predict_proba(X_student)[0]
        dropout_prob = float(prob_array[1])
        top_factors = (
            feature_importance.head(6).to_dict("records")
            if not feature_importance.empty
            else []
        )

        prompt_text = build_prompt(
            student_profile=student_row.to_dict(),
            dropout_probability=dropout_prob,
            academic_factors=top_factors,
        )

        with st.expander("🔍 Lihat Grounded Prompt yang dikirim ke LLM"):
            st.code(prompt_text, language="text")

        if st.button("Jalankan Analisis AI Academic Advisor", type="primary"):
            with st.spinner("AI sedang menganalisis profil mahasiswa dan probabilitas ML..."):
                response = generate_student_strategy(
                    student_profile=student_row.to_dict(),
                    dropout_probability=dropout_prob,
                    academic_factors=top_factors,
                )
            st.markdown(response)

st.markdown("---")
st.caption(
    "Prinsip Pembelajaran: Dataset menyediakan data; BI menyajikan statistik deskriptif; "
    "ML memprediksi probabilitas dropout; LLM mengomunikasikan bukti & saran tindakan taktis bagi institusi."
)
