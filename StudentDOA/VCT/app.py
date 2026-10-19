"""Streamlit teaching dashboard: VCT 2025 Esports Intelligence (Data -> BI -> ML -> LLM).

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
    TARGET,
    clean_players_data,
    load_source_data,
)
from vai_analyst import DEFAULT_MODEL, build_prompt, generate_coach_strategy

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "vct_2025"

st.set_page_config(
    page_title="VCT 2025 Esports Intelligence: ML + LLM",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 VCT 2025 Esports Intelligence — dari Data ke ML dan LLM")
st.caption(
    "Proyek pembelajaran: Eksplorasi statistik pro player Valorant Champions Tour 2025, "
    "prediksi performa High-Impact dengan Machine Learning, dan gunakan LLM Coach untuk menyusun rencana taktis berbasis data."
)


@st.cache_data
def load_all_data():
    players_raw, scores, maps_scores, agents_pick = load_source_data(DATA_DIR)
    players_clean = clean_players_data(players_raw)
    return players_raw, players_clean, scores, maps_scores, agents_pick


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "vct_player_pipeline.pkl"
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


players_raw, players_clean, scores, maps_scores, agents_pick = load_all_data()
model = load_model()
feature_importance, model_metrics = load_optional_outputs()

# Headline BI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pro Players", f"{players_clean['Player'].nunique():,}")
col2.metric("Total Matches", f"{len(scores):,}")
col3.metric("Avg Combat Score (ACS)", f"{players_clean['Average Combat Score'].mean():.1f}")
col4.metric("High-Impact Ratio (Observed)", f"{players_clean[TARGET].mean():.1%}")

# Sidebar controls
st.sidebar.header("Pilih Pro Player")

# Team filter
all_teams = ["Semua Tim"] + sorted(players_clean["Teams"].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Filter Tim:", all_teams)

filtered_players_df = players_clean
if selected_team != "Semua Tim":
    filtered_players_df = filtered_players_df[filtered_players_df["Teams"] == selected_team]

available_players = sorted(filtered_players_df["Player"].unique().tolist())
selected_player = st.sidebar.selectbox("Player ID / Ign:", available_players)

# Player's match performance records
player_records = players_clean[players_clean["Player"] == selected_player].reset_index(drop=True)
if len(player_records) > 1:
    record_options = [
        f"#{i+1} - {row['Tournament']} ({row['Stage']}) | Agent: {row['Agents']}"
        for i, row in player_records.iterrows()
    ]
    selected_record_idx = st.sidebar.selectbox(
        "Pilih Rekor Match / Sesi:",
        range(len(record_options)),
        format_func=lambda x: record_options[x],
    )
else:
    selected_record_idx = 0

player_row = player_records.iloc[selected_record_idx]
decision_threshold = st.sidebar.slider("ML Decision Threshold (High-Impact)", 0.10, 0.90, 0.50, 0.05)

# Sidebar Quick Profile
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Profil Sesi ({selected_player}):**")
st.sidebar.write(f"- **Tim:** {player_row['Teams']}")
st.sidebar.write(f"- **Agen:** {player_row['Agents']}")
st.sidebar.write(f"- **ACS:** {player_row['Average Combat Score']:.0f}")
st.sidebar.write(f"- **K/D Ratio:** {player_row['Kills:Deaths']:.2f}")
st.sidebar.write(f"- **ADR:** {player_row['Average Damage Per Round']:.1f}")
st.sidebar.write(f"- **KAST:** {player_row['KAST_pct']*100:.1f}%")
st.sidebar.write(f"- **Headshot:** {player_row['Headshot_pct']*100:.1f}%")
st.sidebar.write(f"- **Rating Nyata:** {player_row['Rating']:.2f}")

# Prepare features for ML prediction
X_player = pd.DataFrame([player_row])[FEATURE_COLUMNS]

# Four main tabs
tab_data, tab_bi, tab_ml, tab_llm = st.tabs([
    "1 · VCT Esports Data",
    "2 · Esports Intelligence",
    "3 · Machine Learning",
    "4 · LLM Coach & Analyst",
])

# TAB 1: DATA EXPLORER
with tab_data:
    st.subheader("Data Statistik Valorant Champions Tour (VCT) 2025")
    st.markdown(
        "- **players_stats.csv** — Rekor performa individual pemain (ACS, K/D, KAST%, First Kills, Rating per turnamen/stage).\n"
        "- **scores.csv** — Rekapitulasi hasil pertandingan antar tim (Team A vs Team B, Skor, Pemenang).\n"
        "- **maps_scores.csv** — Rincian per map (Attacker/Defender round score, Overtime, Durasi map).\n"
        "- **agents_pick_rates.csv** — Tingkat pick rate agen per map dan turnamen."
    )

    dataset_name = st.selectbox(
        "Lihat tabel data",
        ["players_stats (cleaned)", "scores", "maps_scores", "agents_pick_rates"]
    )
    frames = {
        "players_stats (cleaned)": players_clean,
        "scores": scores,
        "maps_scores": maps_scores,
        "agents_pick_rates": agents_pick,
    }
    df_view = frames[dataset_name]
    st.write(f"Ukuran: **{df_view.shape[0]:,} baris × {df_view.shape[1]} kolom**")
    st.dataframe(df_view.head(100), use_container_width=True)

    st.info(
        "💡 **Data-Quality Insight:** Kolom persentase seperti `Kill, Assist, Trade, Survive %`, `Headshot %`, "
        "dan `Clutch Success %` telah distandardisasi menjadi bentuk desimal rasio (0.0 - 1.0). "
        "Performa dengan Rating ≥ 1.05 dikategorikan sebagai *High-Impact*."
    )

# TAB 2: ESPORTS INTELLIGENCE (BI)
with tab_bi:
    st.subheader("Descriptive BI: Tren Kemenangan, Meta Agen, dan Leaderboard")

    col_bi1, col_bi2 = st.columns(2)
    with col_bi1:
        st.markdown("**Top 10 Tim dengan Kemenangan Match Terbanyak**")
        top_winners = scores["Match Result"].value_counts().head(10)
        st.bar_chart(top_winners)

    with col_bi2:
        st.markdown("**Leaderboard: Top 10 Pemain berdasarkan Rata-rata ACS**")
        player_agg = (
            players_clean.groupby("Player")
            .agg(
                Avg_ACS=("Average Combat Score", "mean"),
                Avg_Rating=("Rating", "mean"),
                Rounds_Sum=("Rounds Played", "sum")
            )
            .query("Rounds_Sum >= 50")
            .sort_values("Avg_ACS", ascending=False)
            .head(10)
        )
        st.bar_chart(player_agg["Avg_ACS"])

    st.markdown("---")
    st.markdown(f"**Riwayat Pertandingan Pro Player: {selected_player}**")
    show_cols = [
        "Tournament", "Stage", "Match Type", "Teams", "Agents", "Rounds Played",
        "Rating", "Average Combat Score", "Kills:Deaths", "KAST_pct", "Average Damage Per Round"
    ]
    st.dataframe(player_records[show_cols], use_container_width=True)

# TAB 3: MACHINE LEARNING
with tab_ml:
    st.subheader("Prediksi Performa High-Impact Pro Player")
    st.dataframe(pd.DataFrame([player_row.to_dict()]), use_container_width=True)

    if model is None:
        st.error("Model Machine Learning belum ditemukan (`vct_player_pipeline.pkl`).")
        if st.button("Latih Model Random Forest Sekarang", type="primary"):
            with st.spinner("Sedang melatih model Random Forest pada dataset VCT 2025..."):
                from ml_model import train_and_save_model
                trained_pipe, metrics_dict, fi_data = train_and_save_model()
                st.success("Model berhasil dilatih dan disimpan!")
                st.rerun()
    else:
        prob_array = model.predict_proba(X_player)[0]
        impact_prob = float(prob_array[1])
        predicted_class = int(impact_prob >= decision_threshold)
        actual_is_high = int(player_row[TARGET])

        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Probabilitas High-Impact",
            f"{impact_prob:.1%}",
            delta=f"{impact_prob - decision_threshold:+.1%} vs threshold",
            delta_color="normal",
        )
        p2.metric(
            "Prediksi Model",
            "🌟 High-Impact (Elite)" if predicted_class else "🛡️ Regular (Role Player)",
            help=f"Threshold = {decision_threshold:.2f}",
        )
        p3.metric(
            "Status Nyata (Dataset)",
            "High-Impact (Rating ≥ 1.05)" if actual_is_high else "Under / Regular (< 1.05)",
            help=f"Rating Aktual: {player_row['Rating']:.2f}",
        )

        st.info(
            f"ℹ️ **Threshold Keputusan:** {decision_threshold:.2f}. "
            "Pemain dengan probabilitas ≥ threshold diklasifikasikan sebagai kontributor kunci dengan dampak pertandingan signifikan."
        )

        if model_metrics:
            st.markdown("**Metrik Evaluasi Model (Test Set):**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.3f}")
            m2.metric("Precision High-Impact", f"{model_metrics.get('precision_high_impact', 0):.3f}")
            m3.metric("Recall High-Impact", f"{model_metrics.get('recall_high_impact', 0):.3f}")
            m4.metric("ROC-AUC Score", f"{model_metrics.get('roc_auc', 0):.3f}")

        if not feature_importance.empty:
            st.markdown("**Top 15 Faktor Terpenting Penentu Performa (Global Feature Importance):**")
            top_fi = feature_importance.head(15).set_index("feature")
            st.bar_chart(top_fi["importance"])

# TAB 4: LLM COACH & ANALYST
with tab_llm:
    st.subheader("LLM Esports Coach & Analyst — Rekomendasi Taktis Berbasis Bukti")
    st.caption(f"Model Ollama: `{DEFAULT_MODEL}` (Dapat diubah via environment variable OLLAMA_MODEL)")

    if model is None:
        st.warning("Latih model terlebih dahulu di Tab 3 agar LLM menerima probabilitas performa yang nyata.")
    else:
        prob_array = model.predict_proba(X_player)[0]
        impact_prob = float(prob_array[1])
        top_factors = feature_importance.head(8).to_dict("records") if not feature_importance.empty else []

        prompt_text = build_prompt(
            player_profile=player_row.to_dict(),
            impact_probability=impact_prob,
            model_factors=top_factors,
        )

        with st.expander("🔍 Lihat Grounded Prompt yang Dikirim ke LLM Coach"):
            st.code(prompt_text, language="text")

        if st.button("Generate Tactical Coaching Plan", type="primary"):
            with st.spinner("AI Coach sedang menyusun analisis taktis dan evaluasi performa..."):
                coaching_plan = generate_coach_strategy(
                    player_profile=player_row.to_dict(),
                    impact_probability=impact_prob,
                    model_factors=top_factors,
                )
            st.markdown(coaching_plan)

st.markdown("---")
st.caption(
    "Prinsip Pembelajaran: Dataset VCT 2025 menyediakan histori turnamen; BI memetakan tren & leaderboard; "
    "ML mengukur probabilitas performa high-impact; LLM Coach merumuskan arahan taktis berbasis bukti."
)
