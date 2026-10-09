"""Streamlit teaching dashboard: E-Commerce Customer Behavior (Data -> BI -> ML -> LLM).

Menjalankan dashboard:
    streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from data_prep import FEATURE_COLUMNS, prepare_customer_features
from vai_analyst import DEFAULT_MODEL, build_prompt, generate_customer_strategy

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="E-Commerce Customer Behavior: ML + LLM",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 E-Commerce Customer Behavior — dari Data ke ML dan LLM")
st.caption(
    "Proyek pembelajaran: Eksplorasi data transaksi e-commerce, prediksi customer churn dengan Machine Learning, "
    "dan gunakan LLM untuk membantu menyusun strategi retensi berbasis data."
)


@st.cache_data
def load_csvs():
    customers = pd.read_csv(BASE_DIR / "customers.csv")
    orders = pd.read_csv(BASE_DIR / "orders.csv")
    products = pd.read_csv(BASE_DIR / "product_summary.csv")
    monthly = pd.read_csv(BASE_DIR / "monthly_revenue.csv")
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    customers["registration_date"] = pd.to_datetime(customers["registration_date"], errors="coerce")
    return customers, orders, products, monthly


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "customer_churn_pipeline.pkl"
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


customers, orders, products, monthly = load_csvs()
model = load_model()
feature_importance, model_metrics = load_optional_outputs()

# Headline BI metrics
delivered = orders[orders["order_status"].eq("Delivered")]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(customers):,}")
col2.metric("Total Orders", f"{len(orders):,}")
col3.metric("Delivered Revenue", f"${delivered['total_amount_usd'].sum():,.0f}")
col4.metric("Observed Churn", f"{customers['churned'].mean():.1%}")

# Sidebar controls
st.sidebar.header("Pilih Pelanggan")

# Search and filter options
filter_churn = st.sidebar.selectbox("Filter Status Churn:", ["Semua", "Churned (1)", "Aktif (0)"])
if filter_churn == "Churned (1)":
    customer_list = customers.loc[customers["churned"] == 1, "customer_id"].tolist()
elif filter_churn == "Aktif (0)":
    customer_list = customers.loc[customers["churned"] == 0, "customer_id"].tolist()
else:
    customer_list = customers["customer_id"].tolist()

selected_id = st.sidebar.selectbox("Customer ID", customer_list)
decision_threshold = st.sidebar.slider("ML Decision Threshold", 0.10, 0.90, 0.50, 0.05)

customer_row = customers.loc[customers["customer_id"].eq(selected_id)].iloc[0]
customer_orders = orders.loc[orders["customer_id"].eq(selected_id)].sort_values("order_date", ascending=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Ringkasan Pelanggan:**")
st.sidebar.write(f"- **Negara:** {customer_row['country']}")
st.sidebar.write(f"- **Membership:** {customer_row['membership_tier']}")
st.sidebar.write(f"- **Total Spend:** ${customer_row['total_spend_usd']:,.2f}")
st.sidebar.write(f"- **Total Order:** {int(customer_row['total_orders'])}")
st.sidebar.write(f"- **Hari sejak order terakhir:** {int(customer_row['days_since_last_purchase'])} hari")

# Snapshot date & engineered feature vector for prediction
snapshot_date = max(orders["order_date"].max(), customers["registration_date"].max())
prepared = prepare_customer_features(customers.loc[customers["customer_id"].eq(selected_id)], snapshot_date)
X_customer = prepared[FEATURE_COLUMNS]

# Four main tabs
tab_data, tab_bi, tab_ml, tab_llm = st.tabs([
    "1 · E-Commerce Data",
    "2 · Business Intelligence",
    "3 · Machine Learning",
    "4 · LLM Analyst",
])

# TAB 1: DATASET
with tab_data:
    st.subheader("Empat Tabel Data Transaksi E-Commerce")
    st.markdown(
        "- **customers.csv** — satu baris per pelanggan; kolom `churned` adalah target klasifikasi ML.\n"
        "- **orders.csv** — satu baris per transaksi pesanan; detail barang, status delivery, diskon, dan return.\n"
        "- **product_summary.csv** — agregasi per produk (total pesanan, revenue, rating, return rate).\n"
        "- **monthly_revenue.csv** — agregasi tren bulanan untuk revenue yang berstatus Delivered."
    )

    dataset_name = st.selectbox("Lihat tabel", ["customers", "orders", "product_summary", "monthly_revenue"])
    frames = {
        "customers": customers,
        "orders": orders,
        "product_summary": products,
        "monthly_revenue": monthly,
    }
    df_view = frames[dataset_name]
    st.write(f"Ukuran: **{df_view.shape[0]:,} baris × {df_view.shape[1]} kolom**")
    st.dataframe(df_view.head(100), use_container_width=True)

    if (monthly["return_rate"] == 0).all():
        st.info(
            "💡 **Catatan Kualitas Data:** `monthly_revenue.return_rate` selalu 0 karena agregasi bulanan hanya mencakup "
            "pesanan berstatus *Delivered*. Analisis detail pengembalian barang (*returns*) sebaiknya menggunakan `orders.csv`."
        )

# TAB 2: BUSINESS INTELLIGENCE
with tab_bi:
    st.subheader("Descriptive BI: Eksplorasi Tren & Pola Transaksi")
    
    col_bi1, col_bi2 = st.columns(2)
    with col_bi1:
        st.markdown("**Delivered Revenue by Month**")
        monthly_plot = monthly.copy()
        monthly_plot["period"] = pd.to_datetime(dict(year=monthly_plot.year, month=monthly_plot.month, day=1))
        st.line_chart(monthly_plot.set_index("period")["revenue_usd"])

    with col_bi2:
        st.markdown("**Delivered Revenue by Category**")
        category_perf = (
            delivered.groupby("category", as_index=False)
            .agg(revenue_usd=("total_amount_usd", "sum"), orders=("order_id", "count"))
            .sort_values("revenue_usd", ascending=False)
        )
        st.bar_chart(category_perf.set_index("category")["revenue_usd"])

    st.markdown("---")
    st.markdown(f"**Riwayat Transaksi Pelanggan ({selected_id})**")
    order_cols = [
        "order_date", "product_name", "category", "total_amount_usd",
        "order_status", "returned", "customer_rating", "payment_method", "device_used"
    ]
    if not customer_orders.empty:
        st.dataframe(customer_orders[order_cols].head(25), use_container_width=True)
    else:
        st.write("Tidak ada data pesanan untuk pelanggan ini.")

# TAB 3: MACHINE LEARNING
with tab_ml:
    st.subheader("Prediksi Customer Churn")
    st.dataframe(pd.DataFrame([customer_row.to_dict()]), use_container_width=True)

    if model is None:
        st.error("Model Machine Learning belum ditemukan (`customer_churn_pipeline.pkl`).")
        if st.button("Latih Model Random Forest Sekarang", type="primary"):
            with st.spinner("Sedang mempersiapkan data dan melatih model Random Forest..."):
                from ml_model import train_and_save_model
                trained_pipe, metrics_dict, fi_data = train_and_save_model()
                st.success("Model berhasil dilatih dan disimpan!")
                st.rerun()
    else:
        churn_prob = float(model.predict_proba(X_customer)[0, 1])
        predicted = int(churn_prob >= decision_threshold)
        actual_label = int(customer_row["churned"])

        a, b, c = st.columns(3)
        a.metric(
            "Probabilitas Churn",
            f"{churn_prob:.1%}",
            delta=f"{churn_prob - decision_threshold:+.1%} vs threshold",
            delta_color="inverse",
        )
        b.metric(
            "Prediksi Model",
            "🚨 Churn (Beresiko)" if predicted else "✅ Tidak Churn (Aktif)",
            help=f"Threshold = {decision_threshold:.2f}",
        )
        c.metric(
            "Status Aktual (Dataset)",
            "Churn" if actual_label else "Tidak Churn",
            help="Label sebenarnya yang tercatat dalam dataset",
        )

        st.info(
            f"ℹ️ **Threshold Keputusan:** {decision_threshold:.2f}. "
            "Karena kasus churn biasanya bersifat minoritas (imbalanced), evaluasi kinerja model mengutamakan "
            "Recall, Precision, F1-Score, dan ROC-AUC dibandingkan akurasi semata."
        )

        if model_metrics:
            st.markdown("**Metrik Evaluasi Model (Test Set):**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.3f}")
            m2.metric("Recall Churn", f"{model_metrics.get('recall_churn', 0):.3f}")
            m3.metric("F1 Churn", f"{model_metrics.get('f1_churn', 0):.3f}")
            m4.metric("ROC-AUC", f"{model_metrics.get('roc_auc', 0):.3f}")

        if not feature_importance.empty:
            st.markdown("**Top 15 Faktor Model Terpenting (Global Feature Importance):**")
            top_fi = feature_importance.head(15).set_index("feature")
            st.bar_chart(top_fi["importance"])

# TAB 4: LLM ANALYST
with tab_llm:
    st.subheader("LLM Business Analyst — Rekomendasi Terstruktur & Berbasis Bukti")
    st.caption(f"Model Ollama: `{DEFAULT_MODEL}` (Dapat diubah melalui environment variable OLLAMA_MODEL)")

    if model is None:
        st.warning("Latih model terlebih dahulu di Tab 3 agar LLM menerima skor probabilitas nyata.")
    else:
        churn_prob = float(model.predict_proba(X_customer)[0, 1])
        recent_order_records = customer_orders[
            ["order_date", "product_name", "category", "total_amount_usd", "order_status", "returned"]
        ].head(5).copy()
        if not recent_order_records.empty:
            recent_order_records["order_date"] = recent_order_records["order_date"].dt.strftime("%Y-%m-%d")
        recent_order_records = recent_order_records.to_dict("records")
        model_factors = feature_importance.head(8).to_dict("records") if not feature_importance.empty else []

        prompt = build_prompt(
            customer_profile=customer_row.to_dict(),
            churn_probability=churn_prob,
            recent_orders=recent_order_records,
            model_factors=model_factors,
        )

        with st.expander("🔍 Lihat Prompt Grounded yang Dikirim ke LLM"):
            st.code(prompt, language="text")

        if st.button("Generate Evidence-Based Action Plan", type="primary"):
            with st.spinner("LLM sedang menganalisis profil pelanggan dan bukti transaksi..."):
                answer = generate_customer_strategy(
                    customer_profile=customer_row.to_dict(),
                    churn_probability=churn_prob,
                    recent_orders=recent_order_records,
                    model_factors=model_factors,
                )
            st.markdown(answer)

st.markdown("---")
st.caption("Prinsip Pembelajaran: Dataset menyediakan data; BI mendeskripsikan tren; ML memprediksi probabilitas churn; LLM mengomunikasikan bukti & rencana aksi retensi.")
