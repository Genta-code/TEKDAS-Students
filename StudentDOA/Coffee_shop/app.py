"""Streamlit BI dashboard for the prepared Coffee Shop Sales data.

Run audit_prepare_coffee.py first, then: streamlit run app.py
"""

from __future__ import annotations

import io
import itertools
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "coffee_sales_cleaned.csv"
REPORT_PATH = BASE_DIR / "coffee_audit_report.json"
WEEKDAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
WEEKDAY_MAP = dict(enumerate(WEEKDAY_NAMES))
DAY_FILTERS = ["Semua hari", "Hari kerja", "Akhir pekan"]

st.set_page_config(page_title="Coffee Shop Sales BI", page_icon="☕", layout="wide")
st.title("Coffee Shop Sales — Business Intelligence")
st.caption(
    "Ringkasan penjualan, transaksi, kategori, produk, dan jam transaksi "
    "berdasarkan data 1 Januari–30 Juni 2023."
)

if not DATA_PATH.exists() or not REPORT_PATH.exists():
    st.error(
        "Data hasil audit belum tersedia. Jalankan "
        "`python audit_prepare_coffee.py` dari folder Coffee_shop terlebih dahulu."
    )
    st.stop()


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"], errors="coerce"
    )
    data["transaction_time"] = pd.to_datetime(
        data["transaction_time"].astype(str), format="%H:%M:%S", errors="coerce"
    )
    if data["sales_amount"].isna().any() or data["transaction_date"].isna().any():
        raise ValueError("Data siap analisis berisi tanggal atau nilai penjualan kosong.")
    return data


@st.cache_data
def load_audit_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def filter_days(frame: pd.DataFrame, day_filter: str) -> pd.DataFrame:
    if day_filter == "Hari kerja":
        return frame.loc[frame["transaction_date"].dt.dayofweek < 5]
    if day_filter == "Akhir pekan":
        return frame.loc[frame["transaction_date"].dt.dayofweek >= 5]
    return frame


def make_excel_report(
    summary: pd.DataFrame, rows: pd.DataFrame
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Ringkasan", index=False)
        rows.to_excel(writer, sheet_name="Data terfilter", index=False)
    return output.getvalue()


def make_pdf_report(
    title: str, period_text: str, summary_rows: list[list[str]], insight_rows: list[str]
) -> bytes:
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(title, styles["Title"]),
        Paragraph(period_text, styles["Normal"]),
        Spacer(1, 14),
    ]
    table = Table(summary_rows, colWidths=[2.6 * inch, 3.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5d9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f7")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.extend([table, Spacer(1, 14), Paragraph("Temuan", styles["Heading2"])])
    elements.extend(Paragraph(f"• {item}", styles["BodyText"]) for item in insight_rows)
    document.build(elements)
    return output.getvalue()


data = load_data()
audit_report = load_audit_report()
store_options = sorted(data["store_location"].dropna().unique().tolist())
category_options = sorted(data["product_category"].dropna().unique().tolist())
min_date = data["transaction_date"].min().date()
max_date = data["transaction_date"].max().date()

st.sidebar.header("Filter analisis")
selected_stores = st.sidebar.multiselect(
    "Gerai",
    options=store_options,
    default=store_options,
    help="Pilih gerai untuk memperbarui semua angka dan grafik.",
)
selected_dates = st.sidebar.date_input(
    "Periode transaksi",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
selected_day_filter = st.sidebar.selectbox("Hari transaksi", DAY_FILTERS)
comparison_mode = st.sidebar.radio(
    "Perbandingan periode",
    ["Bulan ke bulan (MoM)", "Tahun ke tahun (YoY)"],
    help="Membandingkan bulan terbaru dalam cakupan dengan bulan pembandingnya.",
)

if not selected_stores:
    st.warning("Pilih minimal satu gerai untuk menampilkan hasil.")
    st.stop()
if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir untuk menjalankan analisis.")
    st.stop()

period_start, period_end = selected_dates
if period_start > period_end:
    st.warning("Tanggal awal harus sama dengan atau sebelum tanggal akhir.")
    st.stop()

selected_dates_mask = data["transaction_date"].dt.date.between(
    period_start, period_end
)
selected_store_data = data.loc[
    data["store_location"].isin(selected_stores) & selected_dates_mask
].copy()
filtered = filter_days(selected_store_data, selected_day_filter).copy()
if filtered.empty:
    st.warning("Tidak ada transaksi pada kombinasi gerai, periode, dan hari ini.")
    st.stop()

filtered["hour"] = filtered["transaction_time"].dt.hour
filtered["weekday"] = filtered["transaction_date"].dt.dayofweek.map(WEEKDAY_MAP)
transaction_sales = filtered.groupby("transaction_id")["sales_amount"].sum()
total_sales = float(filtered["sales_amount"].sum())
transaction_count = int(filtered["transaction_id"].nunique())
units_sold = int(filtered["transaction_qty"].sum())
average_transaction_value = (
    float(transaction_sales.mean()) if not transaction_sales.empty else 0.0
)
period_label = f"{period_start:%d %b %Y}–{period_end:%d %b %Y}"
store_label = ", ".join(selected_stores)

st.subheader("Ringkasan kinerja")
st.caption(
    f"Periode: {period_label}  ·  Gerai: {store_label}  ·  "
    f"Hari: {selected_day_filter.lower()}"
)
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Total penjualan", f"${total_sales:,.2f}")
metric_2.metric("Jumlah transaksi", f"{transaction_count:,}")
metric_3.metric("Unit terjual", f"{units_sold:,}")
metric_4.metric("Rata-rata nilai transaksi", f"${average_transaction_value:,.2f}")

latest_month = filtered["transaction_date"].max().to_period("M")
current_month_start = max(
    period_start, latest_month.start_time.date()
)
current_month_end = min(period_end, latest_month.end_time.date())
comparison_month_offset = -1 if comparison_mode.startswith("Bulan") else -12
comparison_period = latest_month + comparison_month_offset
comparison_month_start = comparison_period.start_time.date()
comparison_month_end = comparison_period.end_time.date()
comparison_start = (
    comparison_month_start
    if current_month_start == latest_month.start_time.date()
    else comparison_month_start.replace(
        day=min(current_month_start.day, comparison_month_end.day)
    )
)
comparison_end = (
    comparison_month_end
    if current_month_end == latest_month.end_time.date()
    else comparison_month_end.replace(
        day=min(current_month_end.day, comparison_month_end.day)
    )
)
comparison_rows = data.loc[
    data["store_location"].isin(selected_stores)
    & data["transaction_date"].dt.date.between(comparison_start, comparison_end)
]
comparison_rows = filter_days(comparison_rows, selected_day_filter)
if comparison_rows.empty:
    compare_name = "bulan sebelumnya" if comparison_month_offset == -1 else "tahun sebelumnya"
    st.info(
        f"Perbandingan {comparison_mode.split('(')[1].rstrip(')')} belum tersedia: "
        f"data untuk {compare_name} tidak ada dalam dataset (cakupan "
        f"{min_date:%d %b %Y}–{max_date:%d %b %Y})."
    )
else:
    current_month_rows = filtered.loc[
        filtered["transaction_date"].dt.date.between(
            current_month_start, current_month_end
        )
    ]
    current_month_sales = float(current_month_rows["sales_amount"].sum())
    previous_month_sales = float(comparison_rows["sales_amount"].sum())
    current_month_transactions = int(
        current_month_rows["transaction_id"].nunique()
    )
    previous_month_transactions = int(
        comparison_rows["transaction_id"].nunique()
    )
    sales_delta_pct = (
        (current_month_sales / previous_month_sales - 1) * 100
        if previous_month_sales
        else None
    )
    transaction_delta_pct = (
        (current_month_transactions / previous_month_transactions - 1) * 100
        if previous_month_transactions
        else None
    )
    compare_label = (
        "MoM" if comparison_month_offset == -1 else "YoY"
    )
    st.caption(
        f"Perbandingan {compare_label}: {current_month_start:%d %b}–"
        f"{current_month_end:%d %b %Y} vs {comparison_start:%d %b}–"
        f"{comparison_end:%d %b %Y} · penjualan "
        f"${current_month_sales:,.2f} vs ${previous_month_sales:,.2f} "
        f"({f'{sales_delta_pct:+.1f}%' if sales_delta_pct is not None else 'basis pembanding $0.00'}) "
        f"· transaksi {current_month_transactions:,} vs {previous_month_transactions:,} "
        f"({f'{transaction_delta_pct:+.1f}%' if transaction_delta_pct is not None else 'basis pembanding 0'})."
    )

cost_column = next(
    (
        column
        for column in ("cogs_amount", "cost_amount", "cost_of_goods_sold", "hpp_amount")
        if column in filtered.columns
    ),
    None,
)
if cost_column:
    gross_profit = total_sales - float(filtered[cost_column].sum())
    gross_margin = gross_profit / total_sales if total_sales else 0.0
    st.metric("Gross profit margin", f"{gross_margin:.1%}")
else:
    st.caption(
        "Gross profit margin belum dapat dihitung: data tidak memuat HPP/biaya modal."
    )

st.divider()
chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Penjualan per gerai")
    store_sales = (
        filtered.groupby("store_location", as_index=False)["sales_amount"]
        .sum()
        .sort_values("sales_amount", ascending=False)
    )
    store_chart = px.bar(
        store_sales,
        x="store_location",
        y="sales_amount",
        labels={"store_location": "Gerai", "sales_amount": "Penjualan ($)"},
        text_auto=",.2f",
    )
    store_chart.update_traces(
        hovertemplate="%{x}<br>Penjualan: $%{y:,.2f}<extra></extra>"
    )
    store_chart.update_layout(
        margin=dict(l=90, r=20, t=25, b=80),
        xaxis_title_standoff=18,
        yaxis_title_standoff=18,
    )
    st.plotly_chart(store_chart, width="stretch")
    if len(store_sales) > 1:
        highest_store = store_sales.iloc[0]
        lowest_store = store_sales.iloc[-1]
        gap = float(highest_store["sales_amount"] - lowest_store["sales_amount"])
        st.caption(
            f"Temuan: {highest_store['store_location']} tertinggi "
            f"(${highest_store['sales_amount']:,.2f}); selisih dari "
            f"{lowest_store['store_location']} terendah adalah ${gap:,.2f}."
        )

with chart_right:
    st.subheader("Kontribusi penjualan per kategori")
    category_sales = (
        filtered.groupby("product_category", as_index=False)["sales_amount"]
        .sum()
        .sort_values("sales_amount", ascending=False)
    )
    category_chart = px.pie(
        category_sales,
        names="product_category",
        values="sales_amount",
        hole=0.55,
        labels={"product_category": "Kategori", "sales_amount": "Penjualan"},
    )
    category_chart.update_traces(
        hovertemplate=(
            "%{label}<br>Penjualan: $%{value:,.2f}"
            "<br>Kontribusi: %{percent:.1%}<extra></extra>"
        )
    )
    st.plotly_chart(category_chart, width="stretch")
    leading_category = category_sales.iloc[0]
    category_share = float(leading_category["sales_amount"]) / total_sales
    st.caption(
        f"Kategori terbesar: {leading_category['product_category']} — "
        f"${leading_category['sales_amount']:,.2f} ({category_share:.1%})."
    )

product_left, product_right = st.columns(2)
with product_left:
    st.subheader("Produk teratas / terbawah")
    product_category = st.selectbox(
        "Kategori produk",
        ["Semua kategori", *category_options],
        key="product_category_filter",
    )
    product_rank = st.radio(
        "Peringkat produk",
        ["Teratas", "Terbawah"],
        horizontal=True,
        key="product_rank",
    )
    products = filtered
    if product_category != "Semua kategori":
        products = products.loc[products["product_category"] == product_category]
    product_sales = products.groupby("product_detail", as_index=False)[
        "sales_amount"
    ].sum()
    if product_sales.empty:
        st.info("Tidak ada produk pada kategori yang dipilih.")
    else:
        ascending = product_rank == "Terbawah"
        ranked_products = product_sales.nsmallest(10, "sales_amount") if ascending else product_sales.nlargest(10, "sales_amount")
        ranked_products = ranked_products.sort_values("sales_amount", ascending=True)
        product_chart = px.bar(
            ranked_products,
            x="sales_amount",
            y="product_detail",
            orientation="h",
            labels={"product_detail": "Produk", "sales_amount": "Penjualan ($)"},
            text_auto=",.2f",
            color_discrete_sequence=["#00A86B"],
        )
        product_chart.update_traces(
            hovertemplate="%{y}<br>Penjualan: $%{x:,.2f}<extra></extra>"
        )
        product_chart.update_layout(margin=dict(l=20, r=20, t=20, b=45))
        st.plotly_chart(product_chart, width="stretch")
        finding = ranked_products.iloc[-1] if not ascending else ranked_products.iloc[0]
        label = "tertinggi" if not ascending else "terendah"
        st.success(
            f"Penjualan {label}: {finding['product_detail']} — "
            f"${finding['sales_amount']:,.2f}."
        )
        st.caption(
            "Peringkat berdasarkan nilai penjualan (kuantitas × harga), bukan jumlah unit."
        )

with product_right:
    st.subheader("Transaksi berdasarkan jam")
    hourly_transactions = (
        filtered.dropna(subset=["hour"])
        .groupby("hour")["transaction_id"]
        .nunique()
        .rename("transactions")
        .reset_index()
    )
    hour_chart = px.bar(
        hourly_transactions,
        x="hour",
        y="transactions",
        labels={"hour": "Jam", "transactions": "Jumlah transaksi"},
    )
    hour_chart.update_traces(
        hovertemplate="%{x:02d}:00<br>Transaksi: %{y:,}<extra></extra>"
    )
    hour_chart.update_layout(margin=dict(l=20, r=20, t=20, b=55))
    st.plotly_chart(hour_chart, width="stretch")
    if not hourly_transactions.empty:
        peak_hour = hourly_transactions.loc[
            hourly_transactions["transactions"].idxmax()
        ]
        st.caption(
            f"Jam transaksi terbanyak: {int(peak_hour['hour']):02d}:00 "
            f"({int(peak_hour['transactions']):,} transaksi)."
        )
    with st.popover("ⓘ Catatan teknis"):
        st.write(
            "Jumlah transaksi per jam tidak menunjukkan panjang antrean. Data ini "
            "tidak memuat waktu tunggu, durasi layanan, atau jumlah staf."
        )

st.divider()
st.subheader("Tren penjualan")
trend_granularity = st.radio(
    "Granularitas tren",
    ["Harian", "Mingguan"],
    horizontal=True,
)
trend_data = filtered.copy()
if trend_granularity == "Harian":
    trend_data["tanggal"] = trend_data["transaction_date"].dt.floor("D")
    trend_label = "Tanggal"
else:
    trend_data["tanggal"] = (
        trend_data["transaction_date"].dt.to_period("W").dt.start_time
    )
    trend_label = "Awal minggu"
sales_trend = (
    trend_data.groupby("tanggal", as_index=False)["sales_amount"]
    .sum()
    .sort_values("tanggal")
)
trend_chart = px.line(
    sales_trend,
    x="tanggal",
    y="sales_amount",
    markers=True,
    labels={"tanggal": trend_label, "sales_amount": "Penjualan ($)"},
)
trend_chart.update_traces(
    hovertemplate="%{x|%d %b %Y}<br>Penjualan: $%{y:,.2f}<extra></extra>"
)
st.plotly_chart(trend_chart, width="stretch")

st.subheader("Heatmap transaksi — hari dan jam")
heatmap_data = filtered.dropna(subset=["hour"]).copy()
heatmap_data["hari"] = heatmap_data["transaction_date"].dt.dayofweek.map(WEEKDAY_MAP)
heatmap_pivot = (
    heatmap_data.groupby(["hari", "hour"])["transaction_id"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(index=WEEKDAY_NAMES, columns=range(24), fill_value=0)
)
heatmap_chart = px.imshow(
    heatmap_pivot,
    labels={"x": "Jam", "y": "Hari", "color": "Transaksi"},
    x=[f"{hour:02d}:00" for hour in heatmap_pivot.columns],
    y=heatmap_pivot.index,
    aspect="auto",
    color_continuous_scale="YlOrRd",
)
heatmap_chart.update_layout(margin=dict(l=20, r=20, t=20, b=50))
st.plotly_chart(heatmap_chart, width="stretch")

st.subheader("Analisis penjualan silang")
if filtered["transaction_id"].duplicated(keep=False).any():
    product_groups = (
        filtered.groupby("transaction_id")["product_detail"]
        .agg(lambda values: sorted(set(values)))
    )
    pairs = (
        pair
        for products_in_transaction in product_groups
        for pair in itertools.combinations(products_in_transaction, 2)
    )
    basket_counts = pd.Series(pairs, dtype="object").value_counts().head(10)
    if basket_counts.empty:
        st.info("Belum ditemukan transaksi dengan lebih dari satu produk berbeda.")
    else:
        basket_table = pd.DataFrame(
            {
                "Pasangan produk": [
                    f"{pair[0]} + {pair[1]}" for pair in basket_counts.index
                ],
                "Jumlah transaksi bersama": basket_counts.values,
            }
        )
        st.dataframe(basket_table, width="stretch", hide_index=True)
else:
    st.info(
        "Analisis pasangan produk tidak tersedia: setiap transaction_id pada data "
        "saat ini hanya memiliki satu baris produk. Diperlukan ID transaksi yang "
        "dipakai bersama oleh beberapa item dalam satu keranjang."
    )

st.divider()
st.subheader("Insight dan pemeriksaan lanjutan")
if not store_sales.empty and not category_sales.empty:
    insights = [
        f"{transaction_count:,} transaksi menghasilkan penjualan ${total_sales:,.2f}.",
        f"{leading_category['product_category']} adalah kategori terbesar "
        f"(${leading_category['sales_amount']:,.2f}; {category_share:.1%}).",
    ]
    if len(store_sales) > 1:
        insights.append(
            f"{highest_store['store_location']} mencatat penjualan gerai tertinggi; "
            f"selisih dari gerai terendah ${gap:,.2f}."
        )
    for insight in insights:
        st.markdown(f"- {insight}")
    st.caption(
        "Ringkasan ini menunjukkan pola penjualan, bukan penyebab perubahan atau "
        "panjang antrean. Periksa data operasional sebelum mengubah jadwal staf."
    )

st.subheader("Unduh laporan")
export_left, export_right = st.columns(2)
summary_export = pd.DataFrame(
    [
        {"Metrik": "Periode", "Nilai": period_label},
        {"Metrik": "Gerai", "Nilai": store_label},
        {"Metrik": "Filter hari", "Nilai": selected_day_filter},
        {"Metrik": "Total penjualan", "Nilai": total_sales},
        {"Metrik": "Jumlah transaksi", "Nilai": transaction_count},
        {"Metrik": "Unit terjual", "Nilai": units_sold},
        {"Metrik": "Rata-rata nilai transaksi", "Nilai": average_transaction_value},
    ]
)
excel_bytes = make_excel_report(summary_export, filtered.drop(columns=["hour", "weekday"], errors="ignore"))
with export_left:
    st.download_button(
        "Unduh Excel",
        data=excel_bytes,
        file_name="coffee_shop_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

pdf_bytes = make_pdf_report(
    "Laporan Penjualan Coffee Shop",
    f"Periode: {period_label} | Gerai: {store_label} | Hari: {selected_day_filter}",
    [["Metrik", "Nilai"]]
    + [[row["Metrik"], str(row["Nilai"])] for row in summary_export.to_dict("records")],
    insights,
)
with export_right:
    st.download_button(
        "Unduh PDF",
        data=pdf_bytes,
        file_name="coffee_shop_report.pdf",
        mime="application/pdf",
    )

with st.expander("Hasil audit dan persiapan data"):
    trimmed = audit_report["trimmed_cells_by_column"]
    st.write(
        f"{audit_report['row_count']:,} baris dibaca; "
        f"{sum(trimmed.values()):,} sel teks dirapikan; "
        f"{audit_report['rows_dropped']} baris dibuang; "
        "nilai penjualan dihitung sebagai kuantitas × harga."
    )
    st.write(
        "Variasi harga tidak otomatis dianggap kesalahan karena katalog maupun "
        "riwayat harga tidak tersedia. Verifikasi terhadap sumber harga sebelum koreksi."
    )
    st.json(
        {
            "baris_dengan_nilai_kosong": audit_report["missing_value_rows"],
            "baris_duplikat": audit_report["duplicate_rows"],
            "transaction_id_duplikat": audit_report["duplicate_transaction_ids"],
            "tanggal_tidak_valid": audit_report["invalid_dates"],
            "kuantitas_tidak_valid": audit_report["invalid_or_missing_quantities"],
            "harga_tidak_valid": audit_report["invalid_or_missing_prices"],
            "product_id_dengan_beberapa_harga": len(
                audit_report["product_ids_with_multiple_prices"]
            ),
        }
    )
