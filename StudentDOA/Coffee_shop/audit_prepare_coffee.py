"""Audit and prepare Coffee Shop Sales.xlsx for the Streamlit dashboard.

Run from this directory with: python audit_prepare_coffee.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "Coffee Shop Sales.xlsx"
CLEAN_PATH = BASE_DIR / "coffee_sales_cleaned.csv"
REPORT_PATH = BASE_DIR / "coffee_audit_report.json"

REQUIRED_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "transaction_time",
    "transaction_qty",
    "store_id",
    "store_location",
    "product_id",
    "unit_price",
    "product_category",
    "product_type",
    "product_detail",
}


def prepare_data() -> tuple[pd.DataFrame, dict]:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Workbook sumber tidak ditemukan: {SOURCE_PATH}")

    data = pd.read_excel(SOURCE_PATH, sheet_name="Transactions")
    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing_columns)}")

    original_rows = len(data)
    trim_changes: dict[str, int] = {}
    for column in data.select_dtypes(include=["object", "string"]).columns:
        original = data[column]
        cleaned = original.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        changed = int((original.fillna("") != cleaned.fillna("")).sum())
        if changed:
            trim_changes[column] = changed
        data[column] = cleaned.replace("", pd.NA)

    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"], errors="coerce"
    )
    invalid_dates = int(data["transaction_date"].isna().sum())
    data["transaction_date"] = data["transaction_date"].dt.strftime("%Y-%m-%d")

    data["transaction_qty"] = pd.to_numeric(data["transaction_qty"], errors="coerce")
    data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce")
    invalid_quantities = int(
        data["transaction_qty"].isna().sum()
        + data["transaction_qty"].le(0).sum()
    )
    invalid_prices = int(
        data["unit_price"].isna().sum() + data["unit_price"].le(0).sum()
    )

    rows_with_missing_values = int(data.isna().any(axis=1).sum())
    duplicate_rows = int(data.duplicated().sum())
    duplicate_transaction_ids = int(data["transaction_id"].duplicated().sum())

    price_variation = (
        data.groupby("product_id", dropna=False)["unit_price"]
        .agg(lambda prices: sorted(prices.dropna().unique().tolist()))
    )
    variable_prices = {
        str(product_id): [float(price) for price in prices]
        for product_id, prices in price_variation.items()
        if len(prices) > 1
    }

    data["sales_amount"] = (
        data["transaction_qty"] * data["unit_price"]
    ).round(2)
    report = {
        "source_file": SOURCE_PATH.name,
        "output_file": CLEAN_PATH.name,
        "row_count": int(original_rows),
        "date_min": str(data["transaction_date"].min()),
        "date_max": str(data["transaction_date"].max()),
        "trimmed_cells_by_column": trim_changes,
        "rows_dropped": 0,
        "sales_amount_added": True,
        "missing_value_rows": rows_with_missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_transaction_ids": duplicate_transaction_ids,
        "invalid_dates": invalid_dates,
        "invalid_or_missing_quantities": invalid_quantities,
        "invalid_or_missing_prices": invalid_prices,
        "product_ids_with_multiple_prices": variable_prices,
        "price_variation_note": (
            "Harga tidak otomatis dianggap salah atau diseragamkan. Workbook tidak "
            "menyertakan katalog maupun riwayat harga; variasi dapat mencerminkan "
            "perubahan harga/varian atau kesalahan input. Tandai untuk diperiksa "
            "terhadap sumber harga sebelum melakukan koreksi."
        ),
    }

    data.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return data, report


def main() -> None:
    data, report = prepare_data()
    trimmed_cells = sum(report["trimmed_cells_by_column"].values())
    print(f"Audit selesai: {report['row_count']:,} baris diproses.")
    print(f"Spasi tidak perlu dibersihkan: {trimmed_cells:,} sel.")
    print("Kolom sales_amount dihitung dari transaction_qty x unit_price.")
    print(f"Baris dibuang: {report['rows_dropped']}.")
    print(f"Total penjualan: ${data['sales_amount'].sum():,.2f}.")
    print(f"Data siap analisis: {CLEAN_PATH}")
    print(f"Catatan audit: {REPORT_PATH}")


if __name__ == "__main__":
    main()
