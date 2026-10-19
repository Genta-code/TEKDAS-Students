"""Prepare the E-Commerce Customer Behavior data for ML churn classification.

Flow:
    CSVs -> validation -> cleaning -> feature engineering -> train/test CSVs

Run:
    python data_prep.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

CUSTOMERS_FILE = BASE_DIR / "customers.csv"
ORDERS_FILE = BASE_DIR / "orders.csv"
PRODUCT_FILE = BASE_DIR / "product_summary.csv"
MONTHLY_FILE = BASE_DIR / "monthly_revenue.csv"

TARGET = "churned"
ID_COLUMN = "customer_id"

CATEGORICAL_COLUMNS = [
    "country",
    "gender",
    "membership_tier",
    "preferred_category",
    "preferred_device",
    "preferred_payment_method",
    "acquisition_channel",
]

NUMERIC_COLUMNS = [
    "age",
    "total_orders",
    "total_spend_usd",
    "avg_order_value_usd",
    "days_since_last_purchase",
    "reviews_given",
    "avg_review_score",
    "returns_made",
    "wishlist_items",
    "newsletter_subscribed",
    # engineered features
    "customer_tenure_days",
    "orders_per_year",
    "reviews_per_order",
    "returns_per_order",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_source_data(base_dir: Path = BASE_DIR):
    """Load the four CSVs supplied with the E-Commerce dataset."""
    customers = pd.read_csv(base_dir / "customers.csv")
    orders = pd.read_csv(base_dir / "orders.csv")
    products = pd.read_csv(base_dir / "product_summary.csv")
    monthly = pd.read_csv(base_dir / "monthly_revenue.csv")
    return customers, orders, products, monthly


def validate_sources(customers, orders, products, monthly):
    """Validate dataset schemas and required columns."""
    required = {
        "customers.csv": {ID_COLUMN, TARGET, "registration_date", *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS[:10]},
        "orders.csv": {"order_id", ID_COLUMN, "order_date", "order_status", "total_amount_usd", "returned"},
        "product_summary.csv": {"category", "product_name", "total_orders", "total_revenue_usd"},
        "monthly_revenue.csv": {"year", "month", "orders", "revenue_usd", "return_rate"},
    }
    frames = {
        "customers.csv": customers,
        "orders.csv": orders,
        "product_summary.csv": products,
        "monthly_revenue.csv": monthly,
    }
    for name, cols in required.items():
        missing = cols - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def prepare_customer_features(customers: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """Convert the customer table into model-ready raw features.

    Categorical columns remain text and are one-hot encoded inside the pipeline.
    """
    df = customers.copy()
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")

    if snapshot_date is None:
        snapshot_date = df["registration_date"].max()
    snapshot_date = pd.Timestamp(snapshot_date)

    df["customer_tenure_days"] = (snapshot_date - df["registration_date"]).dt.days.clip(lower=0)
    tenure_years = (df["customer_tenure_days"] / 365.25).clip(lower=1 / 365.25)
    order_denominator = df["total_orders"].replace(0, np.nan)

    df["orders_per_year"] = df["total_orders"] / tenure_years
    df["reviews_per_order"] = (df["reviews_given"] / order_denominator).fillna(0)
    df["returns_per_order"] = (df["returns_made"] / order_denominator).fillna(0)

    # Ensure binary target
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
        return df[[ID_COLUMN] + FEATURE_COLUMNS + [TARGET]]
    return df[[ID_COLUMN] + FEATURE_COLUMNS]


def build_data_quality_report(customers, orders, products, monthly) -> dict:
    """Create a summary report exposing key dataset properties."""
    delivered = orders[orders["order_status"].eq("Delivered")]
    non_cancelled = orders[~orders["order_status"].eq("Cancelled")]

    return {
        "customers_rows": int(len(customers)),
        "orders_rows": int(len(orders)),
        "products_rows": int(len(products)),
        "monthly_rows": int(len(monthly)),
        "customer_churn_rate_pct": round(float(customers[TARGET].mean() * 100), 2),
        "order_return_rate_pct": round(float(orders["returned"].mean() * 100), 2),
        "orders_missing_rating_pct": round(float(orders["customer_rating"].isna().mean() * 100), 2),
        "delivered_orders": int(len(delivered)),
        "delivered_revenue_usd": round(float(delivered["total_amount_usd"].sum()), 2),
        "product_summary_matches_non_cancelled_revenue": bool(
            np.isclose(products["total_revenue_usd"].sum(), non_cancelled["total_amount_usd"].sum(), atol=0.05)
        ),
        "monthly_summary_matches_delivered_revenue": bool(
            np.isclose(monthly["revenue_usd"].sum(), delivered["total_amount_usd"].sum(), atol=0.05)
        ),
        "monthly_return_rate_is_constant_zero": bool((monthly["return_rate"] == 0).all()),
        "note": (
            "monthly_revenue.csv represents delivered-order revenue; therefore its return_rate field is zero. "
            "Use orders.csv when analyzing returns."
        ),
    }


def main():
    customers, orders, products, monthly = load_source_data(BASE_DIR)
    validate_sources(customers, orders, products, monthly)

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    customers["registration_date"] = pd.to_datetime(customers["registration_date"], errors="coerce")

    snapshot_date = max(orders["order_date"].max(), customers["registration_date"].max())
    model_df = prepare_customer_features(customers, snapshot_date=snapshot_date)

    train_df, test_df = train_test_split(
        model_df,
        test_size=0.20,
        random_state=42,
        stratify=model_df[TARGET],
    )

    train_df.to_csv(BASE_DIR / "customer_train.csv", index=False)
    test_df.to_csv(BASE_DIR / "customer_test.csv", index=False)
    model_df.to_csv(BASE_DIR / "customer_ml_dataset.csv", index=False)

    report = build_data_quality_report(customers, orders, products, monthly)
    report["snapshot_date"] = str(snapshot_date.date())
    report["train_rows"] = int(len(train_df))
    report["test_rows"] = int(len(test_df))
    report["feature_count_before_one_hot"] = len(FEATURE_COLUMNS)

    with open(BASE_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Data preparation complete.")
    print(f"Snapshot date : {snapshot_date.date()}")
    print(f"Train / test : {len(train_df):,} / {len(test_df):,}")
    print(f"Churn rate   : {model_df[TARGET].mean() * 100:.2f}%")
    print("Created: customer_ml_dataset.csv, customer_train.csv, customer_test.csv, data_quality_report.json")


if __name__ == "__main__":
    main()
