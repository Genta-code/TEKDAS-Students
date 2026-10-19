"""Persiapan data mahasiswa untuk analisis Academic Intelligence dan prediksi Dropout.

Alur:
    dataset.csv -> validasi & penambahan student_id -> feature engineering -> train/test CSVs

Dijalankan dengan:
    python data_prep.py
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset.csv"

ID_COLUMN = "Student_ID"
RAW_TARGET = "Target"
TARGET = "is_dropout"

# Fitur kategorikal (kode kategorikal dalam dataset asli UCI)
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Fitur numerik akademik & makroekonomi
NUMERIC_COLUMNS = [
    "Application order",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    # Engineered features
    "total_approved_units",
    "approval_rate_1st_sem",
    "approval_rate_2nd_sem",
    "grade_improvement",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_raw_dataset(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """Membaca file dataset.csv asli dan menambahkan Student_ID."""
    df = pd.read_csv(base_dir / "dataset.csv")
    if ID_COLUMN not in df.columns:
        df.insert(0, ID_COLUMN, [f"STD_{i+1:04d}" for i in range(len(df))])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur turunan akademik."""
    df = df.copy()

    # Total mata kuliah lulus
    df["total_approved_units"] = (
        df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
    )

    # Tingkat kelulusan matkul semester 1
    enrolled_1 = df["Curricular units 1st sem (enrolled)"].replace(0, 1)
    df["approval_rate_1st_sem"] = (
        df["Curricular units 1st sem (approved)"] / enrolled_1
    ).clip(0, 1)

    # Tingkat kelulusan matkul semester 2
    enrolled_2 = df["Curricular units 2nd sem (enrolled)"].replace(0, 1)
    df["approval_rate_2nd_sem"] = (
        df["Curricular units 2nd sem (approved)"] / enrolled_2
    ).clip(0, 1)

    # Peningkatan nilai antara semester 1 dan semester 2
    df["grade_improvement"] = (
        df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    )

    # Binary target: 1 jika Dropout, 0 jika Graduate atau Enrolled
    if RAW_TARGET in df.columns and TARGET not in df.columns:
        df[TARGET] = (df[RAW_TARGET] == "Dropout").astype(int)

    return df


def prepare_student_features(single_student_df: pd.DataFrame) -> pd.DataFrame:
    """Mempersiapkan fitur untuk satu mahasiswa agar cocok dengan pipeline training."""
    prepared = engineer_features(single_student_df)
    return prepared


def main():
    print(f"Membaca dataset dari: {DATASET_FILE}")
    raw_df = load_raw_dataset()
    print(f"Total baris: {len(raw_df)}, Kolom awal: {raw_df.shape[1]}")

    df = engineer_features(raw_df)
    print(f"Distribusi Target:")
    print(df[RAW_TARGET].value_counts())
    print(f"Distribusi Binary Target ({TARGET} = 1 untuk Dropout):")
    print(df[TARGET].value_counts(normalize=True).round(3))

    # Split train-test
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET],
    )

    train_file = BASE_DIR / "student_train.csv"
    test_file = BASE_DIR / "student_test.csv"
    clean_all_file = BASE_DIR / "student_cleaned.csv"

    df.to_csv(clean_all_file, index=False)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Saved: {clean_all_file} ({len(df)} baris)")
    print(f"Saved: {train_file} ({len(train_df)} baris)")
    print(f"Saved: {test_file} ({len(test_df)} baris)")


if __name__ == "__main__":
    main()
