"""Prepare the VCT 2025 Esports dataset for ML player-impact classification.

Flow:
    vct_2025 CSVs -> validation -> cleaning -> feature engineering -> train/test CSVs

Run:
    python data_prep.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "vct_2025"

PLAYERS_FILE = DATA_DIR / "players_stats" / "players_stats.csv"
SCORES_FILE = DATA_DIR / "matches" / "scores.csv"
MAPS_SCORES_FILE = DATA_DIR / "matches" / "maps_scores.csv"
AGENTS_PICK_FILE = DATA_DIR / "agents" / "agents_pick_rates.csv"

TARGET = "is_high_impact"
ID_COLUMN = "Player"

NUMERIC_COLUMNS = [
    "Average Combat Score",
    "Kills:Deaths",
    "KAST_pct",
    "Average Damage Per Round",
    "Kills Per Round",
    "Assists Per Round",
    "First Kills Per Round",
    "First Deaths Per Round",
    "Headshot_pct",
    "Clutch_pct",
    "Rounds Played",
    "Maximum Kills in a Single Map",
]

CATEGORICAL_COLUMNS = [
    "Teams",
    "Agents",
    "Stage",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def parse_percentage(val) -> float:
    """Safely convert percentage string (e.g., '69%') to float (0.69)."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).replace("%", "").strip()
    try:
        return float(val_str) / 100.0
    except ValueError:
        return 0.0


def load_source_data(data_dir: Path = DATA_DIR):
    """Load core CSV tables from vct_2025 folder."""
    players = pd.read_csv(data_dir / "players_stats" / "players_stats.csv")
    scores = pd.read_csv(data_dir / "matches" / "scores.csv")
    maps_scores = pd.read_csv(data_dir / "matches" / "maps_scores.csv")
    agents_pick = pd.read_csv(data_dir / "agents" / "agents_pick_rates.csv")
    return players, scores, maps_scores, agents_pick


def clean_players_data(players: pd.DataFrame) -> pd.DataFrame:
    """Clean player statistics and perform feature engineering."""
    df = players.copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"]).copy()

    df["KAST_pct"] = df["Kill, Assist, Trade, Survive %"].apply(parse_percentage)
    df["Headshot_pct"] = df["Headshot %"].apply(parse_percentage)
    df["Clutch_pct"] = df["Clutch Success %"].apply(parse_percentage)

    # Numerical cleans
    for col in ["Average Combat Score", "Kills:Deaths", "Average Damage Per Round",
                "Kills Per Round", "Assists Per Round", "First Kills Per Round",
                "First Deaths Per Round", "Rounds Played", "Maximum Kills in a Single Map"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Target: High-impact performer (Rating >= 1.05 represents elite/above-average tier)
    df[TARGET] = (df["Rating"] >= 1.05).astype(int)
    return df


def prepare_player_feature_row(row_series: pd.Series) -> pd.DataFrame:
    """Prepare a single player record for inference."""
    df = pd.DataFrame([row_series])
    return df[FEATURE_COLUMNS]


def main():
    print("Loading VCT 2025 datasets...")
    players, scores, maps_scores, agents_pick = load_source_data(DATA_DIR)

    print("Cleaning player statistics...")
    cleaned_df = clean_players_data(players)

    # Split dataset
    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=0.20,
        random_state=42,
        stratify=cleaned_df[TARGET],
    )

    train_df.to_csv(BASE_DIR / "vct_player_train.csv", index=False)
    test_df.to_csv(BASE_DIR / "vct_player_test.csv", index=False)
    cleaned_df.to_csv(BASE_DIR / "vct_player_ml_dataset.csv", index=False)

    report = {
        "total_raw_player_rows": int(len(players)),
        "valid_rating_rows": int(len(cleaned_df)),
        "total_unique_players": int(cleaned_df["Player"].nunique()),
        "total_teams": int(cleaned_df["Teams"].nunique()),
        "total_matches_scores": int(len(scores)),
        "total_maps_played": int(len(maps_scores)),
        "high_impact_rate_pct": round(float(cleaned_df[TARGET].mean() * 100), 2),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    with open(BASE_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Data preparation complete.")
    print(f"Cleaned records : {len(cleaned_df):,}")
    print(f"High-Impact Rate: {cleaned_df[TARGET].mean() * 100:.2f}%")
    print(f"Train / Test    : {len(train_df):,} / {len(test_df):,}")
    print("Files created: vct_player_train.csv, vct_player_test.csv, vct_player_ml_dataset.csv, data_quality_report.json")


if __name__ == "__main__":
    main()
