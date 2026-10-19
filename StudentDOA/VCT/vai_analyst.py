"""LLM Esports Analyst / VCT Coach: Turn player performance metrics & ML probability into actionable coaching strategies.

Standalone validation:
    ollama serve
    ollama pull llama3.2
    python vai_analyst.py
"""
import argparse
import os
import sys
from typing import Any, Dict
import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(
    player_profile: Dict[str, Any],
    impact_probability: float,
    model_factors: list | None = None,
) -> str:
    """Build a grounded prompt for VCT tactical coaching & analysis."""
    return f"""
You are a Valorant Champions Tour (VCT) Esports Head Coach and Analyst.
Use ONLY the supplied player data. The machine-learning score represents the probability of achieving a High-Impact performance (Rating >= 1.05).
Do not invent player background, personal life, or facts not present in the data.

PLAYER PERFORMANCE PROFILE
- Player: {player_profile.get('Player')}
- Team: {player_profile.get('Teams')}
- Agents: {player_profile.get('Agents')}
- Stage: {player_profile.get('Stage')}
- Average Combat Score (ACS): {player_profile.get('Average Combat Score')}
- Kills:Deaths (K/D): {player_profile.get('Kills:Deaths')}
- KAST % (Kill/Assist/Trade/Survive): {player_profile.get('KAST_pct', 0) * 100:.1f}%
- Average Damage Per Round (ADR): {player_profile.get('Average Damage Per Round')}
- Kills Per Round (KPR): {player_profile.get('Kills Per Round')}
- First Kills Per Round (FKPR): {player_profile.get('First Kills Per Round')}
- First Deaths Per Round (FDPR): {player_profile.get('First Deaths Per Round')}
- Headshot %: {player_profile.get('Headshot_pct', 0) * 100:.1f}%
- Actual Rating: {player_profile.get('Rating')}

ML MODEL PREDICTION
- Predicted Probability of High-Impact Performance: {impact_probability:.1%}

GLOBAL MODEL FACTORS
{model_factors or 'Not supplied.'}

TASK
Tuliskan dalam Bahasa Indonesia untuk Head Coach / Analis Tim Esports:
1. Satu kalimat interpretasi atas probabilitas performa High-Impact pemain.
2. Dua observasi berbasis bukti statistik (ACS, KAST%, rasio duel/first kills, atau efisiensi agen).
3. Dua rekomendasi taktis/latihan yang spesifik dan proporsional untuk meningkatkan performa round (misal setup utilitas, entry dueling, trade killing).
4. Satu batasan/peringatan hal yang tidak dapat disimpulkan hanya dari metrik statistik individual ini.
Jawab secara ringkas dan gunakan poin bullet.
""".strip()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send prompt to Ollama and return text response."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def generate_coach_strategy(
    player_profile: Dict[str, Any],
    impact_probability: float,
    model_factors: list | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_prompt(
        player_profile=player_profile,
        impact_probability=impact_probability,
        model_factors=model_factors,
    )

    try:
        return call_ollama(prompt=prompt, model=model)
    except Exception as exc:
        return (
            f"Tidak dapat menghubungi Ollama model '{model}'. "
            f"Pastikan `ollama serve` aktif dan model sudah tersedia. Detail: {exc}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Validasi vai_analyst VCT Coach secara mandiri.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Nama model Ollama (default: {DEFAULT_MODEL}).")
    parser.add_argument("--prompt-only", action="store_true", help="Cetak prompt tanpa memanggil Ollama.")
    return parser.parse_args()


def run_standalone(model: str = DEFAULT_MODEL, prompt_only: bool = False) -> int:
    demo_player = {
        "Player": "Aspas",
        "Teams": "Leviatán",
        "Agents": "jett",
        "Stage": "Playoffs",
        "Average Combat Score": 268.0,
        "Kills:Deaths": 1.45,
        "KAST_pct": 0.78,
        "Average Damage Per Round": 165.0,
        "Kills Per Round": 0.92,
        "First Kills Per Round": 0.24,
        "First Deaths Per Round": 0.12,
        "Headshot_pct": 0.38,
        "Rating": 1.28,
    }
    demo_factors = [
        {"feature": "Average Combat Score", "importance": 0.35},
        {"feature": "Average Damage Per Round", "importance": 0.25},
        {"feature": "Kills:Deaths", "importance": 0.18},
    ]
    prompt = build_prompt(demo_player, impact_probability=0.88, model_factors=demo_factors)

    print("=== Validasi Standalone VCT Coach Analyst ===")
    print(f"Model: {model}")
    if prompt_only:
        print("\n=== Prompt Demo ===\n" + prompt)
        return 0

    print("Menghubungi Ollama...")
    try:
        answer = call_ollama(prompt, model=model)
        print("\n=== Jawaban VCT Coach ===\n" + answer)
        return 0
    except Exception as exc:
        print(f"Gagal memanggil Ollama: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_standalone(model=args.model, prompt_only=args.prompt_only))
