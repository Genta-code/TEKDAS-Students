"""LLM layer: turn ML output + business facts into an explainable action suggestion.

Standalone validation:
    ollama serve
    ollama pull llama3.2
    python vai_analyst.py

Useful options:
    python vai_analyst.py --model llama3.2
    python vai_analyst.py --prompt-only
"""
import argparse
import os
import sys
from typing import Any, Dict, Iterable
import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def _compact_dict(data: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: data.get(k) for k in keys if k in data}


def build_prompt(
    customer_profile: Dict[str, Any],
    churn_probability: float,
    recent_orders: list | None = None,
    model_factors: list | None = None,
) -> str:
    """Build a grounded prompt students can inspect before calling the LLM."""
    profile = _compact_dict(customer_profile, [
        "customer_id", "country", "age", "gender", "membership_tier",
        "total_orders", "total_spend_usd", "avg_order_value_usd",
        "days_since_last_purchase", "preferred_category", "preferred_device",
        "preferred_payment_method", "acquisition_channel", "reviews_given",
        "avg_review_score", "returns_made", "wishlist_items",
        "newsletter_subscribed",
    ])

    return f"""
You are a Business Intelligence teaching assistant.
Use ONLY the supplied data. The machine-learning score is a prediction, not a proven cause.
Do not invent customer facts, motives, income, complaints, or future behavior.

CUSTOMER PROFILE
{profile}

ML OUTPUT
Predicted probability of churn: {churn_probability:.1%}

RECENT ORDER EVIDENCE
{recent_orders or 'No order rows available for this customer.'}

GLOBAL MODEL FACTORS
{model_factors or 'Not supplied.'}

TASK
Write in Indonesian for a business manager:
1. One sentence interpreting the churn probability.
2. Two evidence-based observations from the supplied customer/order data.
3. Two practical retention actions that are proportional to the evidence.
4. One short caution stating what cannot be concluded from this model.
Keep the answer concise and use bullet points.
""".strip()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prepared prompt to Ollama and return only its text response."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def generate_customer_strategy(
    customer_profile: Dict[str, Any],
    churn_probability: float,
    recent_orders: list | None = None,
    model_factors: list | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_prompt(
        customer_profile=customer_profile,
        churn_probability=churn_probability,
        recent_orders=recent_orders,
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
    """Parse command-line options for standalone validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validasi vai_analyst.py secara mandiri dengan mengirim profil demo "
            "langsung ke Ollama, tanpa menjalankan app.py."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Nama model Ollama yang diuji (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Cetak prompt demo tanpa menghubungi Ollama.",
    )
    return parser.parse_args()


def run_standalone(model: str = DEFAULT_MODEL, prompt_only: bool = False) -> int:
    """Run an end-to-end demo and return a process-friendly status code."""
    demo_customer = {
        "customer_id": "C-DEMO",
        "country": "Indonesia",
        "age": 34,
        "membership_tier": "Free",
        "total_orders": 4,
        "total_spend_usd": 286.63,
        "days_since_last_purchase": 120,
        "preferred_category": "Electronics",
        "newsletter_subscribed": 0,
    }
    demo_orders = [
        {
            "order_date": "2026-01-12",
            "product_name": "Wireless Mouse",
            "category": "Electronics",
            "total_amount_usd": 45.50,
            "order_status": "Delivered",
            "returned": 0,
        }
    ]
    demo_factors = [
        {"feature": "days_since_last_purchase", "importance": 0.18},
        {"feature": "membership_tier_Free", "importance": 0.09},
    ]
    prompt = build_prompt(
        customer_profile=demo_customer,
        churn_probability=0.72,
        recent_orders=demo_orders,
        model_factors=demo_factors,
    )

    print("=== Validasi Standalone vAI Analyst ===")
    print(f"Model: {model}")

    if prompt_only:
        print("\n=== Prompt Demo ===")
        print(prompt)
        return 0

    print("Menghubungi Ollama dan mengirim prompt demo...")
    try:
        answer = call_ollama(prompt=prompt, model=model)
    except Exception as exc:
        print(
            (
                f"VALIDASI GAGAL: tidak dapat menggunakan model '{model}'.\n"
                f"Detail: {exc}\n\n"
                "Pastikan langkah berikut sudah dijalankan:\n"
                "  1. ollama serve\n"
                f"  2. ollama pull {model}\n"
                f"  3. python vai_analyst.py --model {model}"
            ),
            file=sys.stderr,
        )
        return 1

    print("\nVALIDASI BERHASIL: Ollama memberikan respons.")
    print("\n=== Jawaban vAI Analyst ===")
    print(answer)
    return 0


def main() -> int:
    args = parse_args()
    return run_standalone(model=args.model, prompt_only=args.prompt_only)


if __name__ == "__main__":
    raise SystemExit(main())
