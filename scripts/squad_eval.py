# ruff: noqa: E402
"""
squad_eval.py

SQuAD 2.0 RAG Evaluation Pipeline.

Loads a balanced sample of answerable and unanswerable questions from the
SQuAD 2.0 validation set, ingests the context passages using the PRODUCTION
ingestion pipeline, runs the full RAG pipeline, and scores with:

  - LLM-as-a-Judge (semantic correctness via a separate model)
  - Exact Match (EM) — normalized string equality
  - Token F1 — bag-of-words precision/recall/F1
  - Recall@k — whether the gold passage appears in the top-k retrieval
  - Latency percentiles (p50, p95)

Run inside Docker:
    docker compose exec web python squad_eval.py
    docker compose exec web python squad_eval.py --samples 50
    docker compose exec web python squad_eval.py --keep-collection
"""

import argparse
import json
import os
import random
import re
import string
import time
from collections import Counter

# ── Django setup — must happen before importing any app modules ──
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

django.setup()

from django.conf import settings

# Override collection name so all retrieval calls hit the isolated eval collection
# (settings.COLLECTION_NAME is read at call-time, not import-time, so this works)
settings.COLLECTION_NAME = "squad_eval"

from datasets import load_dataset
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

# Import our existing RAG services (now pointed at squad_eval collection)
from apps.generation.services import ask
from apps.retrieval.ingest_service import ingest_document
from apps.retrieval.services import search_chunks

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

EVAL_COLLECTION = "squad_eval"
EMBED_DIM = settings.EMBED_DIM

# Use a DIFFERENT model than the generator for LLM-as-judge to reduce
# self-serving bias (generator uses gemini-2.5-flash)
JUDGE_MODEL = "gemini-3.1-flash-lite"

# Random seed for reproducible sampling
RANDOM_SEED = 42


# ────────────────────────────────────────────────────────────────────────────
# 1. Dataset loading
# ────────────────────────────────────────────────────────────────────────────


def load_squad_sample(n_samples: int) -> list[dict]:
    """
    Load a balanced, reproducibly-random sample from SQuAD 2.0 validation split.
    n_samples // 2 answerable + n_samples // 2 unanswerable.
    """
    print("Loading SQuAD 2.0 validation set from Hugging Face...")
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    answerable = [ex for ex in dataset if len(ex["answers"]["text"]) > 0]
    unanswerable = [ex for ex in dataset if len(ex["answers"]["text"]) == 0]

    half = n_samples // 2

    # Use random.sample with a fixed seed for reproducible, representative samples
    rng = random.Random(RANDOM_SEED)
    sampled_ans = rng.sample(answerable, min(half, len(answerable)))
    sampled_unans = rng.sample(unanswerable, min(half, len(unanswerable)))
    sampled = sampled_ans + sampled_unans

    ans_count = sum(1 for s in sampled if s["answers"]["text"])
    unans_count = len(sampled) - ans_count
    print(
        f"  Sampled {ans_count} answerable + {unans_count} unanswerable = {len(sampled)} total examples."
    )
    return sampled


# ────────────────────────────────────────────────────────────────────────────
# 2. Qdrant collection setup & ingestion (using production path)
# ────────────────────────────────────────────────────────────────────────────


def setup_eval_collection(qdrant_client: QdrantClient) -> None:
    """Drop and recreate a fresh squad_eval collection with dense + sparse schema."""
    if qdrant_client.collection_exists(EVAL_COLLECTION):
        print(f"  Dropping existing '{EVAL_COLLECTION}' collection...")
        qdrant_client.delete_collection(EVAL_COLLECTION)

    qdrant_client.create_collection(
        collection_name=EVAL_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print(f"  Created '{EVAL_COLLECTION}' with dense + sparse schema.")


def ingest_contexts_production(contexts: list[tuple[str, str]]) -> None:
    """
    Ingest unique (source_id, passage_text) tuples using the PRODUCTION
    ingestion pipeline. This tests the real chunking, parent-child logic,
    embedding batching, and Qdrant upsert path.
    """
    if not contexts:
        return

    for source_id, text in contexts:
        # Convert text to bytes and use the production ingest path
        file_bytes = text.encode("utf-8")
        filename = f"{source_id}.txt"
        try:
            ingest_document(filename=filename, file_bytes=file_bytes)
        except Exception as e:
            print(f"  WARNING: Failed to ingest context {source_id}: {e}")
            continue

    print(f"  Ingested {len(contexts)} context passages via production pipeline.")


# ────────────────────────────────────────────────────────────────────────────
# 3. Metrics: EM, Token F1, Recall@k, LLM Judge
# ────────────────────────────────────────────────────────────────────────────


def _normalize_answer(s: str) -> str:
    """Normalize answer for EM/F1: lowercase, remove punctuation/articles/extra whitespace."""
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    """
    Returns 1.0 if the normalized prediction exactly matches any gold answer.
    Only meaningful for answerable questions.
    """
    norm_pred = _normalize_answer(prediction)
    return float(any(_normalize_answer(gold) == norm_pred for gold in gold_answers))


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    """
    Compute the maximum token-level F1 between prediction and each gold answer.
    Only meaningful for answerable questions.
    """
    norm_pred = _normalize_answer(prediction)
    pred_tokens = norm_pred.split()

    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        norm_gold = _normalize_answer(gold)
        gold_tokens = norm_gold.split()
        if not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def recall_at_k(query: str, gold_source_id: str, k: int = 5) -> float:
    """
    Check if any of the top-k retrieved chunks come from a passage
    whose source matches the gold source ID.

    Returns 1.0 if the gold passage appears in the top-k, 0.0 otherwise.
    """
    try:
        hits = search_chunks(query, top_k=k)
        for hit in hits:
            source = hit.payload.get("source", "")
            # Match against the source_id filename we used during ingestion
            if gold_source_id in source:
                return 1.0
        return 0.0
    except Exception as e:
        print(f"  WARNING: Recall@k failed: {e}")
        return 0.0


def llm_judge(
    genai_client: genai.Client, question: str, prediction: str, gold_answers: list[str]
) -> int:
    """
    Uses a SEPARATE Gemini model (not the generator) to judge if the
    prediction is semantically correct. Returns 1 for correct, 0 for incorrect.
    """
    if gold_answers:
        prompt = (
            f"You are a strict evaluator grading a question answering system.\n\n"
            f"Question: {question}\n\n"
            f"Gold answers (any of these is considered correct): {gold_answers}\n\n"
            f"Predicted answer: {prediction}\n\n"
            f"Does the predicted answer correctly and unambiguously answer the question based on the gold answers? "
            f"It is acceptable if the predicted answer contains extra conversational text, as long as the core fact is correct. "
            f"Answer ONLY 'YES' or 'NO'."
        )
    else:
        prompt = (
            f"You are a strict evaluator grading a question answering system.\n\n"
            f"Question: {question}\n\n"
            f"Gold answers: NONE (This question is unanswerable based on the context).\n\n"
            f"Predicted answer: {prediction}\n\n"
            f"Did the predicted answer correctly indicate that the answer is not found, not provided, or unknown? "
            f"Answer ONLY 'YES' or 'NO'."
        )

    response = genai_client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
    )
    result = response.text.strip().upper()
    return 1 if "YES" in result else 0


# ────────────────────────────────────────────────────────────────────────────
# 4. Main evaluation loop
# ────────────────────────────────────────────────────────────────────────────


def run_evaluation(n_samples: int, keep_collection: bool) -> None:
    # ── Load dataset ──────────────────────────────────────────────
    examples = load_squad_sample(n_samples)

    # ── Init clients ─────────────────────────────────────────────
    qdrant_client = QdrantClient(url=settings.QDRANT_URL)
    genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # ── Setup collection ─────────────────────────────────────────
    print("\nSetting up isolated eval collection...")
    setup_eval_collection(qdrant_client)

    # ── Deduplicate and ingest contexts via production pipeline ───
    print("\nIngesting context passages via production pipeline...")
    seen: dict[str, tuple[str, str]] = {}
    for ex in examples:
        key = ex["context"][:120]  # dedup by first 120 chars
        if key not in seen:
            seen[key] = (ex["id"], ex["context"])

    unique_contexts = list(seen.values())
    print(f"  {len(unique_contexts)} unique passages from {len(examples)} examples.")
    ingest_contexts_production(unique_contexts)

    # ── Evaluate ─────────────────────────────────────────────────
    print(f"\nRunning evaluation on {len(examples)} questions...\n" + "─" * 80)

    results = []
    total_correct = 0
    ans_correct, ans_n = 0, 0
    unans_correct, unans_n = 0, 0
    total_em, total_f1 = 0.0, 0.0
    total_recall = 0.0
    latencies = []

    for i, ex in enumerate(examples, 1):
        question = ex["question"]
        gold_answers = ex["answers"]["text"]
        is_answerable = bool(gold_answers)
        source_id = ex["id"]

        # Run full RAG pipeline (rewrite → hybrid retrieval → Gemini generation)
        t0 = time.time()
        prediction = ask(question, history=[])
        elapsed = time.time() - t0
        latencies.append(elapsed)

        # Score with LLM-as-a-judge (using a different model than the generator)
        is_correct = llm_judge(genai_client, question, prediction, gold_answers)

        # Compute deterministic metrics for answerable questions
        em_score = 0.0
        f1_score = 0.0
        if is_answerable:
            em_score = exact_match(prediction, gold_answers)
            f1_score = token_f1(prediction, gold_answers)
            total_em += em_score
            total_f1 += f1_score

        # Compute retrieval metric: does the gold passage appear in top-k?
        r_at_k = recall_at_k(question, f"{source_id}.txt")
        total_recall += r_at_k

        if is_answerable:
            ans_correct += is_correct
            ans_n += 1
            label = "ANSWERABLE"
        else:
            unans_correct += is_correct
            unans_n += 1
            label = "UNANSWERABLE"

        total_correct += is_correct

        gold_display = gold_answers[0] if gold_answers else "(no answer exists)"
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"[{i:02d}/{len(examples)}] [{label}] {status} ({elapsed:.1f}s)")
        print(f"  Q:    {question}")
        print(f"  Gold: {gold_display}")
        print(
            f"  Pred: {prediction[:250].strip()}{'...' if len(prediction) > 250 else ''}"
        )
        if is_answerable:
            print(f"  EM: {em_score:.0f}  F1: {f1_score:.3f}  Recall@k: {r_at_k:.0f}")
        else:
            print(f"  Recall@k: {r_at_k:.0f}")
        print()

        results.append(
            {
                "id": ex["id"],
                "title": ex["title"],
                "question": question,
                "gold_answers": gold_answers,
                "prediction": prediction,
                "is_answerable": is_answerable,
                "is_correct": bool(is_correct),
                "exact_match": em_score,
                "token_f1": f1_score,
                "recall_at_k": r_at_k,
                "time_seconds": round(elapsed, 2),
            }
        )

    # ── Final report ─────────────────────────────────────────────
    n = len(examples)
    overall_acc_pct = 100 * total_correct / n
    avg_time = sum(latencies) / n
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]

    print("\n" + "═" * 60)
    print("       SQuAD 2.0 RAG Evaluation Results")
    print("═" * 60)
    print(f"  Total examples evaluated :  {n}")
    print(f"  Judge model              :  {JUDGE_MODEL}")
    print(f"  {'─' * 56}")
    print(f"  OVERALL LLM-Judge Acc    :  {overall_acc_pct:.1f}%")
    print(f"  {'─' * 56}")
    if ans_n:
        print(
            f"  Answerable Accuracy      :  {100 * ans_correct / ans_n:.1f}%  (n={ans_n})"
        )
        print(f"  Exact Match (EM)         :  {100 * total_em / ans_n:.1f}%")
        print(f"  Token F1                 :  {100 * total_f1 / ans_n:.1f}%")
    if unans_n:
        print(
            f"  Unanswerable Accuracy    :  {100 * unans_correct / unans_n:.1f}%  (n={unans_n})"
        )
    print(f"  {'─' * 56}")
    print(f"  Retrieval Recall@{settings.TOP_K}       :  {100 * total_recall / n:.1f}%")
    print(f"  {'─' * 56}")
    print(f"  Latency  mean            :  {avg_time:.2f}s")
    print(f"  Latency  p50             :  {p50:.2f}s")
    print(f"  Latency  p95             :  {p95:.2f}s")
    print("═" * 60)

    # ── Save JSON ────────────────────────────────────────────────
    output = {
        "summary": {
            "total": n,
            "judge_model": JUDGE_MODEL,
            "overall_accuracy_pct": round(overall_acc_pct, 2),
            "answerable": {
                "count": ans_n,
                "accuracy_pct": round(100 * ans_correct / ans_n, 2) if ans_n else 0,
                "exact_match_pct": round(100 * total_em / ans_n, 2) if ans_n else 0,
                "token_f1_pct": round(100 * total_f1 / ans_n, 2) if ans_n else 0,
            },
            "unanswerable": {
                "count": unans_n,
                "accuracy_pct": round(100 * unans_correct / unans_n, 2)
                if unans_n
                else 0,
            },
            "retrieval": {
                "recall_at_k": round(100 * total_recall / n, 2),
                "k": settings.TOP_K,
            },
            "latency": {
                "avg_s": round(avg_time, 2),
                "p50_s": round(p50, 2),
                "p95_s": round(p95, 2),
            },
        },
        "results": results,
    }
    with open("squad_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n  Detailed results saved → squad_eval_results.json")

    # ── Cleanup ──────────────────────────────────────────────────
    if not keep_collection:
        qdrant_client.delete_collection(EVAL_COLLECTION)
        print(
            f"  Deleted '{EVAL_COLLECTION}' collection. (use --keep-collection to retain it)"
        )
    else:
        print(f"  Kept '{EVAL_COLLECTION}' collection for debugging.")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SQuAD 2.0 RAG Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Total number of examples to evaluate (half answerable, half unanswerable).",
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep the squad_eval Qdrant collection after evaluation (useful for debugging).",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        default=False,
        help="Enable HyDE (Hypothetical Document Embeddings) for retrieval.",
    )
    parser.add_argument(
        "--hyde-mode",
        default="replace",
        choices=["replace", "ensemble"],
        help="The mode for HyDE ('replace' or 'ensemble').",
    )
    parser.add_argument(
        "--crag",
        action="store_true",
        default=False,
        help="Enable CRAG (Corrective RAG) for evaluating retrieval quality.",
    )
    args = parser.parse_args()

    # Override settings for this run
    settings.USE_HYDE = args.hyde
    settings.HYDE_MODE = args.hyde_mode
    settings.CRAG_ENABLED = args.crag

    run_evaluation(n_samples=args.samples, keep_collection=args.keep_collection)
