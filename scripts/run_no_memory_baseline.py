#!/usr/bin/env python3
"""
No-Memory (closed-book) Baseline for AgriMemSynth.

DeepSeek-V3 answers held-out test questions with ZERO retrieved context,
testing whether the five memory systems add value over the LLM's
pre-trained knowledge (Reviewer 2, comment 2.1).

Generation only. LLM-as-judge is run separately with a different model.
Runs via OpenRouter (OpenAI-compatible).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import requests
from tqdm import tqdm


API_BASE = "https://openrouter.ai/api/v1"


CLOSED_BOOK_PROMPT = """You are an agricultural expert. Answer the following question directly from your own knowledge.

Question: {question}

Instructions:
1. Answer concisely and directly
2. If the question refers to a specific consultation, visit, or conversation you cannot see, answer "unknown"
3. If the question asks general agricultural knowledge (crops, diseases, pathogens, treatments), answer from your own knowledge
4. Do not add explanations or hedging

Answer:"""


def call_llm(api_key, messages, max_tokens=100, model=None):
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.0},
        timeout=60,
    )
    if resp.status_code != 200:
        return f"ERROR:{resp.status_code}:{resp.text[:200]}", False
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip(), True
    except Exception:
        return f"ERROR:parse:{data}", False


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def calculate_f1(pred, ref):
    pred_t = set(normalize(pred).split())
    ref_t = set(normalize(ref).split())
    if not pred_t or not ref_t:
        return 0.0
    common = pred_t & ref_t
    p = len(common) / len(pred_t)
    r = len(common) / len(ref_t)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def calculate_em(pred, ref):
    return float(normalize(pred) == normalize(ref))


def calculate_bleu1(pred, ref):
    pred_t = normalize(pred).split()
    ref_t = set(normalize(ref).split())
    if not pred_t:
        return 0.0
    return sum(1 for t in pred_t if t in ref_t) / len(pred_t)


def main():
    parser = argparse.ArgumentParser(description="No-memory baseline (generation only)")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--model", required=True,
                        help="OpenRouter model id, e.g. deepseek/deepseek-chat-v3-0324")
    parser.add_argument("--data-dir", default="data/text")
    parser.add_argument("--output", default="results/no_memory_baseline_predictions.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: only process N questions")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: no API key. Pass --api-key or set OPENROUTER_API_KEY")
        sys.exit(1)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load questions --------------------------------------------------
    with open(Path(args.data_dir) / "agri_locomo_v2" / "test.json") as f:
        locomo = json.load(f)
    locomo_samples = locomo.get("samples", locomo)

    with open(Path(args.data_dir) / "agri_hotpotqa_v4" / "test.json") as f:
        hot_samples = json.load(f)
    if isinstance(hot_samples, dict):
        hot_samples = hot_samples.get("samples", [])

    questions = []  # (dataset, sample_id, category, question, gold)
    for s in locomo_samples:
        for qa in s.get("qa", []):
            questions.append(("AgriConvMem", s["sample_id"],
                              qa.get("category_name", ""),
                              qa["question"], str(qa["answer"])))
    for s in hot_samples:
        questions.append(("AgriMultiHop", s.get("id", "?"),
                          s.get("type", ""),
                          s["question"], str(s["answer"])))

    if args.limit:
        questions = questions[:args.limit]

    print(f"Total questions: {len(questions)}")
    print(f"Model: {args.model}")

    results = []
    errors = 0

    for ds, sid, cat, q, gold in tqdm(questions, desc="Closed-book"):
        prompt = CLOSED_BOOK_PROMPT.format(question=q)
        pred, ok = call_llm(args.api_key,
                            [{"role": "user", "content": prompt}],
                            max_tokens=100, model=args.model)
        if not ok or pred.startswith("ERROR"):
            errors += 1
            pred = "error"

        results.append({
            "dataset": ds, "sample_id": sid, "category": cat,
            "question": q, "reference": gold, "prediction": pred,
            "f1": calculate_f1(pred, gold),
            "em": calculate_em(pred, gold),
            "bleu1": calculate_bleu1(pred, gold),
        })
        time.sleep(0.3)

    # ---- Aggregate ---------------------------------------------------------
    def agg(ds):
        r = [x for x in results if x["dataset"] == ds]
        if not r:
            return None
        return {
            "n": len(r),
            "f1": sum(x["f1"] for x in r) / len(r),
            "em": sum(x["em"] for x in r) / len(r),
            "bleu1": sum(x["bleu1"] for x in r) / len(r),
        }

    conv = agg("AgriConvMem")
    multi = agg("AgriMultiHop")

    out = {
        "config": {"model": args.model, "provider": "openrouter",
                   "temperature": 0.0, "max_tokens": 100,
                   "judge": False,
                   "prompt": CLOSED_BOOK_PROMPT,
                   "run_at": datetime.now().isoformat()},
        "overall": {
            "n": len(results),
            "f1": sum(x["f1"] for x in results) / len(results),
            "em": sum(x["em"] for x in results) / len(results),
            "bleu1": sum(x["bleu1"] for x in results) / len(results),
            "errors": errors,
        },
        "agri_conv_mem": conv,
        "agri_multi_hop": multi,
        "predictions": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("NO-MEMORY BASELINE RESULTS (lexical only)")
    print("=" * 60)
    print(f"Overall  : F1 {out['overall']['f1']:.4f}  EM {out['overall']['em']:.4f}  "
          f"BLEU1 {out['overall']['bleu1']:.4f}  errors {errors}")
    if conv:
        print(f"ConvMem  : F1 {conv['f1']:.4f}  EM {conv['em']:.4f}  BLEU1 {conv['bleu1']:.4f}")
    if multi:
        print(f"MultiHop : F1 {multi['f1']:.4f}  EM {multi['em']:.4f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
