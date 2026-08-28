#!/usr/bin/env python3
"""
Re-judge all system predictions with an independent model via ModelArk.

Inputs:
  - results/full_benchmark_v2/benchmark_results_complete.json  (5 systems)
  - results/no_memory_baseline_deepseek_v3.json                 (no-memory)

Output:
  - results/judged_seed_2_0_pro.json  (same structure, adds judge_seed_2_0_pro)

Judge model: seed-2-0-pro-260328 (independent from DeepSeek-V3 generator).
Keeps original DeepSeek-V3 self-judge label as llm_judge_deepseek_v3 for comparison.
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


API_BASE = "https://ark.ap-southeast.bytepluses.com/api/coding/v3"
JUDGE_MODEL = "seed-2-0-pro-260328"


# Same semantics as the released benchmark's LLM_JUDGE_PROMPT
JUDGE_PROMPT = """Evaluate if the generated answer is correct compared to the gold answer.

Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {response}

Guidelines:
- Be generous: if the answer captures the key information, mark as CORRECT
- Partial matches that contain the essential facts are CORRECT
- For disease names, treatments, or technical terms, allow synonyms and variations
- For dates/times, flexible matching is OK if contextually correct
- Focus on whether the core information is present, not exact wording

Return ONLY a JSON object: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


def call_judge(api_key, question, gold, response, model=JUDGE_MODEL):
    """Return True/False/None (None on error or unparseable)."""
    prompt = JUDGE_PROMPT.format(question=question, gold_answer=gold, response=response)
    try:
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "system",
                                "content": "You are an expert grader."},
                               {"role": "user", "content": prompt}],
                  "max_tokens": 20, "temperature": 0.0},
            timeout=120,
        )
    except requests.RequestException as e:
        return None, f"network:{e}"

    if resp.status_code != 200:
        return None, f"http:{resp.status_code}"

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None, "parse:choices"

    m = re.search(r'\{\s*"label"\s*:\s*"(\w+)"\s*\}', content)
    if m:
        return m.group(1).lower() == "correct", content
    # Fallback: accept bare keyword
    low = content.lower()
    if "correct" in low and "wrong" not in low:
        return True, content
    if "wrong" in low:
        return False, content
    return None, content


def main():
    parser = argparse.ArgumentParser(description="Re-judge with ModelArk")
    parser.add_argument("--api-key", default=os.getenv("MODELARK_API_KEY"))
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--input-released",
                        default="results/full_benchmark_v2/benchmark_results_complete.json")
    parser.add_argument("--input-nomem",
                        default="results/no_memory_baseline_deepseek_v3.json")
    parser.add_argument("--output", default="results/judged_seed_2_0_pro.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-system", default=None,
                        help="Resume: start from this system name")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: no API key. Pass --api-key or set MODELARK_API_KEY")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load previously partial output for resume
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Resuming from existing: {out_path} ({len(existing.get('systems',{}))} systems done)")

    # ---- Collect all judgments to make ---------------------------------
    jobs = []  # (system, dataset, pred_index, question, gold, response)

    # Released 5 systems
    with open(args.input_released) as f:
        released = json.load(f)
    for sys_name, sys_data in released.items():
        for ds_key in ["locomo", "hotpotqa"]:
            preds = sys_data.get(ds_key, {}).get("predictions", [])
            for i, p in enumerate(preds):
                jobs.append((sys_name, ds_key, i,
                             p.get("question", ""), str(p.get("reference", "")),
                             str(p.get("prediction", ""))))

    # No-memory baseline
    with open(args.input_nomem) as f:
        nomem = json.load(f)
    for i, p in enumerate(nomem.get("predictions", [])):
        jobs.append(("no_memory", p["dataset"], i,
                     p.get("question", ""), str(p.get("reference", "")),
                     str(p.get("prediction", ""))))

    if args.limit:
        jobs = jobs[:args.limit]

    print(f"Total judgments: {len(jobs)}")
    print(f"Judge model: {args.model}")
    print(f"~{len(jobs)*1.3/60:.0f} min estimated")

    # ---- Run judgments -------------------------------------------------
    results = {}  # (sys, ds, i) -> (bool/None, raw)
    skipped = 0
    errors = 0

    # Resume: skip jobs already judged
    def already_done(sys_name, ds_key, i):
        key = f"{sys_name}::{ds_key}::{i}"
        if key in existing.get("_judge_cache", {}):
            return existing["_judge_cache"][key]
        return None

    cache = existing.get("_judge_cache", {})
    start_idx = 0
    if args.start_system:
        # Fast-forward to first job of start_system
        for idx, (s, d, i, *_) in enumerate(jobs):
            if s == args.start_system:
                start_idx = idx
                break
        print(f"Resuming from job {start_idx} (system {args.start_system})")

    for idx in range(start_idx, len(jobs)):
        sys_name, ds_key, i, q, gold, resp = jobs[idx]
        key = f"{sys_name}::{ds_key}::{i}"

        if key in cache:
            skipped += 1
            continue

        verdict, raw = call_judge(args.api_key, q, gold, resp, args.model)
        cache[key] = {"correct": verdict, "raw": raw}
        if verdict is None:
            errors += 1

        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(jobs)} done, errors={errors}")
            # checkpoint
            tmp = {"systems": existing.get("systems", {}),
                   "_judge_cache": cache,
                   "config": {"model": args.model, "provider": "modelark"},
                   "saved_at": datetime.now().isoformat()}
            with open(out_path, "w") as f:
                json.dump(tmp, f)
        time.sleep(0.1)

    # ---- Rebuild full results with new judge labels --------------------
    def get_judged(sys_name, ds_key, preds, key_by_row_dataset=False):
        out_preds = []
        for i, p in enumerate(preds):
            key_dataset = p["dataset"] if key_by_row_dataset else ds_key
            key = f"{sys_name}::{key_dataset}::{i}"
            entry = cache.get(key, {})
            new_p = dict(p)
            # Preserve original DeepSeek-V3 self-judge
            if "llm_judge" in new_p:
                new_p["llm_judge_deepseek_v3"] = new_p["llm_judge"]
            new_p["llm_judge"] = entry.get("correct")
            out_preds.append(new_p)
        return out_preds

    systems_out = {}

    # Released systems
    for sys_name, sys_data in released.items():
        s_out = {}
        for ds_key in ["locomo", "hotpotqa"]:
            ds = sys_data.get(ds_key, {})
            preds = ds.get("predictions", [])
            judged_preds = get_judged(sys_name, ds_key, preds)
            correct = sum(1 for p in judged_preds if p["llm_judge"] is True)
            n = sum(1 for p in judged_preds if p["llm_judge"] is not None)
            s_out[ds_key] = {
                "dataset": ds.get("dataset"),
                "technique": ds.get("technique", sys_name),
                "num_samples": ds.get("num_samples"),
                "num_questions": ds.get("num_questions", len(preds)),
                "overall": dict(ds.get("overall", {})),
                "judge_seed_2_0_pro": {
                    "correct": correct, "n": n,
                    "accuracy": correct / n if n else 0,
                },
                "predictions": judged_preds,
            }
        systems_out[sys_name] = s_out

    # No-memory
    nm_preds = get_judged(
        "no_memory", None, nomem.get("predictions", []), key_by_row_dataset=True
    )
    # no_memory preds carry dataset field per-row, so aggregate by that
    def agg_nm(ds_name):
        rows = [p for p in nm_preds if p.get("dataset") == ds_name]
        if not rows:
            return None
        correct = sum(1 for p in rows if p["llm_judge"] is True)
        n = sum(1 for p in rows if p["llm_judge"] is not None)
        return {"correct": correct, "n": n,
                "accuracy": correct / n if n else 0}
    systems_out["no_memory"] = {
        "locomo": {"judge_seed_2_0_pro": agg_nm("AgriConvMem"),
                   "predictions": [p for p in nm_preds if p.get("dataset") == "AgriConvMem"]},
        "hotpotqa": {"judge_seed_2_0_pro": agg_nm("AgriMultiHop"),
                     "predictions": [p for p in nm_preds if p.get("dataset") == "AgriMultiHop"]},
    }

    out = {
        "config": {"judge_model": args.model, "provider": "modelark",
                   "generator": "deepseek-v3 (deepseek/deepseek-chat-v3-0324)",
                   "judge_independent": True,
                   "saved_at": datetime.now().isoformat()},
        "systems": systems_out,
        "_judge_cache": cache,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- Print summary -------------------------------------------------
    print("\n" + "=" * 70)
    print("RE-JUDGE RESULTS (seed-2-0-pro, independent of generator)")
    print("=" * 70)
    print(f"{'System':<14} {'ConvMem Judge':<18} {'MultiHop Judge':<18}")
    print("-" * 70)
    for sys_name in ["nms", "rag", "hybrid", "bm25", "memorygraph", "no_memory"]:
        s = systems_out.get(sys_name, {})
        conv = s.get("locomo", {}).get("judge_seed_2_0_pro") or {}
        mult = s.get("hotpotqa", {}).get("judge_seed_2_0_pro") or {}
        cp = f"{conv.get('accuracy',0)*100:.1f}% ({conv.get('correct',0)}/{conv.get('n',0)})"
        mp = f"{mult.get('accuracy',0)*100:.1f}% ({mult.get('correct',0)}/{mult.get('n',0)})"
        print(f"{sys_name:<14} {cp:<18} {mp:<18}")
    print(f"\nErrors/unparseable: {errors}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
