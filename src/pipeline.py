"""
End-to-End Pipeline for Business Entity Resolution Challenge.
Loads datasets, performs normalization, multi-pass candidate blocking,
extracts rich similarity features, trains tabular gradient boosting model,
optimizes decision threshold for F0.5, generates final submission TSVs,
and validates the submission with utils/validate_submission.py.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import argparse
import subprocess
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_dataframe
from src.blocking import MultiPassBlocker
from src.features import extract_features_for_pairs
from src.model import EntityResolutionClassifier
from src.evaluation import optimize_threshold, evaluate_entity_level, evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Business Entity Resolution End-to-End Pipeline")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory containing train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    parser.add_argument("--output-dir", default="output", help="Directory where matching_results.tsv and candidate_pairs.tsv will be saved")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running utils/validate_submission.py")
    return parser.parse_args()


def load_tsv(filepath: str) -> pd.DataFrame:
    """Loads TSV file with proper tab delimiter."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, sep="\t", dtype=str)


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Parses ground truth into {s1_id: set(matched_target_ids)}.
    Supports both list-style ('matched_entity_ids') and pairwise ('source1_id', 'target_id') formats.
    """
    gt_map = {}
    cols = [c.lower() for c in gt_df.columns]
    
    if "matched_entity_ids" in cols or "matched_ids" in cols:
        s1_col = gt_df.columns[0]
        match_col = gt_df.columns[1]
        for _, row in gt_df.iterrows():
            s1 = str(row[s1_col]).strip()
            raw_val = str(row[match_col]).strip()
            if not raw_val or raw_val in ("[]", "None", "nan"):
                gt_map[s1] = set()
            else:
                cleaned = raw_val.strip("[]")
                parts = [p.strip().strip("'\"") for p in cleaned.replace(";", ",").split(",") if p.strip()]
                gt_map[s1] = set(parts)
    else:
        # Pairwise format: col 0 is s1, col 1 is target
        s1_col = gt_df.columns[0]
        tgt_col = gt_df.columns[1]
        for _, row in gt_df.iterrows():
            s1 = str(row[s1_col]).strip()
            tgt = str(row[tgt_col]).strip()
            if s1 not in gt_map:
                gt_map[s1] = set()
            if tgt and tgt != "nan":
                gt_map[s1].add(tgt)
                
    return gt_map


def format_id_list(id_list: List[str]) -> str:
    """Format list of IDs into clean comma-separated bracketed string."""
    clean_ids = [str(x).strip() for x in id_list if str(x).strip()]
    return "[" + ", ".join(clean_ids) + "]"


def run_pipeline(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    random_seed: int = 42,
    skip_validation: bool = False
) -> Dict[str, Any]:
    """Runs the complete production pipeline."""
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION - PRODUCTION PIPELINE")
    print("=" * 70)
    print(f"Train directory:  {train_dir}")
    print(f"Test directory:   {test_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed:      {random_seed}")
    print("-" * 70)

    # 1. LOAD TRAINING DATA
    print("\n[STEP 1/7] Loading and preprocessing training datasets...")
    train_s1_raw = load_tsv(os.path.join(train_dir, "train_source1.tsv"))
    train_s2_raw = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    train_s3_raw = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    train_gt_raw = load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))

    gt_map = parse_ground_truth(train_gt_raw)

    train_s1 = preprocess_dataframe(train_s1_raw, "source1")
    train_s2 = preprocess_dataframe(train_s2_raw, "source2")
    train_s3 = preprocess_dataframe(train_s3_raw, "source3")
    train_targets = pd.concat([train_s2, train_s3], ignore_index=True)

    print(f"  Source 1 Train Entities: {len(train_s1)}")
    print(f"  Source 2 Train Records:  {len(train_s2)}")
    print(f"  Source 3 Train Records:  {len(train_s3)}")
    print(f"  Combined Target Records: {len(train_targets)}")
    print(f"  Ground Truth Mappings:   {len(gt_map)}")

    # 2. ENTITY-LEVEL TRAIN / VALIDATION SPLIT
    print("\n[STEP 2/7] Splitting entities into Train (80%) and Validation (20%)...")
    all_s1_ids = train_s1["id"].unique()
    train_ids, val_ids = train_test_split(all_s1_ids, test_size=0.20, random_state=random_seed)
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    df_s1_train = train_s1[train_s1["id"].isin(train_ids_set)].reset_index(drop=True)
    df_s1_val = train_s1[train_s1["id"].isin(val_ids_set)].reset_index(drop=True)
    print(f"  S1 Train Split: {len(df_s1_train)} entities")
    print(f"  S1 Val Split:   {len(df_s1_val)} entities (zero entity-level leakage)")

    # 3. BLOCKING & CANDIDATE GENERATION (TRAIN POOL)
    print("\n[STEP 3/7] Indexing target records and generating blocking candidates...")
    blocker = MultiPassBlocker(top_k_fuzzy=15, max_candidates_per_entity=80)
    blocker.fit_targets(train_targets)

    cand_map_train, pairs_train = blocker.generate_candidate_pairs(df_s1_train)
    cand_map_val, pairs_val = blocker.generate_candidate_pairs(df_s1_val)

    # Candidate recall check on validation
    val_true_pairs_count = sum(len(gt_map.get(s1, set())) for s1 in val_ids_set)
    val_retrieved_true_pairs = 0
    for s1 in val_ids_set:
        cands_set = set(cand_map_val.get(s1, []))
        true_set = gt_map.get(s1, set())
        val_retrieved_true_pairs += len(cands_set.intersection(true_set))

    val_cand_recall = val_retrieved_true_pairs / max(val_true_pairs_count, 1)
    print(f"  Validation Candidate Blocking Recall: {val_cand_recall:.2%} ({val_retrieved_true_pairs}/{val_true_pairs_count})")

    # Add ground truth positive pairs into train set to guarantee positive representation
    gt_pairs_train = []
    for s1 in train_ids_set:
        for tgt in gt_map.get(s1, set()):
            gt_pairs_train.append((s1, tgt))
    
    combined_train_pairs = list(set(pairs_train).union(set(gt_pairs_train)))
    print(f"  Total Candidate Pairs for Model Training: {len(combined_train_pairs)}")
    print(f"  Total Candidate Pairs for Validation:     {len(pairs_val)}")

    # 4. FEATURE EXTRACTION
    print("\n[STEP 4/7] Extracting rich similarity features for training and validation pairs...")
    s1_lookup = {r["id"]: r for _, r in train_s1.iterrows()}
    target_lookup = {r["id"]: r for _, r in train_targets.iterrows()}

    X_train_df = extract_features_for_pairs(combined_train_pairs, s1_lookup, target_lookup)
    y_train = np.array([1 if p[1] in gt_map.get(p[0], set()) else 0 for p in zip(X_train_df["s1_id"], X_train_df["target_id"])])

    X_val_df = extract_features_for_pairs(pairs_val, s1_lookup, target_lookup)
    y_val = np.array([1 if p[1] in gt_map.get(p[0], set()) else 0 for p in zip(X_val_df["s1_id"], X_val_df["target_id"])])

    print(f"  Train Set Shape: {X_train_df.shape}, Positives: {int(np.sum(y_train))}, Negatives: {int(np.sum(y_train == 0))}")
    print(f"  Val Set Shape:   {X_val_df.shape}, Positives: {int(np.sum(y_val))}, Negatives: {int(np.sum(y_val == 0))}")

    # 5. MODEL TRAINING & THRESHOLD OPTIMIZATION FOR F0.5
    print("\n[STEP 5/7] Training tabular classification model & optimizing F0.5 threshold...")
    clf = EntityResolutionClassifier(random_state=random_seed)
    clf.fit(X_train_df, y_train, eval_set=(X_val_df, y_val))

    # Feature importance
    importances = clf.get_feature_importances()
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 Important Features: {top_feats}")

    # Predict validation probabilities
    val_probs = clf.predict_proba(X_val_df)

    # Threshold optimization for F0.5
    best_th, best_metrics, threshold_curve = optimize_threshold(y_val, val_probs)
    print(f"\n  [THRESHOLD SEARCH] Optimal Decision Threshold: {best_th:.3f}")
    print(f"  Validation Precision: {best_metrics['precision']:.4f}")
    print(f"  Validation Recall:    {best_metrics['recall']:.4f}")
    print(f"  Validation F0.5:      {best_metrics['f05']:.4f} (Precision-weighted)")

    # Entity-level validation
    entity_val_metrics = evaluate_entity_level(val_ids_set, gt_map, X_val_df, val_probs, best_th)
    print(f"  Validation Singleton Accuracy: {entity_val_metrics['singleton_accuracy']:.2%}")
    print(f"  Validation Exact Entity Match: {entity_val_metrics['exact_entity_accuracy']:.2%}")

    # Retrain on full training set (train + val) for maximum test accuracy
    print("\n  Retraining model on full train dataset for test inference...")
    full_train_pairs = combined_train_pairs + pairs_val
    full_X = pd.concat([X_train_df, X_val_df], ignore_index=True)
    full_y = np.concatenate([y_train, y_val])
    clf.fit(full_X, full_y)

    # 6. INFERENCE ON TEST DATA
    print("\n[STEP 6/7] Running candidate generation and inference on Test data...")
    test_s1_raw = load_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s2_raw = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s3_raw = load_tsv(os.path.join(test_dir, "test_source3.tsv"))

    test_s1 = preprocess_dataframe(test_s1_raw, "source1")
    test_s2 = preprocess_dataframe(test_s2_raw, "source2")
    test_s3 = preprocess_dataframe(test_s3_raw, "source3")
    test_targets = pd.concat([test_s2, test_s3], ignore_index=True)

    print(f"  Test Source 1 Entities: {len(test_s1)}")
    print(f"  Test Target Records:    {len(test_targets)}")

    test_blocker = MultiPassBlocker(top_k_fuzzy=15, max_candidates_per_entity=80)
    test_blocker.fit_targets(test_targets)

    test_cand_map, test_pairs = test_blocker.generate_candidate_pairs(test_s1)
    print(f"  Test Candidate Pairs: {len(test_pairs)}")

    # Extract test features
    test_s1_lookup = {r["id"]: r for _, r in test_s1.iterrows()}
    test_tgt_lookup = {r["id"]: r for _, r in test_targets.iterrows()}

    test_X_df = extract_features_for_pairs(test_pairs, test_s1_lookup, test_tgt_lookup)
    if len(test_X_df) > 0:
        test_probs = clf.predict_proba(test_X_df)
        test_X_df["prob"] = test_probs
        test_X_df["pred_match"] = test_X_df["prob"] >= best_th
    else:
        test_X_df["prob"] = []
        test_X_df["pred_match"] = []

    # Assemble matching results and candidate pairs
    matching_map: Dict[str, List[str]] = {s1: [] for s1 in test_s1["id"]}
    candidate_map: Dict[str, List[str]] = {s1: test_cand_map.get(s1, []) for s1 in test_s1["id"]}

    if len(test_X_df) > 0:
        for _, row in test_X_df[test_X_df["pred_match"]].iterrows():
            s1 = row["s1_id"]
            tgt = row["target_id"]
            matching_map[s1].append(tgt)

    # Format outputs
    # 1. output/matching_results.tsv
    matching_rows = []
    for s1_id in test_s1["id"]:
        matched_ids = sorted(list(set(matching_map.get(s1_id, []))))
        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": format_id_list(matched_ids)
        })
    df_matching_out = pd.DataFrame(matching_rows)
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    df_matching_out.to_csv(matching_out_path, sep="\t", index=False)
    print(f"  Saved: {matching_out_path}")

    # 2. output/candidate_pairs.tsv
    candidate_rows = []
    for s1_id in test_s1["id"]:
        cand_ids = sorted(list(set(candidate_map.get(s1_id, []))))
        # Ensure any matched ID is strictly in candidate list
        matched_ids = set(matching_map.get(s1_id, []))
        merged_cands = sorted(list(set(cand_ids).union(matched_ids)))
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": format_id_list(merged_cands)
        })
    df_candidate_out = pd.DataFrame(candidate_rows)
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    df_candidate_out.to_csv(candidate_out_path, sep="\t", index=False)
    print(f"  Saved: {candidate_out_path}")

    # Save experiment report
    report_data = {
        "val_candidate_recall": float(val_cand_recall),
        "best_threshold": float(best_th),
        "val_precision": float(best_metrics["precision"]),
        "val_recall": float(best_metrics["recall"]),
        "val_f05": float(best_metrics["f05"]),
        "val_singleton_accuracy": float(entity_val_metrics["singleton_accuracy"]),
        "test_s1_count": len(test_s1),
        "test_resolved_matches": sum(len(v) for v in matching_map.values()),
        "test_singletons": sum(1 for v in matching_map.values() if len(v) == 0),
        "top_features": top_feats,
        "threshold_curve": threshold_curve[:20]  # Sample
    }
    report_path = os.path.join(output_dir, "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # 7. AUTOMATIC VALIDATION SCRIPT
    print("\n[STEP 7/7] Running submission validation script...")
    val_script = os.path.join("utils", "validate_submission.py")
    validation_passed = False
    if not skip_validation and os.path.exists(val_script):
        cmd = [
            sys.executable,
            val_script,
            "--matching", matching_out_path,
            "--candidate", candidate_out_path,
            "--test-dir", test_dir
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("[VALIDATION PASSED] Submission files conform strictly to challenge rules.")
            validation_passed = True
        else:
            print(f"[VALIDATION FAILED] Errors:\n{res.stderr}")
    else:
        print("[INFO] Skipped validation script.")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return {
        "validation_passed": validation_passed,
        "metrics": best_metrics,
        "entity_metrics": entity_val_metrics,
        "candidate_recall": val_cand_recall,
        "matching_path": matching_out_path,
        "candidate_path": candidate_out_path
    }


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        skip_validation=args.skip_validation
    )
