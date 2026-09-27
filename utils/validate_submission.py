#!/usr/bin/env python3
"""
Validation script for Business Entity Resolution Challenge.
Validates formatting, schemas, candidate inclusion, and integrity of submission files.
"""

import os
import sys
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Validate submission TSVs for Business Entity Resolution Challenge")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    return parser.parse_args()


def load_entity_ids(file_path, id_col="id"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required test file not found: {file_path}")
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    # Automatically discover ID column if standard name differs
    if id_col not in df.columns:
        possible_cols = [c for c in df.columns if "id" in c.lower()]
        if possible_cols:
            id_col = possible_cols[0]
        else:
            id_col = df.columns[0]
    return set(df[id_col].dropna().astype(str).str.strip())


def parse_id_list(val):
    if pd.isna(val):
        return []
    val_str = str(val).strip()
    if not val_str or val_str in ("[]", "None", "nan"):
        return []
    # Support comma, semicolon, space, or bracketed separated strings
    cleaned = val_str.strip("[]")
    if not cleaned.strip():
        return []
    parts = [p.strip().strip("'\"") for p in cleaned.replace(";", ",").replace("\n", ",").split(",") if p.strip()]
    return parts


def validate_submission(matching_file, candidate_file, test_dir):
    print("=" * 60)
    print("STARTING SUBMISSION VALIDATION")
    print("=" * 60)
    print(f"Matching file:  {matching_file}")
    print(f"Candidate file: {candidate_file}")
    print(f"Test directory: {test_dir}")
    print("-" * 60)

    # 1. Check existence
    if not os.path.exists(matching_file):
        print(f"[FAIL] Missing matching file: {matching_file}")
        return False
    if not os.path.exists(candidate_file):
        print(f"[FAIL] Missing candidate file: {candidate_file}")
        return False

    # 2. Load ground test entity IDs
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")

    try:
        s1_ids = load_entity_ids(s1_file)
        s2_ids = load_entity_ids(s2_file)
        s3_ids = load_entity_ids(s3_file)
        valid_s2_s3_ids = s2_ids.union(s3_ids)
        print(f"[INFO] Loaded Test S1 IDs: {len(s1_ids)}, S2 IDs: {len(s2_ids)}, S3 IDs: {len(s3_ids)}")
    except Exception as e:
        print(f"[FAIL] Error loading test sources: {e}")
        return False

    # 3. Validate matching_results.tsv
    try:
        df_matching = pd.read_csv(matching_file, sep="\t", dtype=str)
    except Exception as e:
        print(f"[FAIL] Could not parse matching_results.tsv as TSV: {e}")
        return False

    expected_cols_m = ["source1_entity_id", "matched_entity_ids"]
    if list(df_matching.columns) != expected_cols_m:
        print(f"[FAIL] matching_results.tsv columns must be exactly {expected_cols_m}, found: {list(df_matching.columns)}")
        return False

    # Exactly one row per Source 1 test entity
    m_s1_list = df_matching["source1_entity_id"].dropna().astype(str).str.strip().tolist()
    if len(m_s1_list) != len(df_matching):
        print("[FAIL] matching_results.tsv contains empty or NaN source1_entity_id values")
        return False

    if len(m_s1_list) != len(set(m_s1_list)):
        print("[FAIL] matching_results.tsv contains duplicate source1_entity_id entries")
        return False

    m_s1_set = set(m_s1_list)
    if m_s1_set != s1_ids:
        missing = s1_ids - m_s1_set
        extra = m_s1_set - s1_ids
        if missing:
            print(f"[FAIL] matching_results.tsv is missing {len(missing)} Source 1 IDs (e.g. {list(missing)[:3]})")
        if extra:
            print(f"[FAIL] matching_results.tsv contains {len(extra)} unknown Source 1 IDs (e.g. {list(extra)[:3]})")
        return False

    # 4. Validate candidate_pairs.tsv
    try:
        df_cand = pd.read_csv(candidate_file, sep="\t", dtype=str)
    except Exception as e:
        print(f"[FAIL] Could not parse candidate_pairs.tsv as TSV: {e}")
        return False

    expected_cols_c = ["source1_entity_id", "candidate_entity_ids"]
    if list(df_cand.columns) != expected_cols_c:
        print(f"[FAIL] candidate_pairs.tsv columns must be exactly {expected_cols_c}, found: {list(df_cand.columns)}")
        return False

    c_s1_list = df_cand["source1_entity_id"].dropna().astype(str).str.strip().tolist()
    if len(c_s1_list) != len(set(c_s1_list)) or set(c_s1_list) != s1_ids:
        print("[FAIL] candidate_pairs.tsv does not have exactly one row per Source 1 test entity")
        return False

    # Build candidate lookup
    cand_lookup = {}
    for _, row in df_cand.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        cands = parse_id_list(row.get("candidate_entity_ids", ""))
        cand_lookup[s1] = set(cands)

    # 5. Check match integrity
    total_matches = 0
    singletons = 0
    multi_matches = 0

    for _, row in df_matching.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matches = parse_id_list(row.get("matched_entity_ids", ""))
        
        # Check duplicate matches
        if len(matches) != len(set(matches)):
            print(f"[FAIL] Duplicate matched entity IDs found for Source 1 entity: {s1}")
            return False

        # Check matched IDs exist in S2 or S3
        for m in matches:
            if m not in valid_s2_s3_ids:
                print(f"[FAIL] Matched entity ID {m} for {s1} does NOT exist in test_source2 or test_source3")
                return False
            
            # Every final match must exist in the candidate list
            if m not in cand_lookup.get(s1, set()):
                print(f"[FAIL] Matched entity ID {m} for {s1} is NOT in candidate_pairs.tsv candidate list!")
                return False

        if len(matches) == 0:
            singletons += 1
        elif len(matches) == 1:
            total_matches += 1
        else:
            total_matches += len(matches)
            multi_matches += 1

    print("[SUCCESS] All validation checks passed successfully!")
    print(f"  Total Source 1 Entities: {len(s1_ids)}")
    print(f"  Total Resolved Matches:  {total_matches}")
    print(f"  Singletons (0 matches):  {singletons} ({singletons / len(s1_ids):.1%})")
    print(f"  Multi-matches (>1 match):{multi_matches}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    args = parse_args()
    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
