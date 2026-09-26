"""
Feature Engineering Module for Business Entity Resolution.
Computes rich similarity metrics across Name, Address, Country, and Source metadata.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False


def levenshtein_sim(s1: str, s2: str) -> float:
    """Computes normalized edit similarity between two strings in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2) / 100.0
    else:
        return difflib.SequenceMatcher(None, s1, s2).ratio()


def token_overlap_metrics(tokens1: List[str], tokens2: List[str]) -> Tuple[float, float, int]:
    """Computes Jaccard similarity, containment overlap, and common token count."""
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 1.0, 1.0, 0
    if not set1 or not set2:
        return 0.0, 0.0, 0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard = len(intersection) / len(union) if union else 0.0
    min_len = min(len(set1), len(set2))
    containment = len(intersection) / min_len if min_len > 0 else 0.0
    return jaccard, containment, len(intersection)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    ngrams1 = {s1[i:i+n] for i in range(len(s1) - n + 1)}
    ngrams2 = {s2[i:i+n] for i in range(len(s2) - n + 1)}
    union = ngrams1.union(ngrams2)
    if not union:
        return 0.0
    return len(ngrams1.intersection(ngrams2)) / len(union)


def numeric_overlap_sim(nums1: List[str], nums2: List[str]) -> Tuple[float, int]:
    """Computes overlap of numeric/postal tokens between two addresses."""
    set1 = set(nums1)
    set2 = set(nums2)
    if not set1 and not set2:
        return 0.5, 0  # Neutral indicator when neither has numeric tokens
    if not set1 or not set2:
        return 0.0, 0
    inter = set1.intersection(set2)
    overlap = len(inter) / min(len(set1), len(set2))
    return overlap, len(inter)


def compute_pair_features(
    s1_record: Dict[str, Any],
    cand_record: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes a complete feature vector for a single (Source 1, Target Candidate) pair.
    """
    s1_name = s1_record.get("norm_name", "")
    c_name = cand_record.get("norm_name", "")
    s1_name_tokens = s1_record.get("name_tokens", [])
    c_name_tokens = cand_record.get("name_tokens", [])
    
    s1_addr = s1_record.get("norm_address", "")
    c_addr = cand_record.get("norm_address", "")
    s1_addr_tokens = s1_record.get("addr_tokens", [])
    c_addr_tokens = cand_record.get("addr_tokens", [])
    s1_num_tokens = s1_record.get("numeric_tokens", [])
    c_num_tokens = cand_record.get("numeric_tokens", [])

    s1_country = s1_record.get("norm_country", "")
    c_country = cand_record.get("norm_country", "")
    target_source = cand_record.get("source", "")

    # 1. NAME FEATURES
    name_exact = 1.0 if s1_name and s1_name == c_name else 0.0
    name_token_exact = 1.0 if s1_name_tokens and s1_name_tokens == c_name_tokens else 0.0
    name_jaccard, name_overlap, name_common_tokens = token_overlap_metrics(s1_name_tokens, c_name_tokens)
    name_lev = levenshtein_sim(s1_name, c_name)
    name_ngram2 = char_ngram_jaccard(s1_name, c_name, n=2)
    name_ngram3 = char_ngram_jaccard(s1_name, c_name, n=3)
    name_len_diff = abs(len(s1_name) - len(c_name))
    name_len_ratio = min(len(s1_name), len(c_name)) / max(len(s1_name), len(c_name), 1)

    # 2. ADDRESS FEATURES
    addr_exact = 1.0 if s1_addr and s1_addr == c_addr else 0.0
    addr_jaccard, addr_overlap, addr_common_tokens = token_overlap_metrics(s1_addr_tokens, c_addr_tokens)
    addr_lev = levenshtein_sim(s1_addr, c_addr)
    addr_ngram3 = char_ngram_jaccard(s1_addr, c_addr, n=3)
    addr_len_diff = abs(len(s1_addr) - len(c_addr))
    num_overlap, num_common_count = numeric_overlap_sim(s1_num_tokens, c_num_tokens)

    # 3. COUNTRY FEATURES
    if s1_country and c_country:
        country_match = 1.0 if s1_country == c_country else 0.0
        country_conflict = 1.0 if s1_country != c_country else 0.0
    else:
        country_match = 0.5  # Unknown / compatible
        country_conflict = 0.0

    # 4. MISSING VALUE INDICATORS
    missing_s1_name = 1.0 if not s1_name else 0.0
    missing_c_name = 1.0 if not c_name else 0.0
    missing_s1_addr = 1.0 if not s1_addr else 0.0
    missing_c_addr = 1.0 if not c_addr else 0.0
    missing_s1_country = 1.0 if not s1_country else 0.0
    missing_c_country = 1.0 if not c_country else 0.0

    # 5. SOURCE INDICATOR
    is_source2 = 1.0 if "2" in str(target_source) else 0.0
    is_source3 = 1.0 if "3" in str(target_source) else 0.0

    # 6. COMBINED / INTERACTION FEATURES
    # Weighted composite similarity
    composite_sim = (0.55 * name_lev) + (0.35 * addr_lev) + (0.10 * country_match)
    min_sim = min(name_lev, addr_lev)
    max_sim = max(name_lev, addr_lev)
    harmonic_sim = 2 * (name_lev * addr_lev) / (name_lev + addr_lev + 1e-6)

    return {
        "name_exact": name_exact,
        "name_token_exact": name_token_exact,
        "name_jaccard": name_jaccard,
        "name_overlap": name_overlap,
        "name_common_tokens": float(name_common_tokens),
        "name_lev": name_lev,
        "name_ngram2": name_ngram2,
        "name_ngram3": name_ngram3,
        "name_len_diff": float(name_len_diff),
        "name_len_ratio": name_len_ratio,
        "addr_exact": addr_exact,
        "addr_jaccard": addr_jaccard,
        "addr_overlap": addr_overlap,
        "addr_common_tokens": float(addr_common_tokens),
        "addr_lev": addr_lev,
        "addr_ngram3": addr_ngram3,
        "addr_len_diff": float(addr_len_diff),
        "num_overlap": num_overlap,
        "num_common_count": float(num_common_count),
        "country_match": country_match,
        "country_conflict": country_conflict,
        "missing_s1_addr": missing_s1_addr,
        "missing_c_addr": missing_c_addr,
        "is_source2": is_source2,
        "is_source3": is_source3,
        "composite_sim": composite_sim,
        "min_sim": min_sim,
        "max_sim": max_sim,
        "harmonic_sim": harmonic_sim,
    }


def extract_features_for_pairs(
    pairs: List[Tuple[str, str]],
    s1_lookup: Dict[str, Dict[str, Any]],
    target_lookup: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Batch feature extraction for a list of candidate pairs (s1_id, target_id).
    Returns DataFrame with columns ['s1_id', 'target_id'] + engineered feature columns.
    """
    rows = []
    for s1_id, target_id in pairs:
        s1_rec = s1_lookup.get(s1_id)
        t_rec = target_lookup.get(target_id)
        if s1_rec is None or t_rec is None:
            continue
        feats = compute_pair_features(s1_rec, t_rec)
        feats["s1_id"] = s1_id
        feats["target_id"] = target_id
        rows.append(feats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
