"""
Multi-Pass High-Recall Candidate Generation / Blocking Module.
Generates candidate pairs (S1, S2) and (S1, S3) using union of multiple independent blocking strategies:
1. Exact normalized business name
2. First token / significant name token index
3. Character 3-prefix index
4. Significant address tokens
5. Postal / numeric tokens
6. TF-IDF character n-gram cosine retrieval for fuzzy name matching
All passes are merged via union and deduplicated.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Common non-distinctive words to avoid huge candidate explosion
STOP_TOKENS = {
    "and", "the", "of", "in", "at", "for", "on", "by", "to", "co", "corp",
    "inc", "ltd", "pvt", "llc", "llp", "gmbh", "sa", "serv", "tech", "soln"
}


class MultiPassBlocker:
    """
    High-recall multi-pass blocking engine for business entities.
    Indexes target records (Source 2 and Source 3) and queries Source 1 entities.
    """

    def __init__(self, top_k_fuzzy: int = 15, max_candidates_per_entity: int = 100):
        self.top_k_fuzzy = top_k_fuzzy
        self.max_candidates_per_entity = max_candidates_per_entity
        
        # Inverted index mappings: key -> set of target entity IDs
        self.exact_name_index = defaultdict(set)
        self.first_token_index = defaultdict(set)
        self.name_prefix3_index = defaultdict(set)
        self.significant_name_token_index = defaultdict(set)
        self.numeric_token_index = defaultdict(set)
        self.addr_token_index = defaultdict(set)
        
        self.target_records: Dict[str, Dict] = {}
        self.target_ids_list: List[str] = []
        self.target_names_list: List[str] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.target_tfidf_matrix = None

    def fit_targets(self, targets_df: pd.DataFrame):
        """Build inverted indices and TF-IDF representations over the target pool (S2 and S3)."""
        print(f"[BLOCKING] Indexing {len(targets_df)} target records (Source 2 & Source 3)...")
        self.target_records.clear()
        self.target_ids_list = targets_df["id"].tolist()
        self.target_names_list = targets_df["norm_name"].tolist()

        for _, row in targets_df.iterrows():
            eid = row["id"]
            norm_name = row["norm_name"]
            name_tokens = row["name_tokens"]
            norm_addr = row["norm_address"]
            addr_tokens = row["addr_tokens"]
            num_tokens = row["numeric_tokens"]
            norm_country = row["norm_country"]

            self.target_records[eid] = {
                "id": eid,
                "source": row["source"],
                "norm_name": norm_name,
                "name_tokens": name_tokens,
                "norm_addr": norm_addr,
                "addr_tokens": addr_tokens,
                "numeric_tokens": num_tokens,
                "norm_country": norm_country,
            }

            # 1. Exact name block
            if norm_name:
                self.exact_name_index[norm_name].add(eid)

            # 2. First significant token
            sig_name_tokens = [t for t in name_tokens if t not in STOP_TOKENS and len(t) >= 2]
            if sig_name_tokens:
                self.first_token_index[sig_name_tokens[0]].add(eid)
                # 3. 3-char prefix of the primary token
                if len(sig_name_tokens[0]) >= 3:
                    self.name_prefix3_index[sig_name_tokens[0][:3]].add(eid)

            # 4. Significant name token index (tokens with length >= 4)
            for tok in sig_name_tokens:
                if len(tok) >= 4:
                    self.significant_name_token_index[tok].add(eid)

            # 5. Numeric / PIN-like tokens (postal codes, building numbers)
            for num in num_tokens:
                if len(num) >= 3:
                    self.numeric_token_index[num].add(eid)

            # 6. Address significant tokens (street name, locality)
            sig_addr_tokens = [t for t in addr_tokens if len(t) >= 4 and t not in STOP_TOKENS]
            for tok in sig_addr_tokens[:3]:  # Limit to top tokens to maintain speed
                self.addr_token_index[tok].add(eid)

        # 7. TF-IDF character n-gram matrix for fuzzy name matching
        try:
            non_empty_names = [n if n else "unknown" for n in self.target_names_list]
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
                max_features=25000,
                dtype=np.float32
            )
            self.target_tfidf_matrix = self.tfidf_vectorizer.fit_transform(non_empty_names)
        except Exception as e:
            print(f"[BLOCKING WARNING] Could not build TF-IDF matrix: {e}")
            self.tfidf_vectorizer = None

        print("[BLOCKING] Inverted index construction complete.")

    def block_entity(self, s1_row: pd.Series) -> Set[str]:
        """
        Generate candidate target IDs for a single Source 1 entity via union of all blocking passes.
        """
        candidates: Set[str] = set()
        s1_name = s1_row["norm_name"]
        s1_tokens = s1_row["name_tokens"]
        s1_num_tokens = s1_row["numeric_tokens"]
        s1_addr_tokens = s1_row["addr_tokens"]
        s1_country = s1_row["norm_country"]

        # Pass 1: Exact normalized name
        if s1_name and s1_name in self.exact_name_index:
            candidates.update(self.exact_name_index[s1_name])

        # Pass 2: First significant token
        sig_name_tokens = [t for t in s1_tokens if t not in STOP_TOKENS and len(t) >= 2]
        if sig_name_tokens:
            first_tok = sig_name_tokens[0]
            if first_tok in self.first_token_index:
                # Add if not excessively dense
                pool = self.first_token_index[first_tok]
                if len(pool) <= 200:
                    candidates.update(pool)
                else:
                    # Sample or use prefix
                    candidates.update(list(pool)[:50])

            # Pass 3: 3-char prefix
            if len(first_tok) >= 3:
                p3 = first_tok[:3]
                if p3 in self.name_prefix3_index:
                    pool = self.name_prefix3_index[p3]
                    if len(pool) <= 150:
                        candidates.update(pool)

        # Pass 4: Significant token overlap
        for tok in sig_name_tokens:
            if len(tok) >= 4 and tok in self.significant_name_token_index:
                pool = self.significant_name_token_index[tok]
                if len(pool) <= 100:
                    candidates.update(pool)

        # Pass 5: Postal / numeric token block (e.g. same pin code or building number)
        for num in s1_num_tokens:
            if len(num) >= 3 and num in self.numeric_token_index:
                pool = self.numeric_token_index[num]
                # Combine numeric match with country agreement or name overlap
                if len(pool) <= 100:
                    candidates.update(pool)

        # Pass 6: Address significant tokens
        sig_addr_tokens = [t for t in s1_addr_tokens if len(t) >= 4 and t not in STOP_TOKENS]
        for tok in sig_addr_tokens[:2]:
            if tok in self.addr_token_index:
                pool = self.addr_token_index[tok]
                if len(pool) <= 60:
                    candidates.update(pool)

        # Country filter / compatibility:
        # Never reject if either country is unknown/missing
        if s1_country:
            # We don't discard blindly, but filter out obvious cross-country conflicts
            # if both have well-defined non-empty differing countries
            filtered_cands = set()
            for cid in candidates:
                cand_info = self.target_records.get(cid)
                if cand_info:
                    cand_country = cand_info["norm_country"]
                    if not cand_country or cand_country == s1_country:
                        filtered_cands.add(cid)
                    else:
                        # Obvious mismatch (e.g. US vs India or France vs India)
                        # Keep only if exact name match
                        if cand_info["norm_name"] == s1_name:
                            filtered_cands.add(cid)
            candidates = filtered_cands

        return candidates

    def generate_candidate_pairs(self, s1_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
        """
        Generates candidate pairs for all Source 1 entities.
        Returns:
            candidate_map: {s1_id: [candidate_target_ids]}
            candidate_pairs: [(s1_id, target_id)]
        """
        print(f"[BLOCKING] Generating candidates for {len(s1_df)} Source 1 entities...")
        candidate_map: Dict[str, List[str]] = {}
        all_pairs: List[Tuple[str, str]] = []

        # Vectorized TF-IDF batch query for fuzzy name recall
        tfidf_cands_by_idx = defaultdict(set)
        if self.tfidf_vectorizer is not None and self.target_tfidf_matrix is not None:
            try:
                s1_names = [n if n else "unknown" for n in s1_df["norm_name"]]
                s1_tfidf = self.tfidf_vectorizer.transform(s1_names)
                # Compute cosine similarities in chunks
                batch_size = 500
                for start_idx in range(0, s1_df.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, s1_df.shape[0])
                    sim_matrix = s1_tfidf[start_idx:end_idx].dot(self.target_tfidf_matrix.T).toarray()
                    for local_i, global_i in enumerate(range(start_idx, end_idx)):
                        scores = sim_matrix[local_i]
                        # Top-K candidates above threshold 0.35
                        top_indices = np.argsort(scores)[-self.top_k_fuzzy:][::-1]
                        for tidx in top_indices:
                            if scores[tidx] >= 0.35:
                                tfidf_cands_by_idx[global_i].add(self.target_ids_list[tidx])
            except Exception as e:
                print(f"[BLOCKING WARNING] Fuzzy TF-IDF batch query encountered error: {e}")

        # Merge inverted index candidates + TF-IDF candidates
        for idx, (_, s1_row) in enumerate(s1_df.iterrows()):
            s1_id = s1_row["id"]
            cands = self.block_entity(s1_row)
            
            # Add fuzzy TF-IDF candidates
            if idx in tfidf_cands_by_idx:
                cands.update(tfidf_cands_by_idx[idx])

            # Deduplicate and cap
            sorted_cands = sorted(list(cands))
            if len(sorted_cands) > self.max_candidates_per_entity:
                sorted_cands = sorted_cands[:self.max_candidates_per_entity]

            candidate_map[s1_id] = sorted_cands
            for cid in sorted_cands:
                all_pairs.append((s1_id, cid))

        print(f"[BLOCKING] Generated {len(all_pairs)} total candidate pairs across {len(s1_df)} S1 entities.")
        avg_cands = len(all_pairs) / max(len(s1_df), 1)
        print(f"[BLOCKING] Average candidates per S1 entity: {avg_cands:.2f}")

        return candidate_map, all_pairs
