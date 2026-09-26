"""
Data Preprocessing and Normalization Module for Business Entity Resolution.
Implements robust, multi-country entity normalization for business names, addresses, and country identifiers.
Preserves original values alongside normalized representations.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Dict, Any, List, Set, Tuple, Optional

import pandas as pd


# Business suffixes and abbreviations
BUSINESS_SUFFIXES = {
    r"\bcorporation\b": "corp",
    r"\bcorp\.?\b": "corp",
    r"\bincorporated\b": "inc",
    r"\binc\.?\b": "inc",
    r"\blimited\b": "ltd",
    r"\bltd\.?\b": "ltd",
    r"\bprivate\b": "pvt",
    r"\bpvt\.?\b": "pvt",
    r"\bpte\.?\b": "pvt",
    r"\bllc\.?\b": "llc",
    r"\bl\.l\.c\.?\b": "llc",
    r"\bllp\.?\b": "llp",
    r"\bl\.l\.p\.?\b": "llp",
    r"\bcompany\b": "co",
    r"\bco\.?\b": "co",
    r"\bgmbh\b": "gmbh",
    r"\bsa\b": "sa",
    r"\bs\.a\.?\b": "sa",
    r"\bsarl\b": "sarl",
    r"\bs\.a\.r\.l\.?\b": "sarl",
    r"\bbv\b": "bv",
    r"\bb\.v\.?\b": "bv",
    r"\bplc\b": "plc",
    r"\benterprises\b": "ent",
    r"\btechnologies\b": "tech",
    r"\btechnology\b": "tech",
    r"\btech\.?\b": "tech",
    r"\bsolutions\b": "soln",
    r"\bsolution\b": "soln",
    r"\bservices\b": "serv",
    r"\bservice\b": "serv",
    r"\binternational\b": "intl",
    r"\bintl\.?\b": "intl",
    r"\bgroup\b": "grp",
    r"\bholdings\b": "hldg",
}

# Common address token abbreviations
ADDRESS_ABBREVIATIONS = {
    r"\bstreet\b": "st",
    r"\bst\.?\b": "st",
    r"\broad\b": "rd",
    r"\brd\.?\b": "rd",
    r"\bavenue\b": "ave",
    r"\bave\.?\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bblvd\.?\b": "blvd",
    r"\bdrive\b": "dr",
    r"\bdr\.?\b": "dr",
    r"\blane\b": "ln",
    r"\bln\.?\b": "ln",
    r"\bfloor\b": "fl",
    r"\bflr\.?\b": "fl",
    r"\bfl\.?\b": "fl",
    r"\bsuite\b": "ste",
    r"\bste\.?\b": "ste",
    r"\bapartment\b": "apt",
    r"\bapt\.?\b": "apt",
    r"\bbuilding\b": "bldg",
    r"\bbldg\.?\b": "bldg",
    r"\bhighway\b": "hwy",
    r"\bhwy\.?\b": "hwy",
    r"\bparkway\b": "pkwy",
    r"\bpkwy\.?\b": "pkwy",
    r"\bcourt\b": "ct",
    r"\bct\.?\b": "ct",
    r"\bplace\b": "pl",
    r"\bpl\.?\b": "pl",
    r"\bsquare\b": "sq",
    r"\bsq\.?\b": "sq",
    r"\bcircle\b": "cir",
    r"\bcir\.?\b": "cir",
    r"\bnumber\b": "no",
    r"\bnum\.?\b": "no",
    r"\bno\.?\b": "no",
    r"\bp\.?\s*o\.?\s*box\b": "pobox",
    r"\bpost\s+office\s+box\b": "pobox",
}


def unicode_normalize(text: str) -> str:
    """Normalize Unicode characters (NFKD) and convert accented characters to ASCII equivalents."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    # Strip combining diacritical marks
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def is_missing(val: Any) -> bool:
    if val is None:
        return True
    if pd is not None:
        return pd.isna(val)
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null")


def normalize_business_name(name: Optional[str]) -> Tuple[str, List[str]]:
    """
    Normalizes a business name:
    - Unicode normalization
    - Lowercase
    - Replace '&' with 'and'
    - Replace business suffixes/abbreviations
    - Strip punctuation while keeping alphanumeric and spaces
    - Collapse whitespace
    Returns (normalized_name, token_list).
    """
    if is_missing(name):
        return "", []
    
    text = unicode_normalize(str(name)).lower()
    
    # Normalize ampersand
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"\s*\+\s*", " and ", text)
    
    # Standardize business abbreviations and legal forms
    for pattern, replacement in BUSINESS_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
        
    # Remove unwanted punctuation (preserve alphanumeric, whitespace, hyphens)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"[-_]", " ", text)
    
    # Collapse whitespace
    tokens = [t for t in text.split() if t]
    normalized_name = " ".join(tokens)
    
    return normalized_name, tokens


def extract_numeric_tokens(text: str) -> List[str]:
    """Extract numeric sequences (like street numbers, zip codes, unit numbers)."""
    if not text:
        return []
    return re.findall(r"\b\d+\b", str(text))


def normalize_address(address: Optional[str]) -> Tuple[str, List[str], List[str]]:
    """
    Normalizes a business address:
    - Unicode normalization
    - Lowercase
    - Standardize common address abbreviations (st, rd, ave, fl, ste, etc.)
    - Clean punctuation
    - Extract numeric and postal-like tokens
    Returns (normalized_address, tokens, numeric_tokens).
    """
    if is_missing(address):
        return "", [], []
    
    raw = str(address)
    text = unicode_normalize(raw).lower()
    
    # Extract numerics before punctuation stripping
    numeric_tokens = extract_numeric_tokens(text)
    
    # Address abbreviations
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
        
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Normalize whitespace
    tokens = [t for t in text.split() if t]
    normalized_addr = " ".join(tokens)
    
    return normalized_addr, tokens, numeric_tokens


def normalize_country(country: Optional[str]) -> str:
    """
    Normalizes country strings without hardcoding fixed country subsets.
    Supports unseen countries (e.g. France, Germany, Japan, India, US, etc.).
    """
    if is_missing(country):
        return ""
    text = unicode_normalize(str(country)).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    
    # Common variations of major country names for high fidelity matching
    country_alias = {
        "usa": "us",
        "united states": "us",
        "united states of america": "us",
        "u s a": "us",
        "u s": "us",
        "ind": "india",
        "in": "india",
        "republic of india": "india",
        "uk": "united kingdom",
        "u k": "united kingdom",
        "great britain": "united kingdom",
        "fra": "france",
        "fr": "france",
        "deu": "germany",
        "de": "germany",
        "can": "canada",
        "ca": "canada",
        "aus": "australia",
        "au": "australia",
    }
    return country_alias.get(text, text)


def discover_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically discovers standard column names in a dataframe.
    Looks for entity ID, business name, address, and country.
    """
    col_map = {}
    cols = {c.lower().strip(): c for c in df.columns}
    
    # 1. ID column
    for candidate in ["id", "entity_id", "record_id", "business_id", "source_id", "source1_id", "source2_id", "source3_id"]:
        if candidate in cols:
            col_map["id"] = cols[candidate]
            break
    if "id" not in col_map:
        # Fallback: first column containing 'id'
        for col, original in cols.items():
            if "id" in col:
                col_map["id"] = original
                break
        if "id" not in col_map:
            col_map["id"] = df.columns[0]
            
    # 2. Business Name
    for candidate in ["name", "business_name", "company_name", "entity_name", "title", "legal_name"]:
        if candidate in cols:
            col_map["name"] = cols[candidate]
            break
    if "name" not in col_map:
        for col, original in cols.items():
            if "name" in col:
                col_map["name"] = original
                break
                
    # 3. Business Address
    for candidate in ["address", "business_address", "street_address", "location", "full_address", "addr"]:
        if candidate in cols:
            col_map["address"] = cols[candidate]
            break
    if "address" not in col_map:
        for col, original in cols.items():
            if "addr" in col or "street" in col:
                col_map["address"] = original
                break
                
    # 4. Country
    for candidate in ["country", "country_code", "nation", "country_name"]:
        if candidate in cols:
            col_map["country"] = cols[candidate]
            break
    if "country" not in col_map:
        for col, original in cols.items():
            if "country" in col or "nation" in col:
                col_map["country"] = original
                break

    return col_map


def preprocess_dataframe(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Preprocesses a raw business source dataframe:
    - Discovers column names
    - Normalizes name, address, country
    - Extracts token sets and numeric tokens
    - Preserves original raw columns
    - Adds source label
    """
    df = df.copy()
    col_map = discover_columns(df)
    
    id_col = col_map.get("id")
    name_col = col_map.get("name")
    addr_col = col_map.get("address")
    country_col = col_map.get("country")
    
    result = pd.DataFrame()
    result["id"] = df[id_col].astype(str).str.strip()
    result["source"] = source_label
    
    # Original columns preserved
    result["raw_name"] = df[name_col].fillna("").astype(str) if name_col else ""
    result["raw_address"] = df[addr_col].fillna("").astype(str) if addr_col else ""
    result["raw_country"] = df[country_col].fillna("").astype(str) if country_col else ""
    
    # Normalized columns
    name_tuples = [normalize_business_name(val) for val in result["raw_name"]]
    result["norm_name"] = [t[0] for t in name_tuples]
    result["name_tokens"] = [t[1] for t in name_tuples]
    
    addr_tuples = [normalize_address(val) for val in result["raw_address"]]
    result["norm_address"] = [t[0] for t in addr_tuples]
    result["addr_tokens"] = [t[1] for t in addr_tuples]
    result["numeric_tokens"] = [t[2] for t in addr_tuples]
    
    result["norm_country"] = [normalize_country(val) for val in result["raw_country"]]
    
    # Missing value indicators
    result["missing_name"] = (result["norm_name"] == "").astype(int)
    result["missing_address"] = (result["norm_address"] == "").astype(int)
    result["missing_country"] = (result["norm_country"] == "").astype(int)
    
    return result
