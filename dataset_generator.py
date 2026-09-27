"""
Benchmark Dataset Generator for Business Entity Resolution.
Uses pure Python standard library (csv, random) to generate benchmark TSV files
without external dependencies.
"""

import os
import csv
import random

SAMPLE_BUSINESSES = [
    # US Businesses
    ("Acme Global Technologies Corporation", "101 Market Street, Suite 400, San Francisco, CA 94105", "US"),
    ("Apex Logistics and Supply Chain LLC", "450 North Michigan Avenue, Chicago, IL 60611", "US"),
    ("Blue Horizon Medical Devices Inc", "1200 Beacon Street, Boston, MA 02116", "US"),
    ("Cascade Financial Advisory Group", "800 Fifth Avenue, Floor 12, Seattle, WA 98104", "US"),
    ("Dynamic Cloud Software Solutions Inc", "3400 Hillview Avenue, Palo Alto, CA 94304", "US"),
    ("Echo Valley Energy Partners LP", "1100 Louisiana Street, Houston, TX 77002", "US"),
    ("Falcon Robotics & Automation Co", "500 West Madison Street, Chicago, IL 60661", "US"),
    ("Golden Gate Hospitality Management LLC", "99 Geary Street, San Francisco, CA 94108", "US"),
    ("Harbor View Shipping & Freight Corp", "200 Seaport Boulevard, Boston, MA 02210", "US"),
    ("Infinity Retail Ventures Incorporated", "767 Fifth Avenue, New York, NY 10153", "US"),
    ("Juniper BioPharma Laboratories LLC", "400 Technology Square, Cambridge, MA 02139", "US"),
    ("Keystone Precision Manufacturing Co", "1500 Spring Garden Street, Philadelphia, PA 19130", "US"),

    # India Businesses
    ("Tata Consultancy Services Limited", "TCS House, Raveline Street, Fort, Mumbai 400001", "India"),
    ("Infosys Technologies Private Limited", "Electronics City, Hosur Road, Bengaluru 560100", "India"),
    ("Wipro Digital Solutions Private Limited", "Doddakannelli, Sarjapur Road, Bengaluru 560035", "India"),
    ("Reliance Retail Enterprises Limited", "Maker Chambers IV, Nariman Point, Mumbai 400021", "India"),
    ("HCL Technologies and Infrastructure Ltd", "Technology Hub, Sector 126, Noida, UP 201304", "India"),
    ("Larsen & Toubro Engineering Corporation", "L&T House, Ballard Estate, Mumbai 400001", "India"),
    ("Bharat Heavy Electricals Limited", "BHEL House, Siri Fort, New Delhi 110049", "India"),
    ("Mahindra Aerospace & Defence Solutions", "Mahindra Towers, Worli, Mumbai 400018", "India"),
    ("Sun Pharmaceutical Industries Ltd", "Sun House, Western Express Highway, Goregaon, Mumbai 400063", "India"),
    ("Tech Mahindra Business Services Pvt Ltd", "Sharda Centre, Erandwane, Pune 411004", "India"),

    # France Businesses (Unseen country testing)
    ("L'Oreal Recherche & Innovation SAS", "41 Rue Martre, Clichy 92117", "France"),
    ("Dassault Systemes Software Solutions SE", "10 Rue Marcel Dassault, Velizy-Villacoublay 78140", "France"),
    ("Capgemini Consulting Services SAS", "11 Rue de Tilsitt, Paris 75017", "France"),
    ("TotalEnergies Renouvelables France", "2 Place Jean Millier, La Defense 92078", "France"),
    ("Schneider Electric Industries SAS", "35 Rue Joseph Monier, Rueil-Malmaison 92500", "France"),
    ("Sanofi Pasteur Vaccines Laboratories", "14 Espace Henry Vallee, Lyon 69007", "France"),
    ("Air Liquide Technologies International", "75 Quai d'Orsay, Paris 75007", "France"),
    ("BNP Paribas Asset Management France", "1 Boulevard Haussmann, Paris 75009", "France"),

    # Germany Businesses
    ("Siemens Digital Industries Software GmbH", "Werner-von-Siemens-Strasse 1, Munich 80333", "Germany"),
    ("SAP Enterprise Solutions SE", "Dietmar-Hopp-Allee 16, Walldorf 69190", "Germany"),
    ("Bosch Mobility Systems Technologies GmbH", "Robert-Bosch-Platz 1, Gerlingen 70839", "Germany"),
    ("Bayer CropScience International AG", "Alfred-Nobel-Strasse 50, Monheim am Rhein 40789", "Germany"),

    # UK Businesses
    ("AstraZeneca Pharmaceuticals UK Limited", "1 Francis Crick Avenue, Cambridge CB2 0AA", "United Kingdom"),
    ("Rolls-Royce Aerospace Technologies PLC", "Kings Place, 90 York Way, London N1 9FX", "United Kingdom"),
    ("Vodafone Enterprise Global Services Ltd", "One Kingdom Street, Paddington Central, London W2 6BD", "United Kingdom"),
    ("Reckitt Benckiser Consumer Health PLC", "103-105 Bath Road, Slough SL1 3UH", "United Kingdom"),
]


def add_name_noise(name: str) -> str:
    tokens = name.split()
    r = random.random()
    if r < 0.25:
        name = name.replace("Corporation", "Corp").replace("Incorporated", "Inc").replace("Limited", "Ltd")
        name = name.replace("Private Limited", "Pvt. Ltd.").replace("LLC", "L.L.C.")
    elif r < 0.45:
        if " & " in name:
            name = name.replace(" & ", " and ")
        elif " and " in name:
            name = name.replace(" and ", " & ")
    elif r < 0.60 and len(tokens) > 2:
        idx = random.randint(1, len(tokens) - 2)
        tokens.pop(idx)
        name = " ".join(tokens)
    elif r < 0.75:
        char_idx = random.randint(2, len(name) - 3)
        if name[char_idx].isalpha():
            name = name[:char_idx] + name[char_idx].lower() + name[char_idx+1:]
    return name


def add_address_noise(addr: str) -> str:
    r = random.random()
    if r < 0.30:
        addr = addr.replace("Street", "St.").replace("Avenue", "Ave").replace("Boulevard", "Blvd").replace("Road", "Rd")
        addr = addr.replace("Suite", "Ste").replace("Floor", "Fl")
    elif r < 0.55:
        parts = addr.split(",")
        if len(parts) > 2:
            parts.pop(1)
            addr = ",".join(parts)
    elif r < 0.70:
        addr = addr.lower()
    return addr


def write_tsv(filepath: str, fieldnames, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def generate_benchmark_datasets(base_dir: str = "dataset", seed: int = 42):
    random.seed(seed)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    businesses = list(SAMPLE_BUSINESSES)
    random.shuffle(businesses)
    train_pool = businesses[:26]
    test_pool = businesses[26:]

    def build_split(pool, prefix, is_train=True):
        s1_rows = []
        s2_rows = []
        s3_rows = []
        ground_truth = []

        s2_counter = 1
        s3_counter = 1

        for i, (name, addr, country) in enumerate(pool):
            s1_id = f"S1_{prefix}_{i+1:04d}"
            s1_rows.append({
                "entity_id": s1_id,
                "business_name": name,
                "business_address": addr,
                "country": country
            })

            match_type = random.random()
            matched_ids = []

            if match_type >= 0.20:
                if match_type < 0.60 or match_type >= 0.80:
                    s2_id = f"S2_{prefix}_{s2_counter:04d}"
                    s2_counter += 1
                    s2_rows.append({
                        "record_id": s2_id,
                        "business_name": add_name_noise(name),
                        "business_address": add_address_noise(addr),
                        "country": country if random.random() > 0.1 else ""
                    })
                    matched_ids.append(s2_id)

                if match_type >= 0.60:
                    s3_id = f"S3_{prefix}_{s3_counter:04d}"
                    s3_counter += 1
                    s3_rows.append({
                        "record_id": s3_id,
                        "business_name": add_name_noise(name),
                        "business_address": add_address_noise(addr),
                        "country": country if random.random() > 0.1 else ""
                    })
                    matched_ids.append(s3_id)

            ground_truth.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": "[" + ", ".join(matched_ids) + "]"
            })

        # Add distractor records in S2 and S3
        distractors = [
            ("Apex Global Technologies", "102 Market St, San Francisco, CA", "US"),
            ("Blue Horizon Healthcare", "1250 Beacon St, Boston, MA", "US"),
            ("Infosys BPO Services", "Electronics City Phase 2, Bengaluru", "India"),
            ("Siemens Healthineers AG", "Henkestrasse 127, Erlangen", "Germany"),
            ("Sanofi Consumer Healthcare", "54 Rue La Boetie, Paris", "France"),
        ]
        for d_name, d_addr, d_cty in distractors:
            s2_id = f"S2_{prefix}_{s2_counter:04d}"
            s2_counter += 1
            s2_rows.append({
                "record_id": s2_id,
                "business_name": d_name,
                "business_address": d_addr,
                "country": d_cty
            })

            s3_id = f"S3_{prefix}_{s3_counter:04d}"
            s3_counter += 1
            s3_rows.append({
                "record_id": s3_id,
                "business_name": d_name + " Intl",
                "business_address": d_addr,
                "country": d_cty
            })

        return s1_rows, s2_rows, s3_rows, ground_truth

    # Build Train
    s1, s2, s3, gt = build_split(train_pool, "TRN", is_train=True)
    write_tsv(os.path.join(train_dir, "train_source1.tsv"), ["entity_id", "business_name", "business_address", "country"], s1)
    write_tsv(os.path.join(train_dir, "train_source2.tsv"), ["record_id", "business_name", "business_address", "country"], s2)
    write_tsv(os.path.join(train_dir, "train_source3.tsv"), ["record_id", "business_name", "business_address", "country"], s3)
    write_tsv(os.path.join(train_dir, "train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"], gt)

    # Build Test
    s1_t, s2_t, s3_t, _ = build_split(test_pool, "TST", is_train=False)
    write_tsv(os.path.join(test_dir, "test_source1.tsv"), ["entity_id", "business_name", "business_address", "country"], s1_t)
    write_tsv(os.path.join(test_dir, "test_source2.tsv"), ["record_id", "business_name", "business_address", "country"], s2_t)
    write_tsv(os.path.join(test_dir, "test_source3.tsv"), ["record_id", "business_name", "business_address", "country"], s3_t)

    print(f"[DATASET GENERATOR] Successfully created benchmark datasets in '{base_dir}'")
    print(f"  Train: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, GT={len(gt)}")
    print(f"  Test:  S1={len(s1_t)}, S2={len(s2_t)}, S3={len(s3_t)}")


if __name__ == "__main__":
    generate_benchmark_datasets()
