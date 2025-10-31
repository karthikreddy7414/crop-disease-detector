import csv
import os
from pathlib import Path

try:
    from Bio import Entrez, SeqIO
except Exception as exc:
    raise SystemExit(
        "Biopython is required. Install with: pip install biopython"
    ) from exc


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dna_csv = project_root / "data" / "dna" / "dna_labels.csv"
    os.makedirs(dna_csv.parent, exist_ok=True)

    # Minimal demo accessions per class (for sample data only; replace for real use)
    accessions_per_class = {
        "bacterial_blight": ["KY849228.1", "KF214706.1"],
        "leaf_blast": ["XM_015781103.2", "XM_015779495.2"],
        "sheath_blight": ["XM_015783640.2", "XM_015783574.2"],
        "healthy": ["NM_001065.4", "NM_000207.3"],
    }

    Entrez.email = os.environ.get("ENTREZ_EMAIL", "user@example.com")
    rows = []
    for disease, accs in accessions_per_class.items():
        for acc in accs:
            try:
                handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text")
                records = list(SeqIO.parse(handle, "fasta"))
                handle.close()
            except Exception:
                continue
            if not records:
                continue
            seq = str(records[0].seq)
            if not seq:
                continue
            rows.append([acc, "rice", disease, seq])

    with open(dna_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "crop_type", "disease", "sequence"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {dna_csv}")


if __name__ == "__main__":
    main()


