"""Reproduce the EXACT Keras Tokenizer ``word_index`` from WELFake_Dataset.csv
*without* TensorFlow, so the exported ONNX model can be served.

It mirrors Compute (1).ipynb step-for-step:
    drop 'Unnamed: 0'  ->  dropna  ->  drop_duplicates
    content = title + " " + text
    Tokenizer(num_words=10000).fit_on_texts(content)   # word_index is deterministic

Keras ``fit_on_texts`` counts words in first-seen order, then stable-sorts by
frequency (desc) and assigns 1-based indices -- which this reproduces exactly.

Usage:
    python scripts/build_tokenizer.py --csv "C:/path/WELFake_Dataset.csv.zip"
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the SAME tokenization the server uses, guaranteeing train/serve parity.
from app.preprocessing import text_to_word_sequence  # noqa: E402


def load_clean_contents(csv_path: str):
    """Return (list_of_content_strings, n_rows_after_clean). Prefers pandas."""
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)  # pandas infers .zip compression automatically
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df = df.dropna(axis=0).drop_duplicates()
        contents = (df["title"].astype(str) + " " + df["text"].astype(str)).tolist()
        return contents, len(df)
    except ImportError:
        return _load_clean_contents_csv(csv_path)


def _load_clean_contents_csv(csv_path: str):
    """Pure-stdlib fallback (no pandas)."""
    import csv
    import io
    import zipfile

    # WELFake articles can exceed the default 131072-char field cap.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit = int(limit // 2)

    p = Path(csv_path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            text = z.read(name).decode("utf-8", errors="replace")
        fileobj = io.StringIO(text)
    else:
        fileobj = open(p, "r", encoding="utf-8", errors="replace", newline="")

    reader = csv.reader(fileobj)
    next(reader, None)  # header: ['', 'title', 'text', 'label']
    rows = []
    for row in reader:
        if len(row) < 4:
            continue
        title, text, label = row[1], row[2], row[3]
        rows.append((title, text, label))
    if not isinstance(fileobj, io.StringIO):
        fileobj.close()

    # dropna: pandas reads empty cells as NaN -> drop rows with any empty field
    rows = [(t, x, l) for (t, x, l) in rows if t != "" and x != "" and l != ""]
    # drop_duplicates on (title, text, label), keep first occurrence
    seen, deduped = set(), []
    for r in rows:
        if r in seen:
            continue
        seen.add(r)
        deduped.append(r)
    contents = [f"{t} {x}" for (t, x, _l) in deduped]
    return contents, len(deduped)


def build_word_index(contents):
    counts = {}
    for text in contents:
        for w in text_to_word_sequence(text):
            counts[w] = counts.get(w, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)  # stable sort
    return {w: i + 1 for i, (w, _c) in enumerate(ordered)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to WELFake_Dataset.csv (or .zip)")
    ap.add_argument(
        "--out",
        default=str(ROOT / "models" / "tokenizer_word_index.json"),
        help="Output JSON path",
    )
    args = ap.parse_args()

    contents, n = load_clean_contents(args.csv)
    print(f"Rows after dropna + drop_duplicates: {n}")
    word_index = build_word_index(contents)
    print(f"Vocabulary size: {len(word_index)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(word_index, f, ensure_ascii=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
