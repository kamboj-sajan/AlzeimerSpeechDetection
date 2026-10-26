import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from tqdm import tqdm

import enhanced_config as config


# Cookie Theft description, phonemic fluency, semantic fluency
TASKS = ("CTD", "PFT", "SFT")

# meta-info.csv column names are matched loosely (case-insensitive substrings, in priority order)
ID_HINTS = ("record", "participant", "id")
LABEL_HINTS = ("class", "diagnos", "label", "group")
SPLIT_HINTS = ("split", "partition", "trainor", "set")

LABEL_SCHEMES = {
    # Healthy controls vs any cognitive impairment (MCI or dementia)
    "impaired": {"hc": 0, "mci": 1, "dementia": 1},
    # Healthy controls vs dementia; MCI participants are dropped
    "dementia": {"hc": 0, "dementia": 1},
}


def find_column(columns: Sequence[str], hints: Sequence[str]) -> Optional[str]:
    for hint in hints:
        for column in columns:
            if hint in column.lower():
                return column
    return None


def normalize_diagnosis(value) -> Optional[str]:
    text = str(value).strip().lower()
    if "mci" in text or "mild cognitive" in text:
        return "mci"
    if "dement" in text or text == "ad":
        return "dementia"
    if text in ("hc", "nc", "control") or "healthy" in text:
        return "hc"
    return None


def normalize_split(value) -> Optional[str]:
    text = str(value).strip().lower()
    if "train" in text:
        return "train"
    if "test" in text or "dev" in text:
        return "test"
    return None


def resolve_folder(record_id, folders: dict) -> Optional[Path]:
    """Map a meta-info record ID (e.g. 'PROCESS-2_rec__001' or 1) to its recording folder."""
    key = str(record_id).strip()
    if key in folders:
        return folders[key]
    numbers = re.findall(r"\d+", key)
    if numbers:
        return folders.get(f"PROCESS-2_rec__{int(numbers[-1]):03d}")
    return None


def gather_metadata(data_dir: Path, tasks: Sequence[str], scheme: str) -> pd.DataFrame:
    meta_path = data_dir / "meta-info.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta-info.csv not found in {data_dir}. "
            "Download PROCESS-2 there or pass --data-dir."
        )

    meta = pd.read_csv(meta_path)
    id_col = find_column(meta.columns, ID_HINTS)
    label_col = find_column(meta.columns, LABEL_HINTS)
    split_col = find_column([c for c in meta.columns if c not in (id_col, label_col)], SPLIT_HINTS)
    print(f"[info] meta-info.csv columns: {list(meta.columns)}")
    print(f"[info] using id='{id_col}', diagnosis='{label_col}', split='{split_col}'")
    if id_col is None or label_col is None:
        raise RuntimeError("Could not identify the record-ID and diagnosis columns; update the *_HINTS lists.")
    if split_col is None:
        print("[WARN] No train/test split column found; training will fall back to cross-validation.")

    folders = {p.name: p for p in data_dir.iterdir() if p.is_dir()}
    label_map = LABEL_SCHEMES[scheme]
    records = []
    skipped = {"unknown diagnosis": 0, "excluded by label scheme": 0, "missing folder": 0, "missing audio": 0}

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Parsing PROCESS-2", unit="participant"):
        diagnosis = normalize_diagnosis(row[label_col])
        if diagnosis is None:
            skipped["unknown diagnosis"] += 1
            continue
        if diagnosis not in label_map:
            skipped["excluded by label scheme"] += 1
            continue
        folder = resolve_folder(row[id_col], folders)
        if folder is None:
            skipped["missing folder"] += 1
            continue

        for task in tasks:
            audio_path = folder / f"{folder.name}__{task}.wav"
            if not audio_path.exists():
                skipped["missing audio"] += 1
                continue
            transcript_path = audio_path.with_suffix(".txt")
            transcription = ""
            if transcript_path.exists():
                transcription = " ".join(transcript_path.read_text(encoding="utf-8", errors="ignore").split())

            record = {
                "file_path": str(audio_path.resolve()),
                "transcription": transcription,
                "label": label_map[diagnosis],
                "participant_id": folder.name,
                "task": task,
                "diagnosis": diagnosis,
            }
            if split_col is not None:
                record["split"] = normalize_split(row[split_col])
            records.append(record)

    print(f"[info] skipped: {skipped}")
    return pd.DataFrame(records)


def main(data_dir: Path, tasks: Sequence[str], scheme: str, output_path: Optional[Path] = None):
    df = gather_metadata(data_dir.resolve(), tasks, scheme)
    if df.empty:
        raise RuntimeError("No usable recordings were found. Check --data-dir and meta-info.csv.")

    csv_path = Path(output_path or config.OUTPUT_CSV_PATH).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"[done] Metadata written to {csv_path}")
    print(f"[done] {len(df)} recordings from {df['participant_id'].nunique()} participants")
    group_cols = ["split", "diagnosis"] if "split" in df.columns else ["diagnosis"]
    print(df.groupby(group_cols)["participant_id"].nunique().rename("participants").to_string())
    print(f"[check] sample transcript: {df['transcription'].iloc[0][:200]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for the PROCESS-2 corpus.")
    parser.add_argument("--data-dir", type=Path, default=Path(config.PROCESS2_DIR),
                        help="Folder containing meta-info.csv and the PROCESS-2_rec__* folders.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=["CTD"],
                        help="Which recordings to use per participant (default: Cookie Theft only).")
    parser.add_argument("--labels", choices=sorted(LABEL_SCHEMES), default="impaired",
                        help="'impaired': healthy vs MCI+dementia; 'dementia': healthy vs dementia (drops MCI).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional override for the CSV output path (defaults to config.OUTPUT_CSV_PATH).")
    args = parser.parse_args()
    main(args.data_dir, args.tasks, args.labels, args.output)
