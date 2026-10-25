import argparse
import os
import re
from pathlib import Path
from typing import Optional, List

import pandas as pd
from tqdm import tqdm

import enhanced_config as config


TIMESTAMP_RE = re.compile(r"\s*\d+_\d+\s*")
CHAT_TAG_RE = re.compile(r"\[.*?\]")
SOUND_EVENT_RE = re.compile(r"&=\S+")
PAREN_RE = re.compile(r"\(\..*?\)")
WORD_ANNOT_RE = re.compile(r"@\w")


def parse_cha_file(file_path: Path) -> Optional[str]:
    """Return a cleaned concatenation of all *PAR lines from a .cha file."""
    utterances: List[str] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as cha_file:
            for line in cha_file:
                if not line.startswith("*PAR:"):
                    continue

                parts = line.strip().split("\t", 1)
                if len(parts) < 2:
                    continue
                text = parts[1]

                text = TIMESTAMP_RE.sub("", text)
                text = CHAT_TAG_RE.sub("", text)
                text = SOUND_EVENT_RE.sub("", text)
                text = PAREN_RE.sub("", text)
                text = text.replace("+...", "")
                text = WORD_ANNOT_RE.sub("", text)

                cleaned = " ".join(text.split()).strip()
                if cleaned and cleaned != ".":
                    utterances.append(cleaned)

    except FileNotFoundError:
        print(f"[WARN] Missing transcript: {file_path}")
        return None
    except Exception as exc:
        print(f"[ERROR] Could not parse {file_path}: {exc}")
        return None

    return " ".join(utterances) if utterances else None


def gather_metadata() -> pd.DataFrame:
    """Walks the Pitt dataset folder and returns a dataframe of usable samples."""
    base_dir = Path(config.DATA_DIR).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {base_dir}. "
            "Update DATA_DIR in enhanced_config.py or mount the dataset."
        )

    records = []
    print(f"[info] Scanning data under {base_dir}")

    for group in ("control", "dementia"):
        label = 0 if group == "control" else 1
        audio_dir = base_dir / group / "audio"
        transcript_dir = base_dir / group / "transcripts"

        if not audio_dir.exists():
            print(f"[WARN] Skipping {group}: audio folder missing at {audio_dir}")
            continue
        if not transcript_dir.exists():
            print(f"[WARN] Skipping {group}: transcript folder missing at {transcript_dir}")
            continue

        # DementiaBank distributes media as .mp3; converted copies may be .wav
        audio_files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in (".wav", ".mp3"))
        print(f"[info] {group}: found {len(audio_files)} audio files")
        for audio_path in tqdm(audio_files, desc=f"Parsing {group}", unit="file"):
            transcript_path = transcript_dir / (audio_path.stem + ".cha")
            transcription = parse_cha_file(transcript_path)
            if not transcription:
                continue  # skip samples without usable transcript

            records.append(
                {
                    "file_path": str(audio_path),
                    "transcription": transcription,
                    "label": label,
                }
            )

    return pd.DataFrame(records)


def main(output_path: Optional[Path] = None):
    df = gather_metadata()
    if df.empty:
        raise RuntimeError(
            "No usable (audio, transcript) pairs were found. "
            "Check the directory layout and transcript contents."
        )

    csv_path = Path(output_path or config.OUTPUT_CSV_PATH).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"[done] Metadata written to {csv_path}")
    print(f"[done] Total usable samples: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for enhanced pipeline.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the CSV output path (defaults to config.OUTPUT_CSV_PATH).",
    )
    args = parser.parse_args()
    main(args.output)