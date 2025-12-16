# extract_papers.py

import os
import json
from pathlib import Path
from collections import defaultdict

from tqdm.auto import tqdm



from openai import OpenAI

from pdf_utils import iter_pdf_chunks
from prompt_template import SYSTEM_PROMPT, build_user_prompt
from label_space import (
    TECHNIQUES, CLIMATE_AREAS, DATA_MODALITIES, TASKS, SUPERVISION, PARADIGMS,
    SPATIAL_SCALES, TEMPORAL_SCALES, METRICS, INTERDISCIPLINARY,
    FOUNDATION_MODELS, OPENNESS, GEOGRAPHY, DEPLOYMENT, UNCERTAINTY,
    CLIMATE_PURPOSE, MODEL_SCALE, COMPUTE_FOOTPRINT
)

import pandas as pd

from metadata_utils_accepted import load_paper_metadata, load_paper_id_mapping

PAPERS_ROOT   = "/mnt/data-r1/JoaquinSalas/Documents/informs/conferences/2025CCAI/final_papers_ccai/"
MASTER_CSV    = "../data/out_master_accepted.csv"
RESUME_CSV    = "../data/resume_accepted.csv"
METADATA_CSV  = "/mnt/data-r1/JoaquinSalas/Documents/informs/conferences/2025CCAI/data/2025.11.11papers.xls.csv"

client = OpenAI()  # assumes OPENAI_API_KEY env var

# -------------------
# Helpers
# -------------------

ALL_CATEGORIES = [
    "techniques", "climate_areas", "data_modalities", "tasks",
    "supervision", "paradigms", "spatial_scales", "temporal_scales",
    "metrics", "interdisciplinary", "foundation_models", "openness",
    "geography", "deployment", "uncertainty",
    "climate_purpose", "model_scale", "compute_footprint",
]

ALLOWED_MAP = {
    "techniques": TECHNIQUES,
    "climate_areas": CLIMATE_AREAS,
    "data_modalities": DATA_MODALITIES,
    "tasks": TASKS,
    "supervision": SUPERVISION,
    "paradigms": PARADIGMS,
    "spatial_scales": SPATIAL_SCALES,
    "temporal_scales": TEMPORAL_SCALES,
    "metrics": METRICS,
    "interdisciplinary": INTERDISCIPLINARY,
    "foundation_models": FOUNDATION_MODELS,
    "openness": OPENNESS,
    "geography": GEOGRAPHY,
    "deployment": DEPLOYMENT,
    "uncertainty": UNCERTAINTY,
    "climate_purpose": CLIMATE_PURPOSE,
    "model_scale": MODEL_SCALE,
    "compute_footprint": COMPUTE_FOOTPRINT,
}



import re

def find_pdfs_in_flat_dir(papers_root: Path):
    """Find all PDFs directly under papers_root (non-recursive or recursive both ok)."""
    return sorted(p for p in papers_root.rglob("*.pdf") if p.is_file())

def pdf_to_paper_id_from_filename(pdf_path: Path) -> str:
    """
    Extracts paper_id from filename prefix 'DDD' (3 digits).
    Examples:
      '023 - Something.pdf' -> '23'
      '117.pdf' -> '117' (only if it is '117' with zero padding is optional)
      '007_anything.pdf' -> '7'
    Returns None if not matched.
    """
    m = re.match(r"^\s*(\d{3})\b", pdf_path.name)
    if not m:
        return None
    return str(int(m.group(1)))  # remove leading zeros





def call_gpt_for_chunk(paper_id, chunk_index, text_chunk, model="gpt-4.1-mini"):
    user_prompt = build_user_prompt(paper_id, chunk_index, text_chunk)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def merge_chunk_labels(running, new_labels):
    for cat in ALL_CATEGORIES:
        labels = new_labels.get(cat, [])
        for lab in labels:
            if lab in ALLOWED_MAP[cat]:
                running[cat].add(lab)
    return running


def finalize_record(paper_id, merged_sets, meta_by_id=None):
    def fmt(cat):
        return ";".join(sorted(merged_sets[cat])) if merged_sets[cat] else ""

    record = {"paper_id": paper_id}
    for cat in ALL_CATEGORIES:
        record[cat] = fmt(cat)

    purposes = merged_sets["climate_purpose"]
    record["primary_climate_purpose"] = (
        list(purposes)[0] if len(purposes) == 1 else ("mixed" if len(purposes) > 1 else "")
    )

    if meta_by_id and paper_id in meta_by_id:
        record.update(meta_by_id[paper_id])
    else:
        record.update({
            "track_name": "",
            "primary_subject_raw": "",
            "primary_subject_topic": "",
            "primary_subject_area": "",
            "secondary_subject_raw": "",
            "secondary_subject_topics": "",
            "secondary_subject_areas": "",
            "all_subject_areas": "",
            "all_subject_pairs": "",
        })

    return record


# -------------------
# Main incremental driver
# -------------------

def process_all_pdfs(pdf_dir, out_master_csv, meta_csv=None, resume_csv=None):
    pdf_dir = Path(pdf_dir)
    out_master_csv = Path(out_master_csv)
    out_master_csv.parent.mkdir(parents=True, exist_ok=True)

    resume_path = Path(resume_csv) if resume_csv else None
    if resume_path:
        resume_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    processed_ids = set()
    rows = []
    if resume_path and resume_path.exists():
        df_old = pd.read_csv(resume_path)
        if "paper_id" in df_old.columns:
            processed_ids = set(df_old["paper_id"].astype(str).tolist())
            rows = df_old.to_dict(orient="records")

    meta_by_id = load_paper_metadata(meta_csv) if meta_csv else None


    # --- find PDFs in final_papers_ccai by filename prefix ---
    pdfs = find_pdfs_in_flat_dir(pdf_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {pdf_dir}")

    paper_to_pdf = {}
    skipped = []
    for pdf_path in pdfs:
        paper_id = pdf_to_paper_id_from_filename(pdf_path)
        if paper_id is None:
            skipped.append(pdf_path.name)
            continue

        # if duplicates exist for same ID, keep largest
        if paper_id not in paper_to_pdf or pdf_path.stat().st_size > paper_to_pdf[paper_id].stat().st_size:
            paper_to_pdf[paper_id] = pdf_path

    if skipped:
        print(f"Skipped {len(skipped)} PDFs (no 3-digit paper_id prefix). Example: {skipped[0]}")

    def sort_key(pid):
        return (0, int(pid)) if str(pid).isdigit() else (1, str(pid))

    for paper_id in tqdm(
            sorted(paper_to_pdf.keys(), key=sort_key),
            desc="Processing papers",
            unit="paper"
    ):
        if paper_id in processed_ids:
            continue

        pdf_path = paper_to_pdf[paper_id]
        merged_sets = {cat: set() for cat in ALL_CATEGORIES}

        for chunk_idx, text_chunk in tqdm(
                list(iter_pdf_chunks(pdf_path)),
                desc=f"Chunks ({paper_id})",
                leave=False
        ):
            labels = call_gpt_for_chunk(paper_id, chunk_idx, text_chunk)
            merged_sets = merge_chunk_labels(merged_sets, labels)

        record = finalize_record(paper_id, merged_sets, meta_by_id=meta_by_id)
        record["pdf_path"] = str(pdf_path)
        rows.append(record)

        df = pd.DataFrame(rows)
        df.to_csv(out_master_csv, index=False)
        if resume_path:
            df.to_csv(resume_path, index=False)


if __name__ == "__main__":
    process_all_pdfs(PAPERS_ROOT, MASTER_CSV, meta_csv=METADATA_CSV, resume_csv=RESUME_CSV)