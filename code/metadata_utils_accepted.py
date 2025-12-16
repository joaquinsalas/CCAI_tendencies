# metadata_utils.py
import pandas as pd
from subject_area_utils import parse_topic_area, parse_secondary_list, normalize_topic


def _is_accepted(status) -> bool:
    return str(status).strip().lower() != "reject"


def load_paper_metadata(meta_csv_path: str):
    """
    Expects columns:
      - 'Paper ID'
      - 'Track Name'
      - 'Primary Subject Area'
      - 'Secondary Subject Areas'
      - 'Status'
    Returns dict: paper_id(str) -> metadata dict
    """
    df = pd.read_csv(meta_csv_path, encoding="cp1252")
    meta = {}

    for _, r in df.iterrows():
        if not _is_accepted(r.get("Status", "")):
            continue

        pid = str(r.get("Paper ID", "")).strip()
        if not pid:
            continue

        primary = r.get("Primary Subject Area", "")
        pt, pa = parse_topic_area(primary)

        second = r.get("Secondary Subject Areas", "")
        sec_pairs = parse_secondary_list(second)

        meta[pid] = {
            "track_name": str(r.get("Track Name", "")).strip(),

            "primary_subject_raw": str(primary).strip(),
            "primary_subject_topic": normalize_topic(pt),
            "primary_subject_area": (pa or "").strip(),

            "secondary_subject_raw": str(second).strip(),
            "secondary_subject_topics": ";".join(
                sorted(set(normalize_topic(t) for t, _ in sec_pairs if t))
            ),
            "secondary_subject_areas": ";".join(
                sorted(set(a.strip() for _, a in sec_pairs if a))
            ),

            "all_subject_areas": ";".join(
                sorted(set(
                    [ (pa or "").strip() ] +
                    [a.strip() for _, a in sec_pairs if a]
                ) - {""})
            ),

            "all_subject_pairs": ";".join(
                sorted(set(
                    ([f"{normalize_topic(pt)}->{(pa or '').strip()}"] if pt and pa else []) +
                    [f"{normalize_topic(t)}->{a.strip()}" for t, a in sec_pairs if t and a]
                ))
            ),
        }

    return meta


def load_paper_id_mapping(meta_csv_path: str):
    """
    Returns dict: paper_id (str) -> paper_id (str)
    Only for ACCEPTED papers.
    """
    df = pd.read_csv(meta_csv_path, encoding="cp1252")

    mapping = {}
    for _, r in df.iterrows():
        if not _is_accepted(r.get("Status", "")):
            continue

        pid = str(r.get("Paper ID", "")).strip()
        if pid.isdigit():
            mapping[pid] = pid

    return mapping
