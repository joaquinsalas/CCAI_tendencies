import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# ----------------------------
# Config
# ----------------------------
main_url = "https://www.climatechange.ai/events/neurips2025#accepted-works"
output_dir = "/mnt/data-r1/JoaquinSalas/Documents/informs/conferences/2025CCAI/final_papers_ccai"
meta_csv   = "/mnt/data-r1/JoaquinSalas/Documents/informs/conferences/2025CCAI/data/2025.11.11papers.xls.csv"
missing_csv = os.path.join(output_dir, "missing_pdfs.csv")

os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Title matching helpers
# ----------------------------
def normalize_title(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s*\(.*track\)\s*$", "", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

try:
    from rapidfuzz import process, fuzz
    def best_match_id(title, choices_norm, id_by_norm, min_score=80):
        q = normalize_title(title)
        if not q:
            return None, None
        m = process.extractOne(q, choices_norm, scorer=fuzz.token_set_ratio)
        if not m:
            return None, None
        best_norm, score, _ = m
        if score < min_score:
            return None, score
        return id_by_norm.get(best_norm), score
except Exception:
    import difflib
    def ratio(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0
    def best_match_id(title, choices_norm, id_by_norm, min_score=80):
        q = normalize_title(title)
        if not q:
            return None, None
        best_norm, best_score = None, -1.0
        for c in choices_norm:
            sc = ratio(q, c)
            if sc > best_score:
                best_norm, best_score = c, sc
        if best_score < min_score:
            return None, best_score
        return id_by_norm.get(best_norm), best_score

# ----------------------------
# Load metadata
# ----------------------------
dfm = pd.read_csv(meta_csv, encoding="cp1252")
need_cols = {"Paper ID", "Paper Title"}
missing = need_cols - set(dfm.columns)
if missing:
    raise ValueError(f"Metadata CSV missing columns: {missing}")

dfm["__norm_title__"] = dfm["Paper Title"].astype(str).map(normalize_title)
dfm = dfm[dfm["__norm_title__"].str.len() > 0].drop_duplicates("__norm_title__")

choices_norm = dfm["__norm_title__"].tolist()
id_by_norm   = dict(zip(dfm["__norm_title__"], dfm["Paper ID"]))

# ----------------------------
# Scrape accepted works
# ----------------------------
session = requests.Session()
resp = session.get(main_url)
if resp.status_code != 200:
    raise RuntimeError(f"Failed to retrieve main page, status {resp.status_code}")

soup = BeautifulSoup(resp.text, "html.parser")

papers_section    = soup.find(lambda t: t.name in ["h3", "h2"] and "Papers" in t.text)
proposals_section = soup.find(lambda t: t.name in ["h3", "h2"] and "Proposals" in t.text)
if not papers_section or not proposals_section:
    raise SystemExit("Accepted works sections not found.")

def get_section_links(section_header_tag):
    links = []
    for tag in section_header_tag.find_all_next():
        if tag.name in ["h2", "h3"] and tag.text.strip() not in ["", "Title", "Authors", "Poster", "Session"]:
            break
        if tag.name == "a" and tag.get("href") and "/papers/neurips2025/" in tag["href"]:
            links.append(requests.compat.urljoin(main_url, tag["href"]))
    return sorted(set(links))

paper_links    = get_section_links(papers_section)
proposal_links = get_section_links(proposals_section)
all_links = paper_links + proposal_links

print(f"Found {len(paper_links)} papers and {len(proposal_links)} proposals (total {len(all_links)}).")

# ----------------------------
# Download PDFs + record missing
# ----------------------------
invalid_chars = '<>:"/\\|?*'

def safe_filename(name: str) -> str:
    name = re.sub(f"[{re.escape(invalid_chars)}]", "_", name).strip().strip(".")
    name = re.sub(r"\s+", " ", name).strip()
    return name

missing_rows = []

for s3_idx, link in tqdm(list(enumerate(all_links, start=1)), desc="Downloading PDFs", unit="paper"):
    try:
        detail_resp = session.get(link, timeout=30)
        if detail_resp.status_code != 200:
            missing_rows.append({
                "s3_idx": s3_idx,
                "link": link,
                "paper_id": "",
                "match_score": "",
                "title": "",
                "reason": f"detail_page_status_{detail_resp.status_code}",
                "pdf_url": "",
            })
            continue

        detail_soup = BeautifulSoup(detail_resp.text, "html.parser")
        title_tag = detail_soup.find(["h1", "h2", "h3"])
        if not title_tag:
            missing_rows.append({
                "s3_idx": s3_idx,
                "link": link,
                "paper_id": "",
                "match_score": "",
                "title": "",
                "reason": "no_title_found_on_detail_page",
                "pdf_url": "",
            })
            continue

        title_text = re.sub(r"\s*\(.*Track\)$", "", title_tag.get_text().strip()) or "untitled"

        paper_id, score = best_match_id(title_text, choices_norm, id_by_norm, min_score=80)
        if paper_id is None:
            missing_rows.append({
                "s3_idx": s3_idx,
                "link": link,
                "paper_id": "",
                "match_score": score if score is not None else "",
                "title": title_text,
                "reason": "title_no_match_in_metadata",
                "pdf_url": "",
            })
            continue

        file_path = os.path.join(output_dir, safe_filename(f"{int(paper_id):03d} - {title_text}") + ".pdf")
        if os.path.exists(file_path):
            continue

        pdf_url = (
            "https://s3.us-east-1.amazonaws.com/"
            f"climate-change-ai/papers/neurips2025/{s3_idx}/paper.pdf"
        )

        pdf_resp = session.get(pdf_url, stream=True, timeout=60)
        if pdf_resp.status_code != 200:
            missing_rows.append({
                "s3_idx": s3_idx,
                "link": link,
                "paper_id": str(paper_id),
                "match_score": score if score is not None else "",
                "title": title_text,
                "reason": f"pdf_status_{pdf_resp.status_code}",
                "pdf_url": pdf_url,
            })
            continue

        with open(file_path, "wb") as f:
            for chunk in pdf_resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        pdf_resp.close()

    except Exception as e:
        missing_rows.append({
            "s3_idx": s3_idx,
            "link": link,
            "paper_id": "",
            "match_score": "",
            "title": "",
            "reason": f"exception_{type(e).__name__}",
            "pdf_url": "",
        })

session.close()

# Write missing report
df_missing = pd.DataFrame(missing_rows, columns=[
    "s3_idx", "paper_id", "match_score", "title", "link", "pdf_url", "reason"
])
df_missing.to_csv(missing_csv, index=False)
print(f"Missing PDF report written to: {missing_csv} ({len(df_missing)} rows)")




