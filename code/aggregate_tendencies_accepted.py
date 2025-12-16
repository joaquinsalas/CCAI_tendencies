# aggregate_tendencies.py

import pandas as pd
from pathlib import Path


MASTER_CSV    = "../data/out_master_accepted.csv"

OUT_DIR  = "/mnt/data-r1/JoaquinSalas/Documents/informs/conferences/2025CCAI/data/"

def explode_counts(df, column):
    """
    df[column] has semicolon-separated labels.
    Returns a DataFrame with columns: [column, count].
    """
    series = df[column].fillna("").astype(str)
    labels = (
        series[series != ""]
        .str.split(";")
        .explode()
        .str.strip()
    )
    counts = labels.value_counts().reset_index()
    counts.columns = [column, "count"]
    return counts

def main(master_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(master_csv)

    tendencies = [
        "techniques",
        "climate_areas",
        "data_modalities",
        "tasks",
        "supervision",
        "paradigms",
        "spatial_scales",
        "temporal_scales",
        "metrics",
        "interdisciplinary",
        "foundation_models",
        "openness",
        "geography",
        "deployment",
        "uncertainty",
        "climate_purpose",
        "model_scale",
        "compute_footprint",
    ]

    tendencies += [
        "primary_subject_area",
        "all_subject_areas",
        "all_subject_pairs",
        "track_name",
        "primary_subject_topic",
        "secondary_subject_areas",
    ]

    for col in tendencies:
        counts = explode_counts(df, col)
        counts.to_csv(out_dir / f"{col}_counts_accepted.csv", index=False)

    # Example of more structured tables:
    # climate-purpose x geography matrix
    df_cp = df.copy()
    df_cp["primary_climate_purpose"] = df_cp["primary_climate_purpose"].fillna("")
    mat = (
        df_cp[df_cp["primary_climate_purpose"] != ""]
        .assign(geo=df_cp["geography"].fillna("").str.split(";"))
        .explode("geo")
        .query("geo != ''")
        .groupby(["primary_climate_purpose", "geo"])
        .size()
        .reset_index(name="count")
    )
    mat.to_csv(out_dir / "climate_purpose_by_geography_accepted.csv", index=False)

    # organizer primary area x GPT climate_areas
    mat1 = (
        df.assign(primary_area=df["primary_subject_area"].fillna(""))
        .assign(cl=df["climate_areas"].fillna("").str.split(";"))
        .explode("cl")
        .query("primary_area != '' and cl != ''")
        .groupby(["primary_area", "cl"])
        .size().reset_index(name="count")
    )
    mat1.to_csv(out_dir / "primary_subject_area_by_gpt_climate_areas_accepted.csv", index=False)

    # organizer all_subject_areas x GPT techniques
    mat2 = (
        df.assign(area=df["all_subject_areas"].fillna("").str.split(";"))
        .assign(tech=df["techniques"].fillna("").str.split(";"))
        .explode("area").explode("tech")
        .query("area != '' and tech != ''")
        .groupby(["area", "tech"])
        .size().reset_index(name="count")
    )
    mat2.to_csv(out_dir / "organizer_area_by_gpt_techniques_accepted.csv", index=False)


if __name__ == "__main__":

    main(MASTER_CSV,  OUT_DIR)