"""
03_rank.py
The fast ranking script. Loads pre-computed features + embeddings,
computes weighted composite scores, writes top-100 submission CSV.

v2: Honeypot/fictional-company filtering is now fully baked into
disq_penalty from feature_gen — this script no longer needs a separate
runtime patch for it. Simpler and less error-prone.

MUST complete in < 5 minutes. Actual runtime: ~10-20 seconds.
No API calls. No GPU. Pure numpy/pandas math.

Usage:
    python scripts/03_rank.py
    python scripts/03_rank.py --out outputs/team_xxx.csv
"""

import argparse
import json
import os
import gzip
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_PATH   = os.path.join(ARTIFACT_DIR, "features.parquet")
EMBED_PATH      = os.path.join(ARTIFACT_DIR, "embeddings.npy")
JD_EMBED_PATH   = os.path.join(ARTIFACT_DIR, "jd_embedding.npy")
JD_REQ_PATH     = os.path.join(ARTIFACT_DIR, "jd_requirements.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.jsonl")
CANDIDATES_GZ   = os.path.join(DATA_DIR, "candidates.jsonl.gz")

# ── load everything ────────────────────────────────────────────────────────────

def load_artifacts():
    print("Loading pre-computed artifacts...")
    df = pd.read_parquet(FEATURES_PATH)
    embeddings = np.load(EMBED_PATH)
    jd_embedding = np.load(JD_EMBED_PATH)
    with open(JD_REQ_PATH, "r") as f:
        jd = json.load(f)
    print(f"  Loaded {len(df)} candidates, embeddings {embeddings.shape}")
    if "disq_penalty" not in df.columns:
        raise RuntimeError(
            "features.parquet is from an old script version (no disq_penalty "
            "column with honeypot handling). Re-run 02_feature_gen.py."
        )
    return df, embeddings, jd_embedding, jd

# ── semantic similarity ────────────────────────────────────────────────────────

def compute_semantic_scores(embeddings, jd_embedding):
    scores = embeddings @ jd_embedding
    scores = np.clip(scores, 0, 1)
    return scores.astype(np.float32)

# ── composite score ────────────────────────────────────────────────────────────

def compute_final_scores(df, semantic_scores, weights):
    w = weights
    composite = (
        w["semantic_similarity"] * semantic_scores +
        w["career_quality"]      * df["career_score"].values +
        w["skill_match"]         * df["skill_score"].values +
        w["location"]            * df["location_score"].values +
        w["education"]           * df["education_score"].values
    )
    final = composite * df["behavioral_mult"].values * df["disq_penalty"].values
    return final.astype(np.float64)

# ── load candidate profiles for reasoning ─────────────────────────────────────

def load_top_candidates(top_ids):
    top_set = set(top_ids)
    result = {}

    def try_file(f):
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id", "")
                if cid in top_set:
                    result[cid] = c
                    if len(result) == len(top_set):
                        return
            except Exception:
                continue

    if os.path.exists(CANDIDATES_PATH):
        with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
            try_file(f)
    elif os.path.exists(CANDIDATES_GZ):
        with gzip.open(CANDIDATES_GZ, "rt", encoding="utf-8") as f:
            try_file(f)

    return result

# ── reasoning generation (pure Python, no LLM, no hallucination) ──────────────

def generate_reasoning(rank, c, feat_row):
    p       = c.get("profile", {}) or {}
    sig     = c.get("redrob_signals", {}) or {}
    history = c.get("career_history", []) or []
    skills  = c.get("skills", []) or []

    yoe     = p.get("years_of_experience", 0) or 0
    title   = p.get("current_title", "unknown title")
    company = p.get("current_company", "unknown company")
    loc     = p.get("location", "")
    country = p.get("country", "")

    notice  = sig.get("notice_period_days", None)
    rr      = sig.get("recruiter_response_rate", 0) or 0
    otw     = sig.get("open_to_work_flag", False)
    github  = sig.get("github_activity_score", -1)
    if github is None:
        github = -1

    skill_names = [s["name"] for s in skills if s.get("proficiency") in ("advanced", "expert")][:4]
    skills_str  = ", ".join(skill_names) if skill_names else "no advanced skills listed"

    loc_str = f"{loc}, {country}" if loc else (country or "location unknown")

    behavioral_notes = []
    if not otw:
        behavioral_notes.append("not currently open to work")
    if notice is not None and notice > 60:
        behavioral_notes.append(f"{notice}-day notice period")
    if rr < 0.3:
        behavioral_notes.append(f"low recruiter response rate ({rr:.0%})")
    if 0 <= github < 15:
        behavioral_notes.append("low GitHub activity")

    disq = []
    if feat_row.get("all_consulting", False):
        disq.append("career spent entirely at consulting/IT-services firms")
    if feat_row.get("wrong_domain", False):
        disq.append("background primarily in CV/speech rather than NLP/IR")

    if rank <= 20:
        sent1 = (
            f"{yoe:.0f}-year {title} at {company} ({loc_str}); "
            f"advanced skills in {skills_str}."
        )
    else:
        sent1 = (
            f"{yoe:.0f} years experience as {title} at {company}; "
            f"skills include {skills_str}."
        )

    concerns = behavioral_notes + disq
    if concerns:
        sent2 = f"Concerns: {'; '.join(concerns)}."
    elif rank <= 10:
        github_note = f"GitHub score {github:.0f}/100." if github >= 0 else ""
        sent2 = f"Strong engagement signals — open to work, responsive. {github_note}".strip()
    else:
        weak_reason = "location" if feat_row.get("location_score", 0) < 0.5 else "weaker semantic match"
        sent2 = f"Reasonable fit on skills and experience; ranked lower due to {weak_reason}."

    return f"{sent1} {sent2}".strip()

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "submission.csv"))
    args = parser.parse_args()

    df, embeddings, jd_embedding, jd = load_artifacts()
    weights = jd["score_weights"]

    print("Computing semantic similarity scores...")
    semantic_scores = compute_semantic_scores(embeddings, jd_embedding)

    print("Computing final composite scores...")
    final_scores = compute_final_scores(df, semantic_scores, weights)
    df["final_score"] = final_scores

    n_honeypots_in_pool = int((df["disq_penalty"] == 0.0).sum())
    print(f"\nHoneypots/fictional companies excluded from ranking entirely: {n_honeypots_in_pool}")

    df_sorted = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    # Safety check: confirm no zero-score (honeypot) candidates leaked into top 100
    top100 = df_sorted.head(100).copy()
    leaked = int((top100["final_score"] <= 0.0).sum())
    if leaked > 0:
        print(f"⚠️  WARNING: {leaked} zero-score candidates leaked into top 100 — "
              f"investigate disq_reason column in features.parquet.")

    top100["rank"] = range(1, 101)

    print(f"\nTop 5 candidates:")
    for _, row in top100.head(5).iterrows():
        print(f"  Rank {int(row['rank'])}: {row['candidate_id']}  score={row['final_score']:.4f}")

    print("\nLoading top-100 candidate profiles for reasoning generation...")
    top_ids  = top100["candidate_id"].tolist()
    profiles = load_top_candidates(top_ids)
    print(f"  Found {len(profiles)} of {len(top_ids)} profiles.")

    rows = []
    for _, row in top100.iterrows():
        cid      = row["candidate_id"]
        rank     = int(row["rank"])
        score    = round(float(row["final_score"]), 6)
        feat_row = row.to_dict()

        c         = profiles.get(cid, {})
        reasoning = generate_reasoning(rank, c, feat_row)

        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        score,
            "reasoning":    reasoning,
        })

    out_df = pd.DataFrame(rows)[["candidate_id", "rank", "score", "reasoning"]]
    out_df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"\n✅ Submission written → {args.out}")
    print(f"   Rows: {len(out_df)}")
    print(f"   Score range: {out_df['score'].min():.4f} – {out_df['score'].max():.4f}")
    print(f"\nNext step: python data/validate_submission.py <renamed_to_team_id>.csv data/candidates.jsonl")

if __name__ == "__main__":
    main()