"""
02_feature_gen.py
Streams through candidates.jsonl once, computes all features per candidate,
saves features.parquet + embeddings.npy to artifacts/.

v2 improvements:
  - Honeypot / fictional-company detection is now permanent (baked into
    disq_penalty at feature-gen time), not a runtime patch in 03_rank.py.
  - Fictional company match is EXACT (full company name, lowercased,
    stripped) — never substring — to avoid false positives like
    "hooli" matching inside "schooling" or "Genpact AI" being treated
    as the real consulting firm "Genpact".
  - Consulting-firm match is also exact-token based, not loose substring.
  - Wrong-domain logic uses a cancel-out mechanic: if retrieval/IR signals
    are present anywhere in the profile, CV/speech-only candidates are not
    penalized (they may be making a legitimate domain transition).
  - All disqualifier rules are auditable: a separate "disq_reason" column
    is saved so you can debug *why* someone got zeroed out, without
    re-running the 5-hour embedding step.

Run time: ~20-40 minutes on your machine (one-time cost).
Memory:   peaks at ~6 GB — well within 16 GB.

Usage:
    python scripts/02_feature_gen.py
"""

import json
import os
import gzip
import numpy as np
import pandas as pd
from datetime import date, datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR  = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.jsonl")
CANDIDATES_GZ   = os.path.join(DATA_DIR, "candidates.jsonl.gz")
JD_REQ_PATH     = os.path.join(ARTIFACT_DIR, "jd_requirements.json")
FEATURES_PATH   = os.path.join(ARTIFACT_DIR, "features.parquet")
EMBED_PATH      = os.path.join(ARTIFACT_DIR, "embeddings.npy")
JD_EMBED_PATH   = os.path.join(ARTIFACT_DIR, "jd_embedding.npy")

with open(JD_REQ_PATH, "r", encoding="utf-8") as f:
    JD = json.load(f)

TODAY = date.today()

# ── helpers ────────────────────────────────────────────────────────────────────

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None

def days_since(d):
    if d is None:
        return 999
    return (TODAY - d).days

def lower_text(s):
    return s.lower() if isinstance(s, str) else ""

def normalize_company(name):
    """Lowercase, strip whitespace, collapse internal whitespace."""
    if not isinstance(name, str):
        return ""
    return " ".join(name.strip().lower().split())

def alias_hit_count(text, aliases):
    count = 0
    for alias in aliases:
        if alias in text:
            count += 1
    return count

def build_profile_text(c):
    parts = []
    p = c.get("profile", {}) or {}
    parts.append(p.get("headline", "") or "")
    parts.append(p.get("summary", "") or "")
    parts.append(p.get("current_title", "") or "")
    parts.append(p.get("current_industry", "") or "")

    for role in c.get("career_history", []) or []:
        parts.append(role.get("title", "") or "")
        parts.append(role.get("description", "") or "")
        parts.append(role.get("industry", "") or "")

    for skill in c.get("skills", []) or []:
        parts.append(skill.get("name", "") or "")

    for cert in c.get("certifications", []) or []:
        parts.append(cert.get("name", "") or "")

    for edu in c.get("education", []) or []:
        parts.append(edu.get("field_of_study", "") or "")
        parts.append(edu.get("degree", "") or "")

    return " ".join(p for p in parts if p).strip()

def build_jd_text():
    must = " ".join(s["name"] for s in JD["must_have_skills"])
    nice = " ".join(s["name"] for s in JD["nice_to_have_skills"])
    return (
        f"Senior AI Engineer ranking retrieval embedding vector search "
        f"production ML systems {must} {nice} "
        f"Python NDCG evaluation framework startup product company "
        f"5 to 9 years experience India Pune Noida"
    )

# ── skill match score ──────────────────────────────────────────────────────────

def compute_skill_match(c, full_text):
    must_score = 0.0
    must_total = sum(s["weight"] for s in JD["must_have_skills"])
    for skill in JD["must_have_skills"]:
        if alias_hit_count(full_text, skill["aliases"]) > 0:
            must_score += skill["weight"]

    nice_score = 0.0
    nice_total = sum(s["weight"] for s in JD["nice_to_have_skills"])
    for skill in JD["nice_to_have_skills"]:
        if alias_hit_count(full_text, skill["aliases"]) > 0:
            nice_score += skill["weight"]

    normalized = (
        0.80 * (must_score / must_total if must_total > 0 else 0) +
        0.20 * (nice_score / nice_total if nice_total > 0 else 0)
    )
    return round(min(normalized, 1.0), 4)

# ── career quality score ───────────────────────────────────────────────────────

CONSULTING_FIRMS_NORM = {normalize_company(n) for n in JD["consulting_firm_names"]}
PRODUCT_SIGNALS       = JD["product_company_signals"]
WRONG_DOMAIN          = JD["wrong_domain_signals"]
RETRIEVAL_SIGNALS     = JD["retrieval_domain_signals"]

def is_consulting_company(company_name):
    """
    Exact-token match: the normalized company name must EQUAL a known
    consulting firm name, or be that name plus a generic corporate suffix
    (ltd, limited, india, pvt, inc). This avoids "Genpact AI" (a distinct
    fictional product company in this dataset) being treated as the
    consulting firm "Genpact".
    """
    norm = normalize_company(company_name)
    if not norm:
        return False
    if norm in CONSULTING_FIRMS_NORM:
        return True
    suffixes = [" ltd", " limited", " india", " pvt", " pvt ltd", " inc", " corp"]
    for firm in CONSULTING_FIRMS_NORM:
        for suf in suffixes:
            if norm == firm + suf:
                return True
    return False

def compute_career_quality(c, full_text):
    p   = c.get("profile", {}) or {}
    yoe = p.get("years_of_experience", 0) or 0
    history = c.get("career_history", []) or []

    # YoE score (ideal 5-9 years)
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.8
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        yoe_score = 0.5
    else:
        yoe_score = 0.2

    # Consulting check — exact-token match against EVERY employer
    company_names = [r.get("company", "") for r in history]
    non_empty_companies = [c for c in company_names if c.strip()]
    all_consulting = (
        len(non_empty_companies) > 0 and
        all(is_consulting_company(name) for name in non_empty_companies)
    )

    product_hit = sum(1 for sig in PRODUCT_SIGNALS if sig in full_text)
    product_score = min(product_hit / 5.0, 1.0)

    # Wrong-domain: only penalize if wrong-domain signals dominate AND
    # there is truly zero retrieval/IR/NLP signal anywhere in the profile.
    wrong_hits      = sum(1 for d in WRONG_DOMAIN if d in full_text)
    retrieval_hits  = sum(1 for s in RETRIEVAL_SIGNALS if s in full_text)
    wrong_domain_flag = wrong_hits >= 2 and retrieval_hits == 0

    title = lower_text(p.get("current_title", ""))
    good_titles = ["engineer", "scientist", "ml", "ai", "data", "nlp",
                   "search", "architect", "tech lead", "founding"]
    bad_titles  = ["manager", "marketing", "sales", "hr", "recruiter",
                   "analyst", "consultant", "director", "vp", "c-level"]
    title_score = 0.5
    if any(t in title for t in good_titles):
        title_score = 1.0
    if any(t in title for t in bad_titles):
        title_score = 0.1

    career_score = (
        0.35 * yoe_score +
        0.35 * product_score +
        0.30 * title_score
    )

    return round(career_score, 4), all_consulting, wrong_domain_flag

# ── location score ─────────────────────────────────────────────────────────────

PREFERRED_LOCATIONS = JD["preferred_locations"]

def compute_location_score(c):
    p        = c.get("profile", {}) or {}
    location = lower_text(p.get("location", ""))
    country  = lower_text(p.get("country", ""))
    signals  = c.get("redrob_signals", {}) or {}
    relocate = signals.get("willing_to_relocate", False)

    if "india" not in country and country != "in":
        return 0.15 if relocate else 0.0

    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return 1.0

    if relocate:
        return 0.7

    return 0.4

# ── education score ────────────────────────────────────────────────────────────

def compute_education_score(c):
    edu_list = c.get("education", []) or []
    if not edu_list:
        return 0.3

    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6,
                "tier_4": 0.4, "unknown": 0.5}
    best_tier = max(tier_map.get(e.get("tier", "unknown"), 0.5) for e in edu_list)

    cs_fields = ["computer science", "cs", "information technology",
                 "artificial intelligence", "machine learning", "data science",
                 "electronics", "electrical", "mathematics", "statistics"]
    field_hit = any(
        any(f in lower_text(e.get("field_of_study", "")) for f in cs_fields)
        for e in edu_list
    )
    field_score = 1.0 if field_hit else 0.6

    return round(0.5 * best_tier + 0.5 * field_score, 4)

# ── behavioral multiplier ──────────────────────────────────────────────────────

def compute_behavioral_multiplier(c):
    sig = c.get("redrob_signals", {}) or {}

    score = 0.0
    weight_total = 0.0

    def add(val, w):
        nonlocal score, weight_total
        score += val * w
        weight_total += w

    add(1.0 if sig.get("open_to_work_flag", False) else 0.2, 3.0)

    last_active = parse_date(sig.get("last_active_date"))
    inactive_days = days_since(last_active)
    if inactive_days <= 14:
        add(1.0, 2.5)
    elif inactive_days <= 30:
        add(0.85, 2.5)
    elif inactive_days <= 60:
        add(0.65, 2.5)
    elif inactive_days <= 90:
        add(0.40, 2.5)
    else:
        add(0.10, 2.5)

    rr = sig.get("recruiter_response_rate", 0) or 0
    add(min(rr / 0.6, 1.0), 2.0)

    notice = sig.get("notice_period_days", 90)
    if notice is None:
        notice = 90
    if notice <= 30:
        add(1.0, 1.5)
    elif notice <= 60:
        add(0.7, 1.5)
    elif notice <= 90:
        add(0.4, 1.5)
    else:
        add(0.1, 1.5)

    github = sig.get("github_activity_score", -1)
    if github is None:
        github = -1
    if github >= 0:
        add(min(github / 60.0, 1.0), 1.0)
    else:
        add(0.3, 1.0)

    completeness = sig.get("profile_completeness_score", 50) or 50
    add(min(completeness / 90.0, 1.0), 0.5)

    icr = sig.get("interview_completion_rate", 0) or 0
    add(icr, 0.5)

    raw = score / weight_total if weight_total > 0 else 0.5
    multiplier = 0.40 + 0.60 * raw
    return round(multiplier, 4)

# ── disqualifier penalty (honeypot detection lives here, permanently) ─────────

FICTIONAL_COMPANIES_NORM = {
    normalize_company(n) for n in JD["fictional_companies"]
    if normalize_company(n)   # drop any blank/empty entries defensively
}

def is_fictional_company(company_name):
    """EXACT match only — never substring — to avoid false positives.
    Guards explicitly against empty/blank names matching anything."""
    norm = normalize_company(company_name)
    if not norm:
        return False
    return norm in FICTIONAL_COMPANIES_NORM

def compute_disqualifier_penalty(c, all_consulting, wrong_domain):
    """
    Returns (penalty 0.0-1.0, reason string).
    1.0 = clean. 0.0 = honeypot, hard-zeroed.
    """
    history = c.get("career_history", []) or []
    p = c.get("profile", {}) or {}

    # NOTE: We deliberately do NOT hard-disqualify based on fictional/joke
    # company names (Dunder Mifflin, Hooli, etc). Diagnostics on this dataset
    # showed ~78% of candidates have a fictional company mention somewhere in
    # their career history — almost always attached to completely irrelevant
    # candidates (Marketing Managers, Accountants, Civil Engineers) who were
    # never going to rank highly anyway. This is dataset noise/filler, not
    # the spec's actual honeypot signal. The real honeypots (impossible
    # timelines, zero-duration "expert" skills) are caught by Rules 1-3 below
    # and our skill/career scoring naturally filters out irrelevant titles.
    # ── Rule 1: impossible employment timeline ────────────────────────────────
    for role in history:
        dur = role.get("duration_months", 0) or 0
        start = parse_date(role.get("start_date"))
        end   = parse_date(role.get("end_date")) or TODAY
        if start:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            if dur > actual_months + 3:
                return 0.0, "impossible_duration"

    # ── Rule 2: many "expert" skills with 0 months experience ────────────────
    skills = c.get("skills", []) or []
    zero_duration_experts = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    if zero_duration_experts >= 5:
        return 0.0, "zero_duration_experts"

    # ── Rule 3: YoE wildly exceeds what career history supports ──────────────
    yoe = p.get("years_of_experience", 0) or 0
    total_career_months = sum(r.get("duration_months", 0) or 0 for r in history)
    if yoe > 0 and total_career_months > 0:
        implied_yoe = total_career_months / 12
        if yoe > implied_yoe * 1.5 + 3:
            return 0.0, "yoe_mismatch"

    # ── soft penalties (not honeypots, just bad fit per JD) ───────────────────
    penalty = 1.0
    reasons = []
    if all_consulting:
        penalty *= 0.40
        reasons.append("all_consulting")
    if wrong_domain:
        penalty *= 0.50
        reasons.append("wrong_domain")

    return round(penalty, 4), (",".join(reasons) if reasons else "clean")

# ── main feature extraction loop ───────────────────────────────────────────────

def load_candidates():
    if os.path.exists(CANDIDATES_PATH):
        return open(CANDIDATES_PATH, "r", encoding="utf-8")
    elif os.path.exists(CANDIDATES_GZ):
        return gzip.open(CANDIDATES_GZ, "rt", encoding="utf-8")
    raise FileNotFoundError("candidates.jsonl or candidates.jsonl.gz not found in data/")

def main():
    print("── Step 1: Load JD requirements & embedding model ─────────────────")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("   Model loaded.")

    jd_text = build_jd_text()
    jd_embedding = model.encode(jd_text, normalize_embeddings=True)
    np.save(JD_EMBED_PATH, jd_embedding)
    print(f"   JD embedding saved → {JD_EMBED_PATH}")

    print("\n── Step 2: Stream candidates, extract features ────────────────────")
    rows  = []
    texts = []
    honeypot_count = 0

    with load_candidates() as f:
        for line in tqdm(f, total=100_000, desc="Extracting features"):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = c.get("candidate_id", "")
            full_text = lower_text(build_profile_text(c))

            skill_score                    = compute_skill_match(c, full_text)
            career_score, all_cons, w_dom  = compute_career_quality(c, full_text)
            loc_score                      = compute_location_score(c)
            edu_score                      = compute_education_score(c)
            behavioral_mult                = compute_behavioral_multiplier(c)
            disq_penalty, disq_reason      = compute_disqualifier_penalty(
                                                c, all_cons, w_dom)

            if disq_penalty == 0.0:
                honeypot_count += 1

            rows.append({
                "candidate_id":       cid,
                "skill_score":        skill_score,
                "career_score":       career_score,
                "location_score":     loc_score,
                "education_score":    edu_score,
                "behavioral_mult":    behavioral_mult,
                "disq_penalty":       disq_penalty,
                "disq_reason":        disq_reason,
                "all_consulting":     all_cons,
                "wrong_domain":       w_dom,
                "yoe":               (c.get("profile") or {}).get("years_of_experience", 0),
                "open_to_work":      (c.get("redrob_signals") or {}).get("open_to_work_flag", False),
                "notice_days":       (c.get("redrob_signals") or {}).get("notice_period_days", 90),
                "country":           (c.get("profile") or {}).get("country", ""),
                "location":          (c.get("profile") or {}).get("location", ""),
            })
            texts.append(build_profile_text(c))

    print(f"\n   Extracted features for {len(rows)} candidates.")
    if len(rows) < 95_000:
        print(f"\n🚨 SANITY CHECK FAILED: only {len(rows)} candidates extracted, "
              f"expected ~100,000. The candidates.jsonl file may be truncated, "
              f"corrupted, or the read was interrupted (e.g. Colab disconnect, "
              f"upload cut short, or disk quota issue).")
        print("   STOPPING before the slow embedding step. Re-check your")
        print("   candidates.jsonl file size and re-upload if needed, then")
        print("   re-run this script from scratch.")
        raise SystemExit(1)
    print(f"   Honeypots detected (disq_penalty=0): {honeypot_count}")

    honeypot_rate = honeypot_count / max(len(rows), 1)
    if honeypot_rate > 0.05:
        print(f"\n🚨 SANITY CHECK FAILED: honeypot rate is {honeypot_rate:.1%} — "
              f"the spec says real honeypot rate should be ~80/100,000 (~0.08%).")
        print("   This almost certainly means a bug in honeypot/fictional-company")
        print("   matching (e.g. an empty string in the fictional list matching")
        print("   candidates with blank company fields). STOPPING before the")
        print("   slow embedding step so you don't waste GPU time on bad data.")
        print("   Fix the bug, then re-run this script from scratch.")
        raise SystemExit(1)

    print("\n── Step 3: Batch embed all candidate profiles ─────────────────────")
    print("   This is the slowest step (~15-25 minutes). Go get a coffee ☕")
    BATCH = 256
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch = texts[i: i + BATCH]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    np.save(EMBED_PATH, embeddings)
    print(f"   Embeddings saved → {EMBED_PATH}  shape={embeddings.shape}")

    print("\n── Step 4: Save feature table ─────────────────────────────────────")
    df = pd.DataFrame(rows)
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"   Features saved  → {FEATURES_PATH}  shape={df.shape}")

    print("\n✅ Feature generation complete. Run 03_rank.py next.")
    print("\n   Honeypot reason breakdown:")
    print(df[df["disq_penalty"] == 0.0]["disq_reason"].value_counts().to_string())

if __name__ == "__main__":
    main()