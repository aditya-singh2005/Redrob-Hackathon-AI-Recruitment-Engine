# Redrob Hackathon — Intelligent Candidate Ranking System

**Team:** Mysterio  
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Approach:** Hybrid weighted scoring with sentence-transformer embeddings

---

## What this system does

Ranks 100,000 candidates against a Senior AI Engineer job description using a
two-phase pipeline:

**Phase 1 — Offline pre-computation** (run once, can use GPU):
- Parses the JD into structured requirements with weighted skill categories
- Embeds all 100K candidate profiles using `all-MiniLM-L6-v2`
- Computes 6 feature scores per candidate: skill match, career quality,
  location, education, behavioral multiplier, and disqualifier penalty
- Saves pre-computed artifacts to disk

**Phase 2 — Fast ranking** (CPU only, under 60 seconds):
- Loads pre-computed features and embeddings
- Computes semantic similarity via a single matrix multiply (dot product)
- Combines all scores with tuned weights into a final composite score
- Outputs top-100 ranked candidates with fact-grounded reasoning

---

## Scoring architecture

```
final_score = (
    0.20 × semantic_similarity     # embedding cosine sim vs JD vector
  + 0.30 × career_quality          # YoE in range, product company, title fit
  + 0.25 × skill_match             # must-have + nice-to-have skill coverage
  + 0.15 × location                # India + preferred city bonus
  + 0.10 × education               # institution tier + field relevance
) × behavioral_multiplier          # 0.4–1.0: recency, response rate, notice
  × disqualifier_penalty           # 0.0 for honeypots, 0.4 for consulting-only
```

**Key design decisions:**
- Career quality outweighs raw semantic similarity, because the JD explicitly
  warns against keyword-stuffing. A candidate whose career description shows
  shipped retrieval systems ranks higher than one whose skills list has all
  the right words but whose title is "Marketing Manager."
- Behavioral signals are a multiplier, not an additive bonus — a
  perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
  recruiter response rate is, for hiring purposes, not actually available.
- Honeypot detection uses three rules: impossible employment timelines,
  expert-level skills with zero months of usage, and YoE that exceeds what
  the career history supports. The fictional-company names scattered across
  the dataset (~78% of candidates) are dataset noise attached to irrelevant
  profiles (Marketing Managers, Accountants), not a meaningful signal —
  our scoring naturally buries these without special-casing.

---

## Repo structure

```
redrob-hackathon/
├── scripts/
│   ├── 01_parse_jd.py      # JD → structured requirements JSON (run once)
│   ├── 02_feature_gen.py   # 100K candidates → features.parquet + embeddings.npy
│   └── 03_rank.py          # fast ranking → submission CSV (CPU only, <60s)
├── requirements.txt
├── README.md
└── submission_metadata.yaml
```

---

## Setup

```bash
git clone https://github.com/aditya-singh2005/Redrob-Hackathon-AI-Recruitment-Engine
cd redrob-hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place `candidates.jsonl` (the 100K candidate pool) in a `data/` folder:

```
redrob-hackathon/
└── data/
    └── candidates.jsonl
```

---

## Reproducing the submission

### Step 1 — Parse JD (instant)
```bash
python scripts/01_parse_jd.py
```
Output: `artifacts/jd_requirements.json`

### Step 2 — Generate features and embeddings (slow — run once)
```bash
python scripts/02_feature_gen.py
```
Output: `artifacts/features.parquet`, `artifacts/embeddings.npy`,
`artifacts/jd_embedding.npy`

This step embeds 100K profiles using `all-MiniLM-L6-v2`. On CPU this takes
~5 hours. On a GPU (e.g. Colab T4) it takes ~5 minutes. This step may exceed
the 5-minute wall-clock limit — it is pre-computation, not the ranking step.

### Step 3 — Rank (fast — CPU only, under 60 seconds)
```bash
python scripts/03_rank.py --out outputs/Mysterio.csv
```

This is the ranking step that must complete within the 5-minute compute
budget. It loads pre-computed artifacts, does a single matrix multiply for
semantic similarity, applies weighted scoring, and writes the CSV.
No GPU. No API calls. No network.

**Single command for Stage 3 reproduction** (assumes artifacts already exist):
```bash
python scripts/03_rank.py --candidates data/candidates.jsonl --out outputs/Mysterio.csv
```

---

## Compute environment

- Pre-computation: Google Colab T4 GPU, Python 3.10
- Ranking step: CPU only, Samsung Galaxy Book 3, Intel Core i7, 16GB RAM, Windows 11, Python 3.11
- Ranking step wall-clock time: ~15 seconds

---

## Sandbox / demo

A small-sample demo (≤100 candidates) that runs end-to-end on CPU in under
5 minutes is available here:

👉 **[Run on Google Colab](https://colab.research.google.com/drive/10aLL_aFq-8Z7IfVfMysBfCK0DI4JCBA1?usp=sharing)**

The sandbox accepts a small candidate JSON upload, runs the full ranking
pipeline, and outputs a ranked CSV.

---

## Dependencies

See `requirements.txt`. Key packages:
- `sentence-transformers==3.0.1` — profile and JD embedding
- `pandas`, `pyarrow` — feature storage and manipulation
- `numpy` — matrix operations for scoring
- `scikit-learn` — utilities
- `tqdm` — progress bars

---

## AI tools used

Claude (Anthropic) was used as a development assistant throughout — for
architecture design, debugging, and code review. All engineering decisions,
debugging sessions, and architectural choices were made by the team with
AI assistance. Declared honestly per hackathon rules.

---

## Methodology summary (≤200 words)

We built a two-phase hybrid ranking system. In the offline phase, we parse
the JD into structured requirements (must-have skills with weights, location
preferences, behavioral thresholds, disqualifier rules) and embed all 100K
candidate profiles using `all-MiniLM-L6-v2`. The online ranking phase loads
these pre-computed artifacts and produces a composite score in under 60
seconds on CPU.

The scoring formula combines semantic similarity (cosine distance between
candidate and JD embeddings), career quality (years of experience in range,
product vs consulting company background, title relevance), skill match
(weighted alias matching against must-have and nice-to-have skills), location
fit (India + preferred city bonus), and education. These five dimensions are
multiplied by a behavioral signal (recency, recruiter response rate, notice
period, GitHub activity) and a disqualifier penalty (honeypot detection via
impossible timelines and zero-experience expert claims).

Career quality is weighted highest because the JD explicitly warns against
keyword-stuffing — a candidate who shipped a retrieval system at a product
company is more valuable than one whose skill list matches perfectly but
whose career shows no production deployment. Behavioral signals are applied
as a multiplier rather than additive, so engagement quality modulates the
entire fit score rather than compensating for poor skill match.