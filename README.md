<h1 align="center">🕵️ Mysterio - AI Recruitment Engine</h1>
<p align="center"><b>Redrob Hackathon · Data & AI Challenge</b></p>

<p align="center">
  <img src="Image/banner.png" alt="Mysterio — Redrob Candidate Ranking Engine" width="100%">
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Team-Mysterio-6f42c1?style=for-the-badge" alt="Team">
  <img src="https://img.shields.io/badge/Candidates-100K-blue?style=for-the-badge" alt="Candidates">
  <img src="https://img.shields.io/badge/Rank%20Time-%3C60s-brightgreen?style=for-the-badge" alt="Rank Time">
  <img src="https://img.shields.io/badge/Python-3.10%2F3.11-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Model-all--MiniLM--L6--v2-orange?style=for-the-badge" alt="Model">
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/10aLL_aFq-8Z7IfVfMysBfCK0DI4JCBA1?usp=sharing"><img src="https://img.shields.io/badge/Colab-Live%20Demo-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Colab Demo"></a>
  <a href="https://github.com/aditya-singh2005/Redrob-Hackathon-AI-Recruitment-Engine/blob/main/outputs/Mysterio.csv"><img src="https://img.shields.io/badge/Output-Mysterio.csv-informational?style=flat-square&logo=googlesheets&logoColor=white" alt="Output CSV"></a>
  <a href="https://drive.google.com/file/d/1Y2rXfWJGSqexjwzDxn1AE8VCdjdtCe7E/view"><img src="https://img.shields.io/badge/Architecture-Diagram-4285F4?style=flat-square&logo=googledrive&logoColor=white" alt="Architecture Diagram"></a>
</p>

---

## 🔗 Quick Links

| 🔗 Resource | 📄 Description |
|---|---|
| ▶️ [**Colab Demo**](https://colab.research.google.com/drive/10aLL_aFq-8Z7IfVfMysBfCK0DI4JCBA1?usp=sharing) | Run the full pipeline end-to-end on a small sample, no setup needed |
| 📊 [**Submission Output**](https://github.com/aditya-singh2005/Redrob-Hackathon-AI-Recruitment-Engine/blob/main/outputs/Mysterio.csv) | Final ranked `Mysterio.csv` |
| 🧭 [**Architecture Diagram**](https://drive.google.com/file/d/1Y2rXfWJGSqexjwzDxn1AE8VCdjdtCe7E/view) | Visual walkthrough of the two-phase pipeline |

---

## 🚀 What This System Does

Mysterio ranks **100,000 candidates** against a Senior AI Engineer job description through a **two-phase pipeline** — heavy lifting done offline, ranking done instantly.

### 🧠 Phase 1 — Offline Pre-computation *(run once, GPU optional)*
- 📝 Parses the JD into structured requirements with weighted skill categories
- 🧬 Embeds all 100K candidate profiles using `all-MiniLM-L6-v2`
- 📐 Computes 6 feature scores per candidate — skill match, career quality, location, education, behavioral multiplier, disqualifier penalty
- 💾 Saves pre-computed artifacts to disk

### ⚡ Phase 2 — Fast Ranking *(CPU only, under 60 seconds)*
- 📂 Loads pre-computed features and embeddings
- ➗ Computes semantic similarity via a single matrix multiply
- 🎯 Combines all scores with tuned weights into a final composite score
- 🏆 Outputs top-100 ranked candidates with fact-grounded reasoning

---

## 🏗️ Scoring Architecture

<p align="center">
  <a href="https://drive.google.com/file/d/1Y2rXfWJGSqexjwzDxn1AE8VCdjdtCe7E/view">🧭 <u> View System Architecture </u></a>
</p>

```text
final_score = (
    0.20 × semantic_similarity     # embedding cosine sim vs JD vector
  + 0.30 × career_quality          # YoE in range, product company, title fit
  + 0.25 × skill_match             # must-have + nice-to-have skill coverage
  + 0.15 × location                # India + preferred city bonus
  + 0.10 × education               # institution tier + field relevance
) × behavioral_multiplier          # 0.4–1.0: recency, response rate, notice
  × disqualifier_penalty           # 0.0 for honeypots, 0.4 for consulting-only
```

### 🎯 Key Design Decisions

- **🏆 Career quality > raw similarity** — the JD explicitly warns against keyword-stuffing. A candidate whose career shows shipped retrieval systems ranks above one whose skills list has all the right words but whose title is "Marketing Manager."
- **✖️ Behavioral signals are a multiplier, not a bonus** — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available.
- **🍯 Honeypot detection** uses three rules: impossible employment timelines, expert-level skills with zero months of usage, and YoE exceeding what the career history supports. Fictional-company names scattered across ~78% of candidates are dataset noise attached to irrelevant profiles (Marketing Managers, Accountants) — our scoring naturally buries these without special-casing.

---

## 📁 Repo Structure

```text
redrob-hackathon/
├── scripts/
│   ├── 01_parse_jd.py      # JD → structured requirements JSON (run once)
│   ├── 02_feature_gen.py   # 100K candidates → features.parquet + embeddings.npy
│   └── 03_rank.py          # fast ranking → submission CSV (CPU only, <60s)
├── outputs/
│   └── Mysterio.csv        # 🏆 final ranked submission
├── images/
│   └── banner.png
├── requirements.txt
├── README.md
└── submission_metadata.yaml
```

---

## ⚙️ Setup

```bash
git clone https://github.com/aditya-singh2005/Redrob-Hackathon-AI-Recruitment-Engine
cd redrob-hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place `candidates.jsonl` (the 100K candidate pool) in a `data/` folder:

```text
redrob-hackathon/
└── data/
    └── candidates.jsonl
```

---

## 🔁 Reproducing the Submission

### 1️⃣ Parse JD *(instant)*
```bash
python scripts/01_parse_jd.py
```
📤 Output: `artifacts/jd_requirements.json`

### 2️⃣ Generate Features & Embeddings *(slow — run once)*
```bash
python scripts/02_feature_gen.py
```
📤 Output: `artifacts/features.parquet`, `artifacts/embeddings.npy`, `artifacts/jd_embedding.npy`

> ⏱️ Embeds 100K profiles with `all-MiniLM-L6-v2`. ~5 hours on CPU, ~5 minutes on a GPU (e.g. Colab T4). This step may exceed the 5-minute wall-clock limit — it's pre-computation, not the ranking step.

### 3️⃣ Rank *(fast — CPU only, under 60 seconds)*
```bash
python scripts/03_rank.py --out outputs/Mysterio.csv
```

✅ This is the step that must complete within the 5-minute compute budget. It loads pre-computed artifacts, does a single matrix multiply for semantic similarity, applies weighted scoring, and writes the CSV. **No GPU. No API calls. No network.**

**One-command Stage 3 reproduction** *(assumes artifacts already exist)*:
```bash
python scripts/03_rank.py --candidates data/candidates.jsonl --out outputs/Mysterio.csv
```

---

## 🖥️ Compute Environment

| | |
|---|---|
| 🔧 Pre-computation | Google Colab T4 GPU, Python 3.10 |
| 💻 Ranking step | CPU only — Samsung Galaxy Book 3, Intel Core i7, 16GB RAM, Windows 11, Python 3.11 |
| ⏱️ Ranking wall-clock | ~15 seconds |

---

## 🎮 Sandbox / Demo

A small-sample demo (≤100 candidates) that runs end-to-end on CPU in under 5 minutes:
<div align="center">
  <a href="https://colab.research.google.com/drive/10aLL_aFq-8Z7IfVfMysBfCK0DI4JCBA1?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</div>


👉 **[Run on Google Colab](https://colab.research.google.com/drive/10aLL_aFq-8Z7IfVfMysBfCK0DI4JCBA1?usp=sharing)** — just run once. The sandbox accepts a small candidate JSON upload, runs the full ranking pipeline, and outputs a ranked CSV.

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt). Key packages:

| Package | Purpose |
|---|---|
| 🔤 `sentence-transformers==3.0.1` | Profile and JD embedding |
| 🐼 `pandas`, `pyarrow` | Feature storage and manipulation |
| 🔢 `numpy` | Matrix operations for scoring |
| 🧪 `scikit-learn` | Utilities |
| 📊 `tqdm` | Progress bars |

---

## 🤖 AI Tools Used

Claude (Anthropic) was used as a development assistant throughout — for architecture design, debugging, and code review. All engineering decisions, debugging sessions, and architectural choices were made by the team with AI assistance. Declared honestly per hackathon rules.

---

## 📝 Methodology Summary *(≤200 words)*

We built a two-phase hybrid ranking system. In the offline phase, we parse the JD into structured requirements (must-have skills with weights, location preferences, behavioral thresholds, disqualifier rules) and embed all 100K candidate profiles using `all-MiniLM-L6-v2`. The online ranking phase loads these pre-computed artifacts and produces a composite score in under 60 seconds on CPU.

The scoring formula combines semantic similarity (cosine distance between candidate and JD embeddings), career quality (years of experience in range, product vs. consulting company background, title relevance), skill match (weighted alias matching against must-have and nice-to-have skills), location fit (India + preferred city bonus), and education. These five dimensions are multiplied by a behavioral signal (recency, recruiter response rate, notice period, GitHub activity) and a disqualifier penalty (honeypot detection via impossible timelines and zero-experience expert claims).

Career quality is weighted highest because the JD explicitly warns against keyword-stuffing — a candidate who shipped a retrieval system at a product company is more valuable than one whose skill list matches perfectly but whose career shows no production deployment. Behavioral signals are applied as a multiplier rather than additive, so engagement quality modulates the entire fit score rather than compensating for poor skill match.

---

<p align="center">Built with ❤️ by Aditya Singh (<b>Team Mysterio</b>)</p>
