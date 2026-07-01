"""
01_parse_jd.py
Converts the job description into a structured JSON requirements file.
Run once. Output: artifacts/jd_requirements.json

v2: Tuned weights based on JD re-read. Career quality and skill match
matter more than raw semantic similarity for this JD, because the JD
explicitly warns against "keyword stuffer" candidates winning on semantic
similarity alone — career narrative and actual fit matter more.
Also includes a curated fictional-company list for honeypot detection,
used directly inside feature_gen so it's baked into the artifacts
(not a runtime patch that's easy to forget to apply).
"""

import json
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

JD_REQUIREMENTS = {

    "role_title": "Senior AI Engineer",
    "company":    "Redrob AI",
    "yoe_min":    5,
    "yoe_max":    9,

    "preferred_locations": [
        "pune", "noida", "delhi", "delhi ncr", "gurugram", "gurgaon",
        "hyderabad", "mumbai", "bengaluru", "bangalore"
    ],
    "preferred_country": "india",

    "notice_ideal_days":  30,
    "notice_max_days":    90,

    "must_have_skills": [
        {
            "name": "embeddings / semantic retrieval",
            "aliases": [
                "embedding", "embeddings", "sentence-transformers",
                "sentence_transformers", "openai embeddings", "bge", "e5",
                "text embeddings", "dense retrieval", "semantic search",
                "bi-encoder", "bi encoder", "sentence transformers"
            ],
            "weight": 3.0
        },
        {
            "name": "vector database / hybrid search",
            "aliases": [
                "pinecone", "weaviate", "qdrant", "milvus", "faiss",
                "opensearch", "elasticsearch", "vector database", "vector db",
                "vector store", "hybrid search", "ann", "approximate nearest neighbour",
                "pgvector"
            ],
            "weight": 3.0
        },
        {
            "name": "ranking / retrieval systems",
            "aliases": [
                "ranking", "ranker", "retrieval", "information retrieval",
                "search ranking", "recommendation system", "recommendation systems",
                "recommender", "learning to rank", "ltr", "re-ranking", "reranking",
                "candidate retrieval", "bm25", "search backend", "search engineer"
            ],
            "weight": 3.0
        },
        {
            "name": "Python",
            "aliases": ["python", "pyspark", "pytorch", "tensorflow"],
            "weight": 1.0
        },
        {
            "name": "evaluation frameworks (NDCG/MRR/MAP)",
            "aliases": [
                "ndcg", "mrr", "map", "mean average precision",
                "a/b test", "ab test", "offline evaluation",
                "ranking evaluation", "eval framework", "evaluation framework",
                "recall@k", "precision@k"
            ],
            "weight": 1.5
        }
    ],

    "nice_to_have_skills": [
        {
            "name": "LLM fine-tuning",
            "aliases": [
                "fine-tuning", "fine-tuning llms", "finetuning", "lora", "qlora", "peft",
                "instruction tuning", "rlhf", "fine tuned", "llm fine"
            ],
            "weight": 1.0
        },
        {
            "name": "learning to rank",
            "aliases": [
                "learning to rank", "lambdamart", "xgboost rank",
                "lightgbm rank", "neural ranker", "pointwise", "pairwise", "listwise"
            ],
            "weight": 1.0
        },
        {
            "name": "LLMs / generative AI",
            "aliases": [
                "llm", "llms", "large language model", "gpt", "claude", "gemini",
                "rag", "retrieval augmented", "langchain", "llamaindex",
                "transformer", "bert", "roberta", "t5", "hugging face", "haystack"
            ],
            "weight": 0.8
        },
        {
            "name": "distributed systems / scale",
            "aliases": [
                "distributed", "kafka", "spark", "airflow", "kubernetes",
                "k8s", "microservices", "large-scale", "high throughput", "kubeflow",
                "mlops", "mlflow"
            ],
            "weight": 0.6
        }
    ],

    # Exact tokenized company names for pure-consulting detection.
    # Matched against company name SPLIT INTO WORDS, not substring —
    # so "Genpact AI" (a fictional product company in this dataset, NOT
    # the real consulting firm) is not falsely flagged. We only flag if
    # the company name is essentially identical to a known consulting firm.
    "consulting_firm_names": [
        "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
        "mphasis", "hexaware", "l&t infotech", "ltimindtree", "mindtree"
    ],

    "wrong_domain_signals": [
        "computer vision", "object detection", "image segmentation",
        "image classification", "robotics", "speech recognition",
        "text-to-speech", "tts", "autonomous driving", "lidar", "slam",
        "yolo", "opencv", "asr", "automatic speech recognition"
    ],

    "retrieval_domain_signals": [
        "retrieval", "ranking", "ranker", "embedding", "embeddings",
        "search", "nlp", "recommendation", "recommender", "vector",
        "information retrieval", "semantic search", "bm25", "rag",
        "hybrid search", "re-ranking", "reranking"
    ],

    "product_company_signals": [
        "product", "saas", "platform", "marketplace", "b2b", "b2c",
        "startup", "series a", "series b", "series c", "seed",
        "launched", "shipped", "deployed to production", "real users",
        "at scale", "production system", "millions of users"
    ],

    "strong_career_keywords": [
        "end-to-end", "owned", "led", "architected", "designed",
        "built from scratch", "shipped", "launched", "drove",
        "production", "millions of users", "at scale"
    ],

    # Known fictional / honeypot company names — EXACT match only
    # (case-insensitive, whitespace-trimmed). This list is intentionally
    # narrow and literal to avoid false positives against real companies
    # that happen to share a word (e.g. "Genpact AI" must NOT match here).
    "fictional_companies": [
        "dunder mifflin", "hooli", "wayne enterprises", "stark industries",
        "umbrella corporation", "initech", "cyberdyne", "acme corp",
        "acme corporation", "pied piper", "prestige worldwide",
        "bluth company", "globo gym", "vandelay industries", "wonka industries",
        "oscorp", "globex corporation", "soylent corp", "tyrell corporation",
        "massive dynamic", "aperture science", "weyland-yutani", "los pollos hermanos"
    ],

    "behavioral": {
        "max_inactive_days":       90,
        "min_response_rate":        0.3,
        "max_notice_penalty_days": 90,
        "github_active_threshold": 20,
        "min_profile_completeness": 60
    },

    # Re-tuned: career_quality and skill_match dominate, since the JD
    # explicitly says career narrative > keyword presence. Semantic
    # similarity is kept moderate to avoid rewarding keyword-stuffed profiles.
    "score_weights": {
        "semantic_similarity": 0.20,
        "career_quality":      0.30,
        "skill_match":         0.25,
        "location":            0.15,
        "education":           0.10
    },
}

out_path = os.path.join(OUTPUT_DIR, "jd_requirements.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(JD_REQUIREMENTS, f, indent=2, ensure_ascii=False)

print(f"✅ JD requirements saved → {out_path}")
print(f"   Must-have skills : {len(JD_REQUIREMENTS['must_have_skills'])}")
print(f"   Nice-to-have     : {len(JD_REQUIREMENTS['nice_to_have_skills'])}")
print(f"   Fictional cos    : {len(JD_REQUIREMENTS['fictional_companies'])}")
print(f"   Score weights    : {JD_REQUIREMENTS['score_weights']}")