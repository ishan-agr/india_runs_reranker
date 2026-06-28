"""
JD spec — structured contract derived from job_description.docx (Senior AI Engineer,
Founding Team @ Redrob, Series A).

This module turns the prose JD into machine-usable signals for the rank-time rule
layers (Phase 5) and for the LLM-teacher prompt (Phase 2). Every block cites the JD
intent it encodes. Keyword lists are v1 — they get tuned against the contender pool
in Phase 5, not treated as final truth.

Design philosophy the JD is explicit about:
  - Title + career trajectory BEAT a keyword-stuffed skills list.
  - Product-company applied-ML experience BEATS services/consulting tenure.
  - Corroboration (title x skills x duration x endorsements) separates real from poser.
  - Behavioral signals are a MODIFIER on fit (availability/responsiveness), not fit itself.
  - "We'd rather see 10 great matches than 1000 maybes" -> precision at the top matters most.
No deps; pure stdlib so any layer (or analysis) can import it.
"""

ROLE = {
    "title": "Senior AI Engineer — Founding Team",
    "yoe_band": (5, 9),          # stated range, treated as soft
    "yoe_ideal": (6, 8),         # "between the lines" ideal
    "yoe_is_soft": True,         # JD: "a range, not a requirement"; strong other signals can override
    "stage": "Series A startup, building AI org from scratch (4 -> 12 engineers)",
    "bias": "shipper > researcher",  # JD: "tilt slightly toward shipper"
}

# --- MUST-HAVES ("Things you absolutely need"). Production experience is the key word;
#     each is a concept group, matched against career descriptions + skills, not just skill tags.
MUST_HAVES = {
    "embeddings_retrieval": [
        "embedding", "embeddings", "sentence-transformer", "sentence transformers",
        "sbert", "bge", " e5", "openai embedding", "dense retrieval", "semantic search",
        "vector search", "retrieval", "ann", "nearest neighbor", "rag",
    ],
    "vector_db_hybrid_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
        "faiss", "vespa", "pgvector", "chroma", "hybrid search", "bm25", "lucene", "solr",
    ],
    "strong_python": ["python"],
    "ranking_eval": [
        "ndcg", "mrr", " map", "mean average precision", "learning to rank", "ltr",
        "ranking", "recommendation", "recommender", "relevance", "a/b test", "ab test",
        "offline evaluation", "online evaluation", "click-through", "ctr",
    ],
}

# --- NICE-TO-HAVES ("we'd like but won't reject you for")
NICE_TO_HAVES = {
    "llm_finetune": ["fine-tune", "fine tuning", "finetune", "lora", "qlora", "peft",
                     "sft", "instruction tuning", "distillation", "rlhf", "dpo"],
    "ltr_models": ["xgboost", "lightgbm", "gbdt", "learning to rank", "neural ranker"],
    "hr_tech": ["recruiting", "recruitment", "hr-tech", "hr tech", "talent", "ats",
                "marketplace", "two-sided", "matching"],
    "distributed_inference": ["distributed", "kubernetes", "ray", "spark", "triton",
                              "onnx", "quantization", "inference optimization", "latency"],
    "open_source": ["open source", "open-source", "github", "maintainer", "contributor", "paper", "arxiv"],
}

# --- HARD DISQUALIFIERS ("disqualifiers we actually apply"). These are strong negatives.
#     Most are weak-signal in this synthetic data, so default action is heavy down-weight
#     unless clearly detectable; honeypots are handled by a separate consistency gate.
HARD_DISQUALIFIERS = {
    # Pure research / academia, no production deployment. Proxy: industry/title research-only.
    "pure_research_no_prod": {
        "industry_terms": ["research", "academia", "university", "institute", "laboratory"],
        "title_terms": ["research scientist", "phd", "postdoc", "research fellow", "lecturer", "professor"],
        "action": "downweight_strong",
        "jd": "pure research env without production deployment -> will not move forward",
    },
    # AI experience = only recent (<12mo) LangChain->OpenAI, without pre-LLM ML production.
    "langchain_only_recent": {
        "terms": ["langchain", "llamaindex", "openai api", "prompt engineering"],
        "needs_counter": "pre_llm_ml_production",  # redeemed by older ML production experience
        "action": "downweight_strong",
        "jd": "recent LangChain-calling-OpenAI only, no substantial pre-LLM ML -> probably not",
    },
    # Senior who hasn't written production code in 18 months (moved to architecture/tech-lead).
    "no_recent_code": {
        "title_terms": ["architect", "tech lead", "engineering manager", "vp ", "director", "head of"],
        "action": "downweight_moderate",
        "jd": "no production code in last 18 months -> probably not. This role writes code.",
    },
}

# --- "Things we explicitly do NOT want" -> down-weights (not auto-zero, but strong).
DO_NOT_WANT = {
    "title_chaser": {
        # Senior->Staff->Principal by switching companies every ~1.5 years; wants 3+ yr commitment.
        "max_avg_tenure_months": 18,
        "action": "downweight_moderate",
        "jd": "optimizing titles by job-hopping every 1.5y -> not a fit; want 3+ years.",
    },
    "framework_enthusiast": {
        "terms": ["langchain tutorial", "demo", "boilerplate", "hello world", "tutorial"],
        "action": "downweight_mild",
        "jd": "GitHub full of framework tutorials / demo blog posts -> not what we need.",
    },
    "consulting_only_career": {
        # Entire career at IT-services/consulting. Redeemed by ANY prior product-company role.
        "firms": [
            "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
            "capgemini", "mindtree", "ltimindtree", "lti", "hcl", "tech mahindra",
            "mphasis", "hexaware", "syntel", "l&t infotech", "deloitte", "ibm global services",
        ],
        "services_industries": ["it services", "consulting", "staffing", "outsourcing"],
        "action": "downweight_moderate",
        "redeemed_if_any_product_role": True,
        "jd": "only ever at consulting firms -> not a fit; currently-there + prior product = OK.",
    },
    "wrong_domain_primary": {
        # Primary expertise CV/speech/robotics WITHOUT significant NLP/IR exposure.
        "anti_terms": [
            "computer vision", "image classification", "object detection", "opencv",
            "cnn", "segmentation", "speech recognition", "asr", "tts", "text-to-speech",
            "robotics", "slam", "point cloud", "gan", "gans", "image generation", "ocr",
        ],
        "redeem_terms": [
            "nlp", "natural language", "information retrieval", "search", "ranking",
            "retrieval", "recommendation", "language model", "text", "embeddings",
        ],
        "action": "downweight_moderate",
        "jd": "primary CV/speech/robotics without NLP/IR -> you'd be re-learning fundamentals.",
    },
    "closedsource_no_validation": {
        # 5+ yrs entirely closed-source proprietary, no external validation.
        "action": "downweight_mild",
        "jd": "5+ yrs closed-source only, no papers/talks/OSS -> we need to see how you think.",
    },
}

# --- IDEAL candidate ("how to read between the lines") -> positive boosts.
IDEAL = {
    "applied_ml_at_product_company": "4-5 yrs applied ML/AI at product (not services) companies",
    "shipped_end_to_end_system": [
        "shipped", "deployed", "production", "launched", "built", "at scale", "real users",
        "ranking system", "search system", "recommendation system", "recommender",
    ],
    "opinionated_systems_thinker": "strong defensible opinions on retrieval/eval/LLM integration",
    "located_or_relocate": True,
    "active_on_platform": "clear signal of being in the job market / reachable",
}

# --- LOCATION preference -> mild boost/penalty (JD: Pune/Noida preferred, India strong,
#     outside India case-by-case, NO visa sponsorship).
LOCATION = {
    "preferred_cities": ["noida", "pune"],
    "welcome_cities": ["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore", "bengaluru", "ncr"],
    "strong_country": "india",
    "outside_india": "case_by_case_no_visa",  # mild penalty; not a gate
}

# --- BEHAVIORAL guidance (the redrob_signals modifier). JD: down-weight inactive /
#     low-response "not actually available" candidates. Tuned vs audit percentiles.
BEHAVIORAL = {
    "availability_positive": {
        "notice_period_days_low": 30,        # JD loves sub-30; buy-out up to 30
        "open_to_work_flag": True,
        "recent_active_days": 60,            # logged in within ~2 months
    },
    "responsiveness_positive": {
        "recruiter_response_rate_med": 0.44,  # pool median (audit); reward above
        "interview_completion_rate_med": 0.62,
    },
    "trust_positive": ["verified_email", "verified_phone", "linkedin_connected"],
    "sentinel_unknown_value": -1,  # github_activity_score / offer_acceptance_rate; treat as neutral, NEVER 0
    "modifier_range": (0.7, 1.15),  # multiplicative; nudges, never dominates fit
    "jd": "perfect-on-paper but 6mo inactive + 5% response = not available -> down-weight.",
}

# --- "AI keyword set" that stuffers abuse. Presence of these ALONE is NOT fit (the trap).
#     Used to detect title<->skill incongruity: many AI keywords + non-tech title = stuffer.
STUFFER_AI_KEYWORDS = [
    "rag", "llm", "gpt", "langchain", "openai", "transformer", "nlp", "machine learning",
    "deep learning", " ai ", "prompt", "vector database", "pinecone", "hugging face",
    "fine-tuning llms", "generative ai", "genai", "chatgpt", "embeddings", "pytorch", "tensorflow",
]

# --- Non-tech current titles (audit-derived) that are NOT a fit for this role regardless of
#     listed skills -> strong negative / honeypot-stuffer flag when paired with AI keywords.
NONTECH_TITLES = [
    "business analyst", "hr manager", "accountant", "project manager", "customer support",
    "operations manager", "content writer", "sales executive", "graphic designer",
    "marketing manager", "mechanical engineer", "civil engineer", "electrical engineer",
    "recruiter", "teacher", "nurse", "doctor", "chef", "lawyer",
]

# --- Tech titles that are a genuine positive starting point for THIS role.
TECH_TITLES_POSITIVE = [
    "ml engineer", "machine learning engineer", "ai engineer", "applied scientist",
    "applied ml", "data scientist", "research engineer", "nlp engineer", "search engineer",
    "ranking engineer", "software engineer", "backend engineer", "full stack", "platform engineer",
]


def _norm(s):
    return (s or "").lower()


def text_blob(candidate):
    """Concatenate the searchable free-text of a candidate (headline, summary, titles,
    descriptions, skill names) for keyword predicates. Lowercased."""
    p = candidate.get("profile", {}) or {}
    parts = [p.get("headline"), p.get("summary"), p.get("current_title"), p.get("current_industry")]
    for job in candidate.get("career_history", []) or []:
        parts += [job.get("title"), job.get("description"), job.get("industry")]
    for sk in candidate.get("skills", []) or []:
        parts.append(sk.get("name"))
    return _norm(" || ".join([x for x in parts if x]))


def any_term(blob, terms):
    return any(t in blob for t in terms)


def count_terms(blob, terms):
    return sum(1 for t in terms if t in blob)
