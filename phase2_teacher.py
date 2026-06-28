"""
Phase 2 — LLM teacher labeling (OFFLINE ONLY; never part of rank.py).

Claude judges the retrieval contender pool as a premium technical recruiter and assigns,
per candidate: relevance tier 0-5, fit score 0-100, archetype, rationale, concerns.
These silver labels (a) calibrate the rule-layer weights, (b) are CPRD distillation
targets, (c) validate the ranker. Independent judgment — we do NOT feed our own
company/honeypot computations, so the labels stay an independent signal.

Safety: key via keyloader (env or .secrets/, gitignored). Never printed. Resumable
(skips already-labeled ids) and rate-limited with backoff.

Usage:
  python phase2_teacher.py --limit 20         # cheap test run first
  python phase2_teacher.py --limit 400        # full head of the pool
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anthropic
from keyloader import get_api_key, masked

ART = os.path.join(HERE, "artifacts")
POOL = os.path.join(ART, "retrieval_topk.json")
OUT = os.path.join(ART, "teacher_labels.jsonl")
CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl")

JD_BRIEF = """ROLE: Senior AI Engineer, FOUNDING TEAM at a Series-A product startup (Redrob).
Owns the intelligence layer: ranking, retrieval, matching systems. Scrappy shipper > pure researcher.

MUST HAVE (production, real users): embeddings-based retrieval / semantic search (sentence-transformers,
BGE, E5, OpenAI embeddings); vector DB / hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Milvus,
Elasticsearch); strong Python; rigorous ranking evaluation (NDCG, MRR, MAP, A/B, offline-online).
NICE: LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, HR-tech, large-scale inference, OSS.

IDEAL (read between the lines): ~6-8 yrs, 4-5 in APPLIED ML at PRODUCT companies (not services);
shipped >=1 end-to-end ranking/search/recommendation system at real scale; opinionated systems thinker;
in/near Noida-Pune (India strong; outside India case-by-case, no visa); active & reachable.

HARD NEGATIVES / DO NOT WANT: pure research with no production; only recent (<12mo) LangChain->OpenAI
with no pre-LLM ML; senior who hasn't coded in 18mo; title-chasers (hop every ~1.5y); framework
enthusiasts (tutorials/demos); ENTIRE career at IT-services/consulting (TCS/Infosys/Wipro/Accenture/
Cognizant/Capgemini/Mindtree...) UNLESS redeemed by a prior product-company role; primary CV/speech/
robotics WITHOUT NLP/IR; 5+ yrs closed-source only with no external validation.

TRAPS in the data: (1) keyword-stuffers = many AI buzzwords but an out-of-domain title like
"Marketing Manager" -> NOT a fit. (2) plain-language Tier-5s = genuine fit who built a recsys/search
at a product company but DON'T use jargon (RAG/Pinecone) -> still a strong fit; reward substance over
keywords. (3) honeypots = logically IMPOSSIBLE profiles (e.g. a single job longer than the whole
career; >=3 'expert' skills with 0 months used; experience exceeding the timeline) -> tier 0.
Behavioral availability matters: inactive 6mo + low response rate = "not actually available", down-weight.

JUDGE LIKE A 15-YEAR HEAD-HUNTER: title + career trajectory beat a keyword list; product/AI-native
company experience beats services tenure; from-scratch/first-principles depth beats flashy-library
name-dropping; corroboration (title x skills x duration x endorsements) separates real from poser.
"""

RUBRIC = """For EACH candidate output: tier (0-5), fit (0-100), archetype, rationale, concerns.
TIER scale (relevant = tier 3+):
 5 = exceptional founding-team fit: shipped retrieval/ranking/search ML at a product/AI-native company,
     strong trajectory, available. Rare.
 4 = strong fit: clear production retrieval/ranking/recsys ML at a product company; minor gaps.
 3 = relevant/solid: real applied ML adjacent to retrieval/ranking; plausible with some gaps.
 2 = weak/adjacent: ML-ish but off-target, OR junior/overqualified, OR services-heavy without product redemption.
 1 = poor: wrong focus or keyword-match only.
 0 = no-fit / keyword-stuffer with out-of-domain title / honeypot (impossible profile) / unavailable shell.
fit (0-100) is a finer continuous score consistent with tier.
archetype: one of standout|strong|solid|adjacent|wannabe|wrong_domain|overqualified|junior|services_only|honeypot|unavailable.
rationale: 1-2 sentences citing SPECIFIC profile facts (years, title, named skills/systems, company, signals) AND the JD connection.
concerns: brief honest gaps/risks (or "none").
Return ONLY a JSON array of objects with keys: candidate_id, tier, fit, archetype, rationale, concerns. No prose, no markdown."""


def to_teacher_text(c):
    p = c.get("profile", {}) or {}
    s = c.get("redrob_signals", {}) or {}
    L = [f"candidate_id: {c['candidate_id']}",
         f"headline: {p.get('headline')}",
         f"years_experience: {p.get('years_of_experience')}; current: {p.get('current_title')} "
         f"at {p.get('current_company')} ({p.get('current_industry')}, {p.get('current_company_size')})"]
    if p.get("summary"):
        L.append(f"summary: {p['summary'][:600]}")
    for j in c.get("career_history", []) or []:
        when = "current" if j.get("is_current") else f"{j.get('start_date','?')}..{j.get('end_date')}"
        L.append(f"- {j.get('title')} @ {j.get('company')} [{j.get('industry')}, "
                 f"{j.get('duration_months')}mo, {when}]: {(j.get('description') or '')[:300]}")
    sk = ", ".join(f"{x.get('name')}({x.get('proficiency')},{x.get('endorsements')}e,"
                   f"{x.get('duration_months')}mo)" for x in (c.get("skills", []) or [])[:25])
    L.append("skills: " + sk)
    ed = "; ".join(f"{e.get('degree')} {e.get('field_of_study')} @ {e.get('institution')} [{e.get('tier')}]"
                   for e in (c.get("education", []) or []))
    L.append("education: " + ed)
    L.append(f"signals: response_rate={s.get('recruiter_response_rate')}, last_active={s.get('last_active_date')}, "
             f"notice_days={s.get('notice_period_days')}, open_to_work={s.get('open_to_work_flag')}, "
             f"interview_completion={s.get('interview_completion_rate')}, github={s.get('github_activity_score')}, "
             f"offer_accept={s.get('offer_acceptance_rate')}, verified_email={s.get('verified_email')}, "
             f"completeness={s.get('profile_completeness_score')}")
    return "\n".join(str(x) for x in L)


def extract_json_array(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    a, b = t.find("["), t.rfind("]")
    if a == -1 or b == -1:
        raise ValueError("no JSON array in response")
    return json.loads(t[a:b + 1])


def call_batch(client, model, batch, retries=4):
    prompt = (JD_BRIEF + "\n\n" + RUBRIC + "\n\nCANDIDATES (" + str(len(batch)) + "):\n\n"
              + "\n\n---\n\n".join(to_teacher_text(c) for c in batch))
    delay = 4.0
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=8192,
                system="You are a meticulous 15-year technical head-hunter. Judge genuine role fit, "
                       "not keyword overlap. Be calibrated and honest; catch impossible (honeypot) profiles.",
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_json_array(msg.content[0].text)
        except anthropic.BadRequestError:
            raise  # deterministic client error (bad param etc.) — never retry
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt+1} after API error ({type(e).__name__}); sleeping {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
        except ValueError as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt+1} parse error: {e}")
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="top-N of the pool to label")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    key = get_api_key()
    print("using key", masked(key), "| model", args.model)
    client = anthropic.Anthropic(api_key=key)

    pool = json.load(open(POOL, encoding="utf-8"))[:args.limit]
    score_by_id = {p["id"]: p["score"] for p in pool}
    want = list(score_by_id.keys())

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["candidate_id"])
    todo_ids = [i for i in want if i not in done]
    print(f"pool {len(want)} | already labeled {len(done)} | to label {len(todo_ids)}")
    if not todo_ids:
        return

    need = set(todo_ids)
    recs = {}
    with open(CAND, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in need:
                recs[c["candidate_id"]] = c
                if len(recs) == len(need):
                    break

    out = open(OUT, "a", encoding="utf-8")
    n = 0
    for i in range(0, len(todo_ids), args.batch):
        ids = todo_ids[i:i + args.batch]
        batch = [recs[x] for x in ids if x in recs]
        try:
            results = call_batch(client, args.model, batch)
        except Exception as e:
            print(f"batch {i//args.batch} failed: {type(e).__name__}: {e}")
            continue
        by_id = {r.get("candidate_id"): r for r in results}
        for cid in ids:
            r = by_id.get(cid)
            if not r:
                continue
            r["retrieval_score"] = score_by_id.get(cid)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        out.flush()
        print(f"  labeled {n}/{len(todo_ids)} (batch {i//args.batch + 1})")
        time.sleep(1.0)
    out.close()
    print("done. labels at", OUT)


if __name__ == "__main__":
    main()
