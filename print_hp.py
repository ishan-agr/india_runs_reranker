import json
import os
import sys

ART = r"E:\[PUB] India_runs_data_and_ai_challenge\redrob_ranker\artifacts"
hp = json.load(open(os.path.join(ART, "honeypots_review.json"), encoding="utf-8"))
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
e = hp[idx]
print("id:", e["candidate_id"], "| detected_by:", e["detected_by"])
print("profile yoe:", e["profile"]["years_of_experience"], "years")
print("deterministic_flags:", e["deterministic_flags"])
print("diagnostics:", json.dumps(e["diagnostics"]))
print("teacher tier/fit/arch:", e["teacher"]["tier"], e["teacher"]["fit"], e["teacher"]["archetype"])
print("teacher rationale:", e["teacher"]["rationale"])
print("teacher concerns:", e["teacher"]["concerns"])
print("career_history:")
tot = 0
for j in e["profile"]["career_history"]:
    tot += j["duration_months"] or 0
    print(f"  {j['title']} @ {j['company']:12s} | {j['duration_months']}mo | {j['dates']} | current={j['is_current']}")
print(f"SUM of listed durations: {tot} months = {tot/12:.1f} years   (profile yoe = {e['profile']['years_of_experience']} yrs)")
