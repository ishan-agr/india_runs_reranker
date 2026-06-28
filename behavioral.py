"""
Behavioral hiring-probability modifier (Phase 5).

The JD is explicit: a perfect-on-paper candidate who is inactive / unresponsive is
"not actually available — down-weight them." So this is a MULTIPLIER on fit, not fit
itself. Range [0.7, 1.15]: it nudges, never dominates. Asymmetric by design — we
punish unavailability harder than we reward eagerness (passive strong candidates exist).

Sentinels: github_activity_score = -1 and offer_acceptance_rate = -1 mean UNKNOWN
(65% / 60% of the pool) — treated as neutral, never as 0.

Weights/percentiles are audit-derived (response median 0.44, interview-completion 0.62,
notice median 90, recruiter-saves median 7) and live as tunable constants.
"""
import json
from datetime import date

TODAY = date(2026, 6, 28)
RR_MED, IC_MED, OAR_MED, GH_MED, SAVES_MED = 0.44, 0.62, 0.48, 29.0, 7
MULT_LO, MULT_HI = 0.70, 1.15
SENTINEL = -1


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _parse(s):
    try:
        y, m, d = str(s).split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def behavioral_modifier(c):
    """Return (multiplier in [0.7,1.15], facts dict, short_reason str)."""
    sig = c.get("redrob_signals", {}) or {}
    total = 0.0
    facts = {}

    rr = sig.get("recruiter_response_rate")
    if isinstance(rr, (int, float)):
        total += _clamp((rr - RR_MED) / 0.5, -1, 1) * 0.06
        facts["response_rate"] = rr

    ic = sig.get("interview_completion_rate")
    if isinstance(ic, (int, float)):
        total += _clamp((ic - IC_MED) / 0.4, -1, 1) * 0.03
        facts["interview_completion"] = ic

    la = _parse(sig.get("last_active_date"))
    if la:
        days = (TODAY - la).days
        facts["days_inactive"] = days
        if days <= 30:
            total += 0.03
        elif days <= 90:
            total += 0.0
        elif days <= 180:
            total -= 0.05
        elif days <= 365:
            total -= 0.12
        else:
            total -= 0.20

    np = sig.get("notice_period_days")
    if isinstance(np, (int, float)):
        facts["notice_period_days"] = np
        if np <= 30:
            total += 0.04
        elif np <= 60:
            total += 0.01
        elif np <= 90:
            total += 0.0
        elif np <= 120:
            total -= 0.03
        else:
            total -= 0.06

    if sig.get("open_to_work_flag") is True:
        total += 0.03
        facts["open_to_work"] = True

    verified = sum(1 for k in ("verified_email", "verified_phone", "linkedin_connected")
                   if sig.get(k) is True)
    total += 0.01 * verified
    facts["verified_count"] = verified

    saves = sig.get("saved_by_recruiters_30d")
    if isinstance(saves, (int, float)):
        total += _clamp((saves - SAVES_MED) / 20.0, -0.5, 1.0) * 0.03

    oar = sig.get("offer_acceptance_rate")
    if isinstance(oar, (int, float)) and oar != SENTINEL:
        total += _clamp((oar - OAR_MED) / 0.4, -1, 1) * 0.02
        facts["offer_acceptance"] = oar

    gh = sig.get("github_activity_score")
    if isinstance(gh, (int, float)) and gh != SENTINEL:
        total += _clamp((gh - GH_MED) / 40.0, -1, 1) * 0.02
        facts["github_activity"] = gh

    mult = _clamp(1.0 + total, MULT_LO, MULT_HI)

    # short reason
    bits = []
    if "response_rate" in facts:
        bits.append(("responsive" if rr >= 0.62 else "low-response" if rr < 0.25 else "avg-response")
                    + f" {rr:.2f}")
    if "days_inactive" in facts:
        d = facts["days_inactive"]
        bits.append(f"active {d}d ago" if d <= 90 else f"inactive {d}d")
    if "notice_period_days" in facts:
        bits.append(f"{int(facts['notice_period_days'])}d notice")
    reason = ", ".join(bits)

    return mult, facts, reason


if __name__ == "__main__":
    CAND = (r"E:\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge"
            r"\India_runs_data_and_ai_challenge\candidates.jsonl")
    vals = []
    samples = []
    with open(CAND, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            m, facts, reason = behavioral_modifier(c)
            vals.append(m)
            if i < 5:
                samples.append((c["candidate_id"], round(m, 3), reason))
    vals.sort()
    n = len(vals)

    def pct(q):
        return round(vals[min(n - 1, int(q * n))], 3)

    print(f"n={n} multiplier: min={vals[0]:.3f} p10={pct(.10)} p25={pct(.25)} "
          f"median={pct(.50)} p75={pct(.75)} p90={pct(.90)} max={vals[-1]:.3f}")
    frac_lo = sum(1 for v in vals if v <= 0.80) / n
    frac_hi = sum(1 for v in vals if v >= 1.10) / n
    print(f"frac <=0.80 (penalized): {frac_lo:.3f} | frac >=1.10 (boosted): {frac_hi:.3f}")
    print("samples:", samples)
