# agents/critic.py
# ─────────────────────────────────────────────────────────────────────────────
# CRITIC AGENT — Phase D
#
# RESPONSIBILITY:
#   Review the Analyst's work and flag any problems:
#     1. Unrealistic numbers (rule-based, deterministic)
#     2. Evidence gaps (LLM checks if narrative is grounded in data)
#     3. Internal inconsistencies (LLM cross-checks assumptions vs outputs)
#     4. Missing data (deterministic checks)
#
# PATTERN:
#   This is a "Critic" agent in multi-agent system design. The idea is that
#   having a separate agent review the work catches errors the Analyst missed,
#   just like a second analyst reviewing a junior analyst's model.
#
#   In production, you could loop: if Critic scores < 6, send back to Analyst
#   for revision. We implement one round of critique here.
# ─────────────────────────────────────────────────────────────────────────────

from groq import Groq
import os
from models.schemas import FinancialData, ValuationResult, CritiqueResult


def run_critic(financial_data: FinancialData, valuation: ValuationResult) -> CritiqueResult:
    """
    Main entry point for the Critic agent.
    Runs deterministic checks first, then LLM qualitative review.
    """

    print(f"\n{'='*60}")
    print(f"  CRITIC AGENT: Reviewing valuation for {financial_data.ticker}")
    print(f"{'='*60}")

    # ── Step 1: Deterministic rule-based checks ───────────────────────────────
    # These never need an LLM — they're simple if/else logic
    inconsistencies   = _check_numeric_consistency(financial_data, valuation)
    missing_data      = _check_missing_data(financial_data, valuation)
    unrealistic       = _check_unrealistic_assumptions(valuation)

    print(f"  [Critic] Found {len(inconsistencies)} inconsistencies, "
          f"{len(missing_data)} missing data flags, "
          f"{len(unrealistic)} unrealistic assumptions")

    # ── Step 2: LLM qualitative review ───────────────────────────────────────
    # Claude reads the full analyst report and gives a holistic critique
    evidence_gaps, critique_summary = _llm_evidence_audit(financial_data, valuation)

    # ── Step 3: Calculate revised confidence ─────────────────────────────────
    # Deterministic: lower score based on how many issues were found
    total_issues = len(inconsistencies) + len(missing_data) + len(unrealistic) + len(evidence_gaps)
    revised_confidence = max(1, valuation.confidence_score - total_issues)

    passed = len(inconsistencies) == 0 and len(unrealistic) == 0 and revised_confidence >= 6

    print(f"  [Critic] Revised confidence: {revised_confidence}/10 | Passed: {passed}")

    # ── Step 4: Recommended actions ───────────────────────────────────────────
    recommendations = _generate_recommendations(
        inconsistencies, missing_data, unrealistic, evidence_gaps
    )

    return CritiqueResult(
        ticker=financial_data.ticker,
        passed_validation=passed,
        inconsistencies_found=inconsistencies,
        missing_data_flags=missing_data,
        evidence_gaps=evidence_gaps,
        unrealistic_assumptions=unrealistic,
        critique_summary=critique_summary,
        revised_confidence_score=revised_confidence,
        recommended_actions=recommendations,
    )


def _check_numeric_consistency(financial_data: FinancialData, valuation: ValuationResult) -> list[str]:
    """
    Deterministic numeric checks — pure Python logic, no LLM.
    Catches the most obvious data quality and model errors.
    """
    issues = []

    # ── Market data consistency ────────────────────────────────────────────────
    if financial_data.net_margin and financial_data.net_margin > 1.0:
        issues.append(
            f"Net margin of {financial_data.net_margin:.1%} exceeds 100% — impossible, data error"
        )

    if financial_data.net_margin and financial_data.net_margin < -1.0:
        issues.append(
            f"Net margin of {financial_data.net_margin:.1%} is extremely negative — "
            "DCF assumptions may not apply to a deeply unprofitable company"
        )

    # ── P/E vs earnings sign check ────────────────────────────────────────────
    if (financial_data.pe_ratio and financial_data.pe_ratio > 0 and
            financial_data.net_income_ttm and financial_data.net_income_ttm < 0):
        issues.append(
            "P/E ratio is positive but net income is negative — P/E is not meaningful here"
        )

    # ── DCF result sanity check ───────────────────────────────────────────────
    if valuation.dcf_result:
        dcf_price = valuation.dcf_result.dcf_price_per_share
        current   = financial_data.current_price

        if current and current > 0:
            upside = (dcf_price - current) / current

            # If DCF implies >200% upside or >80% downside, flag it
            if upside > 2.0:
                issues.append(
                    f"DCF price (${dcf_price:.2f}) implies {upside:.0%} upside vs "
                    f"current price (${current:.2f}) — review growth assumptions"
                )
            elif upside < -0.8:
                issues.append(
                    f"DCF price (${dcf_price:.2f}) implies {upside:.0%} downside vs "
                    f"current price (${current:.2f}) — review WACC or FCF inputs"
                )

    # ── Multiples vs market check ─────────────────────────────────────────────
    if valuation.multiples_result and valuation.multiples_result.blended_target_price:
        mult_price = valuation.multiples_result.blended_target_price
        current    = financial_data.current_price

        if current and current > 0:
            mult_upside = (mult_price - current) / current
            if abs(mult_upside) > 1.5:  # More than 150% difference
                issues.append(
                    f"Multiples-based price (${mult_price:.2f}) diverges significantly "
                    f"from current price (${current:.2f}) — sector multiples may not apply"
                )

    # ── DCF vs Multiples divergence ───────────────────────────────────────────
    if (valuation.dcf_result and valuation.multiples_result and
            valuation.multiples_result.blended_target_price):
        dcf_p = valuation.dcf_result.dcf_price_per_share
        mul_p = valuation.multiples_result.blended_target_price
        if dcf_p > 0 and mul_p > 0:
            divergence = abs(dcf_p - mul_p) / ((dcf_p + mul_p) / 2)
            if divergence > 0.50:  # More than 50% apart
                issues.append(
                    f"DCF (${dcf_p:.2f}) and Multiples (${mul_p:.2f}) diverge by "
                    f"{divergence:.0%} — high model uncertainty"
                )

    return issues


# REPLACE the entire _check_missing_data function with this:
def _check_missing_data(financial_data: FinancialData, valuation: ValuationResult) -> list[str]:
    """
    Flags missing data that materially affects result reliability.
    Note: we only flag here if the Analyst could NOT have caught this already.
    """
    flags = []

    if not valuation.dcf_result and not valuation.multiples_result:
        flags.append("CRITICAL: Neither DCF nor Multiples could be calculated — no valuation basis")
    if not financial_data.beta:
        flags.append("Beta unavailable — WACC estimate is less precise")
    if not financial_data.recent_news:
        flags.append("No news data — qualitative context missing from analysis")
    if not financial_data.total_debt:
        flags.append("Debt data missing — equity bridge may be inaccurate")

    return flags


def _check_unrealistic_assumptions(valuation: ValuationResult) -> list[str]:
    """
    Checks whether the LLM's chosen assumptions are in realistic ranges.
    These boundaries are based on standard financial analysis practice.
    """
    issues = []

    if valuation.dcf_result:
        wacc = valuation.dcf_result.wacc
        tgr  = valuation.dcf_result.terminal_growth_rate

        if wacc < 0.05:
            issues.append(f"WACC of {wacc:.1%} is unusually low (typical range: 6-14%)")
        if wacc > 0.20:
            issues.append(f"WACC of {wacc:.1%} is unusually high (typical range: 6-14%)")
        if tgr > 0.04:
            issues.append(
                f"Terminal growth rate of {tgr:.1%} exceeds long-run GDP growth — "
                "implies the company will eventually be larger than the global economy"
            )
        if tgr < 0:
            issues.append(f"Negative terminal growth rate of {tgr:.1%} implies permanent decline")

    # Check if target price range is internally consistent
    if valuation.target_price_low > valuation.target_price_high:
        issues.append("Target price low is higher than target price high — model error")

    return issues


def _llm_evidence_audit(
    financial_data: FinancialData,
    valuation: ValuationResult
) -> tuple[list[str], str]:
    """
    LLM-powered qualitative review. Claude reads the analyst's full output
    and checks whether the narrative is logically consistent with the data.

    LLM ROLE HERE: Cross-referencing qualitative narrative against quantitative data.
    This is hard to do with rules — it requires reading comprehension and reasoning.

    RETURNS: (list of evidence gaps, overall critique summary paragraph)
    """

    # Build a compact summary of what the analyst produced
    dcf_summary = "Not available"
    if valuation.dcf_result:
        dcf_summary = (
            f"DCF price: ${valuation.dcf_result.dcf_price_per_share:.2f} "
            f"(WACC: {valuation.dcf_result.wacc:.1%}, "
            f"TGR: {valuation.dcf_result.terminal_growth_rate:.1%})"
        )

    mult_summary = "Not available"
    if valuation.multiples_result and valuation.multiples_result.blended_target_price:
        mult_summary = f"Multiples price: ${valuation.multiples_result.blended_target_price:.2f}"

    prompt = f"""You are a senior risk officer performing a critical review of an analyst's valuation report.
Your job is to find flaws, not validate — be skeptical.

COMPANY: {financial_data.company_name} ({financial_data.ticker})
SECTOR: {financial_data.sector}

ANALYST NARRATIVE:
{valuation.analyst_narrative}

GROWTH RATIONALE:
{valuation.growth_assumption_rationale}

RISK FACTORS IDENTIFIED:
{chr(10).join(f"- {r}" for r in valuation.risk_factors)}

VALUATION RESULTS:
- {dcf_summary}
- {mult_summary}
- Target Range: ${valuation.target_price_low:.2f} – ${valuation.target_price_high:.2f}
- Confidence: {valuation.confidence_score}/10

SUPPORTING NEWS THEMES:
{chr(10).join(f"- {n}" for n in financial_data.recent_news[:3]) or "None available"}

KEY FINANCIAL METRICS:
- Net Margin: {f"{financial_data.net_margin:.1%}" if financial_data.net_margin else "N/A"}
- Revenue Growth YoY: {f"{financial_data.revenue_growth_yoy:.1%}" if financial_data.revenue_growth_yoy else "N/A"}
- P/E: {financial_data.pe_ratio or "N/A"} | EV/EBITDA: {financial_data.ev_ebitda or "N/A"}
- FCF: {"${:,.0f}".format(financial_data.free_cash_flow) if financial_data.free_cash_flow else "N/A"}

Identify:
1. Any claims in the narrative NOT supported by the data or news (evidence gaps)
2. Internal logical contradictions
3. Missing risk factors that should have been flagged
4. Overall assessment

Return ONLY valid JSON (no markdown):
{{
  "evidence_gaps": ["<gap 1>", "<gap 2>"],
  "critique_summary": "<2-3 paragraph honest critique of this valuation>"
}}"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences more aggressively
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Extract just the JSON object if there's surrounding text
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        parsed = json.loads(raw)
        evidence_gaps    = parsed.get("evidence_gaps", [])
        critique_summary = parsed.get("critique_summary", "No critique generated.")
    except Exception:
        print(f"  [Critic] WARNING: Could not parse LLM critique JSON")
        evidence_gaps    = ["Could not parse critic output — manual review recommended"]
        critique_summary = "Automated critique parsing failed. Manual review of this valuation is strongly recommended."

    return evidence_gaps, critique_summary


def _generate_recommendations(
    inconsistencies: list[str],
    missing_data: list[str],
    unrealistic: list[str],
    evidence_gaps: list[str],
) -> list[str]:
    """
    Deterministically generates action items based on the issues found.
    Simple mapping from issue type → recommended action.
    """
    actions = []

    if inconsistencies:
        actions.append("Re-verify financial data from primary sources (SEC EDGAR, company IR page)")
    if any("FCF" in m for m in missing_data):
        actions.append("Obtain FCF directly from cash flow statement — do not estimate")
    if any("EBITDA" in m for m in missing_data):
        actions.append("Source EBITDA from income statement before running multiples")
    if any("Beta" in m for m in missing_data):
        actions.append("Estimate beta from historical returns if not available in data feed")
    if unrealistic:
        actions.append("Review WACC components (risk-free rate, ERP, beta) against current market conditions")
    if evidence_gaps:
        actions.append("Strengthen narrative with specific data citations before finalizing report")
    if not actions:
        actions.append("No major actions required — valuation appears internally consistent")

    return actions


# Required for JSON parsing in critic
import json
