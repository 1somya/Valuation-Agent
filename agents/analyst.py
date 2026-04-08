# agents/analyst.py
# ─────────────────────────────────────────────────────────────────────────────
# ANALYST AGENT — Phase B & C
#
# RESPONSIBILITY:
#   1. Use Gemini to reason about appropriate valuation assumptions
#      (growth rate, WACC, key risks) based on the financial data + news
#   2. Pass those assumptions to Python for the actual math
#   3. Compile everything into a structured ValuationResult
#
# LLM DOES:   Choose growth rate, WACC, narrative, risk factors
# PYTHON DOES: All DCF math, multiples math, sensitivity table
# ─────────────────────────────────────────────────────────────────────────────

from groq import Groq
import os
import json
from models.schemas import FinancialData, ValuationResult
from tools.financial_calc import run_dcf, run_multiples, run_sensitivity, validate_margins
from tools.data_fetcher import get_shares_outstanding


def run_analyst(financial_data: FinancialData) -> ValuationResult:
    """
    Main entry point for the Analyst agent.
    Orchestrates: LLM assumptions → Python math → structured output.
    """

    print(f"\n{'='*60}")
    print(f"  ANALYST AGENT: Starting valuation for {financial_data.ticker}")
    print(f"{'='*60}")

    # ── Step 1: Run deterministic sanity checks first ─────────────────────────
    warnings = validate_margins(financial_data)
    if warnings:
        print(f"  [Analyst] Data warnings detected:")
        for w in warnings:
            print(f"    ⚠ {w}")

    # ── Step 2: Ask Gemini for valuation assumptions ──────────────────────────
    # LLM reads the data and decides what growth rate and WACC are appropriate
    assumptions = _get_llm_assumptions(financial_data)

    growth_rate     = assumptions["growth_rate"]
    wacc            = assumptions["wacc"]
    terminal_growth = assumptions["terminal_growth_rate"]
    narrative       = assumptions["narrative"]
    risk_factors    = assumptions["risk_factors"]
    rationale       = assumptions["growth_rationale"]

    print(f"  [Analyst] LLM assumptions: growth={growth_rate:.1%}, WACC={wacc:.1%}, TGR={terminal_growth:.1%}")

    # ── Step 3: Fetch shares outstanding ─────────────────────────────────────
    shares = get_shares_outstanding(financial_data.ticker)

    # ── Step 4: Run DCF — pure Python math ───────────────────────────────────
    dcf_result = None
    try:
        dcf_result = run_dcf(
            financial_data=financial_data,
            shares_outstanding=shares,
            growth_rate=growth_rate,
            wacc=wacc,
            terminal_growth=terminal_growth,
        )
        print(f"  [Analyst] DCF price: ${dcf_result.dcf_price_per_share:.2f}")
    except ValueError as e:
        print(f"  [Analyst] DCF skipped: {e}")

    # ── Step 5: Run Multiples — pure Python math ──────────────────────────────
    multiples_result = None
    try:
        multiples_result = run_multiples(
            financial_data=financial_data,
            shares_outstanding=shares,
        )
        if multiples_result.blended_target_price:
            print(f"  [Analyst] Multiples price: ${multiples_result.blended_target_price:.2f}")
    except Exception as e:
        print(f"  [Analyst] Multiples skipped: {e}")

    # ── Step 6: Sensitivity Analysis — pure Python math ──────────────────────
    sensitivity_table = run_sensitivity(
        financial_data=financial_data,
        shares_outstanding=shares,
        base_growth_rate=growth_rate,
        base_wacc=wacc,
        base_terminal_growth=terminal_growth,
    )
    print(f"  [Analyst] Generated {len(sensitivity_table)} sensitivity scenarios")

    # ── Step 7: Calculate the target price range ──────────────────────────────
    target_base = _calculate_target_price(dcf_result, multiples_result)
    target_low  = target_base * 0.85
    target_high = target_base * 1.15

    # ── Step 8: Calculate confidence score ───────────────────────────────────
    confidence = _calculate_confidence(financial_data, dcf_result, multiples_result)

    # ── Step 9: Compile all sources for traceability ──────────────────────────
    sources = ["Yahoo Finance (yfinance)"] + financial_data.news_sources[:5]

    print(f"  [Analyst] Target range: ${target_low:.2f} – ${target_high:.2f} (base: ${target_base:.2f})")
    print(f"  [Analyst] Confidence: {confidence}/10")

    return ValuationResult(
        ticker=financial_data.ticker,
        company_name=financial_data.company_name,
        valuation_method="DCF + Comparable Multiples",
        analyst_narrative=narrative,
        risk_factors=risk_factors,
        growth_assumption_rationale=rationale,
        dcf_result=dcf_result,
        multiples_result=multiples_result,
        sensitivity_table=sensitivity_table,
        target_price_low=round(target_low, 2),
        target_price_high=round(target_high, 2),
        target_price_base=round(target_base, 2),
        current_price=financial_data.current_price,
        evidence_sources=sources,
        confidence_score=confidence,
    )


def _get_llm_assumptions(financial_data: FinancialData) -> dict:
    """
    The key LLM reasoning call. Groq reads all the financial context and
    decides on appropriate valuation assumptions.

    LLM DOES:   Qualitative reasoning about growth, risk, WACC
    PYTHON DOES: Clamps values to realistic ranges after LLM responds
    """

    # Build the financial data block to inject into the prompt
    financials_block = f"""
    Company: {financial_data.company_name} ({financial_data.ticker})
    Sector: {financial_data.sector} | Industry: {financial_data.industry}

    MARKET DATA:
    - Current Price: ${financial_data.current_price or 'N/A'}
    - Market Cap: ${f"{financial_data.market_cap:,.0f}" if financial_data.market_cap else 'N/A'}
    - P/E Ratio: {financial_data.pe_ratio or 'N/A'}
    - Forward P/E: {financial_data.forward_pe or 'N/A'}
    - EV/EBITDA: {financial_data.ev_ebitda or 'N/A'}
    - Beta: {financial_data.beta or 'N/A'}

    INCOME STATEMENT (TTM):
    - Revenue: ${f"{financial_data.revenue_ttm:,.0f}" if financial_data.revenue_ttm else 'N/A'}
    - Net Income: ${f"{financial_data.net_income_ttm:,.0f}" if financial_data.net_income_ttm else 'N/A'}
    - EBITDA: ${f"{financial_data.ebitda_ttm:,.0f}" if financial_data.ebitda_ttm else 'N/A'}
    - Gross Margin: {f"{financial_data.gross_margin:.1%}" if financial_data.gross_margin else 'N/A'}
    - Net Margin: {f"{financial_data.net_margin:.1%}" if financial_data.net_margin else 'N/A'}
    - Revenue Growth YoY: {f"{financial_data.revenue_growth_yoy:.1%}" if financial_data.revenue_growth_yoy else 'N/A'}

    CASH FLOW & BALANCE SHEET:
    - Free Cash Flow: ${f"{financial_data.free_cash_flow:,.0f}" if financial_data.free_cash_flow else 'N/A'}
    - Total Debt: ${f"{financial_data.total_debt:,.0f}" if financial_data.total_debt else 'N/A'}
    - Cash: ${f"{financial_data.cash_and_equivalents:,.0f}" if financial_data.cash_and_equivalents else 'N/A'}

    RECENT NEWS THEMES:
    {chr(10).join(f"- {n}" for n in financial_data.recent_news[:3]) or "No news available"}
    """

    prompt = f"""You are a senior equity research analyst at a top-tier investment bank.
    Analyze the following company data and determine appropriate valuation assumptions.

    {financials_block}

    Based on this data, provide your valuation assumptions. Consider:
    1. Revenue growth rate vs peers and historical trend
    2. WACC: use the CAPM approach — risk-free rate (~4.5% US 10Y) + equity risk premium (~5.5%) x beta
    Then adjust for company-specific risk (leverage, size, sector)
    3. Terminal growth rate: typically 2-3% for mature companies, slightly higher for high-growth
    4. Key risks from the news and financial profile

    Return ONLY a valid JSON object with exactly these fields (no markdown, no explanation):
    {{
    "growth_rate": <float, e.g. 0.10 for 10%>,
    "wacc": <float, e.g. 0.09 for 9%>,
    "terminal_growth_rate": <float, e.g. 0.025 for 2.5%>,
    "narrative": "<2-3 sentence summary of the investment case>",
    "growth_rationale": "<1-2 sentences explaining why you chose this growth rate>",
    "risk_factors": ["<risk 1>", "<risk 2>", "<risk 3>"]
    }}"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if Gemini added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        assumptions = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [Analyst] WARNING: LLM returned non-JSON, using conservative defaults")
        assumptions = {
            "growth_rate": 0.07,
            "wacc": 0.10,
            "terminal_growth_rate": 0.025,
            "narrative": "Could not parse LLM assumptions. Conservative defaults applied.",
            "growth_rationale": "Defaulting to 7% growth due to parse error.",
            "risk_factors": ["Data quality risk", "Assumption uncertainty", "Model parse error"],
        }

    # ── Clamp values to realistic ranges ─────────────────────────────────────
    # Even if the LLM hallucinates a 50% growth rate, we cap it here
    assumptions["growth_rate"]          = max(0.0,  min(assumptions.get("growth_rate", 0.08),  0.40))
    assumptions["wacc"]                 = max(0.04, min(assumptions.get("wacc", 0.09),          0.25))
    assumptions["terminal_growth_rate"] = max(0.01, min(assumptions.get("terminal_growth_rate", 0.025), 0.05))

    return assumptions


def _calculate_target_price(dcf_result, multiples_result) -> float:
    """
    Blends DCF (60%) and multiples (40%) into one base price.
    Weighting is a standard practitioner convention.
    """
    prices  = []
    weights = []

    if dcf_result and dcf_result.dcf_price_per_share > 0:
        prices.append(dcf_result.dcf_price_per_share)
        weights.append(0.60)

    if multiples_result and multiples_result.blended_target_price and multiples_result.blended_target_price > 0:
        prices.append(multiples_result.blended_target_price)
        weights.append(0.40)

    if not prices:
        return 0.0

    total_weight = sum(weights)
    return sum(p * w for p, w in zip(prices, weights)) / total_weight


def _calculate_confidence(financial_data, dcf_result, multiples_result) -> int:
    """
    Deterministic scoring rubric. Starts at 5, adjusts based on data quality.
    NOT LLM-generated — consistent and auditable.
    """
    score = 5

    if financial_data.free_cash_flow:       score += 1
    if financial_data.ebitda_ttm:           score += 1
    if financial_data.revenue_growth_yoy:   score += 1
    if financial_data.recent_news:          score += 1
    if dcf_result and multiples_result:     score += 1

    if not financial_data.free_cash_flow:   score -= 1
    if not financial_data.net_income_ttm:   score -= 1
    if financial_data.net_margin and financial_data.net_margin > 1.0:
        score -= 2

    return max(1, min(10, score))
