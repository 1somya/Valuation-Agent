# tools/financial_calc.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   All actual math lives here — NO LLM is involved in this file.
#   This is a core architectural decision: LLMs are bad at arithmetic,
#   so we use them only for reasoning and narrative, then hand off to
#   pure Python for calculations.
#
# WHAT'S IN HERE:
#   1. run_dcf()            — Discounted Cash Flow valuation
#   2. run_multiples()      — Comparable multiples valuation
#   3. run_sensitivity()    — Sensitivity table (±1% WACC/growth)
#   4. validate_margins()   — Basic sanity checks on the raw data
# ─────────────────────────────────────────────────────────────────────────────

from models.schemas import (
    FinancialData, DCFResult, MultiplesResult, SensitivityRow
)


# ── SECTOR BENCHMARKS ─────────────────────────────────────────────────────────
# These are rough industry-average multiples used when we don't have peer data.
# In a production system you'd pull these from a live database.
# Source: typical public market multiples (2024 approximations)
SECTOR_MULTIPLES = {
    "Technology":            {"ev_ebitda": 28, "pe": 35},
    "Healthcare":            {"ev_ebitda": 16, "pe": 22},
    "Consumer Cyclical":     {"ev_ebitda": 12, "pe": 18},
    "Consumer Defensive":    {"ev_ebitda": 14, "pe": 20},
    "Financials":            {"ev_ebitda":  9, "pe": 13},
    "Energy":                {"ev_ebitda":  8, "pe": 12},
    "Industrials":           {"ev_ebitda": 13, "pe": 18},
    "Communication Services":{"ev_ebitda": 12, "pe": 17},
    "Real Estate":           {"ev_ebitda": 18, "pe": 25},
    "Utilities":             {"ev_ebitda": 11, "pe": 16},
    "Basic Materials":       {"ev_ebitda": 10, "pe": 14},
    "DEFAULT":               {"ev_ebitda": 14, "pe": 20},  # Fallback
}


def _dcf_single(fcf_base: float, growth_rate: float, wacc: float,
                terminal_growth: float, total_debt: float,
                cash: float, shares: float) -> DCFResult:
    """
    Internal helper: runs one DCF calculation with given parameters.
    Called by both run_dcf() and run_sensitivity().

    THE DCF FORMULA (simplified):
      1. Project FCF for 5 years using a constant growth rate
      2. Calculate Terminal Value = FCF_year5 × (1 + g) / (WACC - g)
         This represents the value of ALL cash flows after year 5
      3. Discount everything back to today using WACC
         PV = FCF_t / (1 + WACC)^t
      4. Enterprise Value = sum of all discounted FCFs + discounted terminal value
      5. Equity Value = Enterprise Value - Debt + Cash
      6. Price per Share = Equity Value / Shares Outstanding
    """

    # Step 1: Project 5 years of FCF
    # We grow FCF by `growth_rate` each year compounding
    fcf_years = []
    for year in range(1, 6):
        projected = fcf_base * ((1 + growth_rate) ** year)
        fcf_years.append(projected)

    # Step 2: Discount each year's FCF back to present value
    # PV of FCF_t = FCF_t / (1 + WACC)^t
    pv_fcfs = []
    for t, fcf in enumerate(fcf_years, start=1):
        pv = fcf / ((1 + wacc) ** t)
        pv_fcfs.append(pv)

    # Step 3: Terminal Value — Gordon Growth Model
    # TV = FCF_year5 × (1 + g) / (WACC - g)
    # This captures all value AFTER the 5-year forecast window
    if wacc <= terminal_growth:
        # Safety guard: if growth > WACC, the formula blows up to infinity
        # This would be flagged by the Critic as unrealistic
        terminal_value = fcf_years[-1] * 15  # Use a rough 15x exit multiple instead
    else:
        terminal_value = fcf_years[-1] * (1 + terminal_growth) / (wacc - terminal_growth)

    # Step 4: Discount terminal value to present
    pv_terminal = terminal_value / ((1 + wacc) ** 5)

    # Step 5: Enterprise Value = sum of PV of FCFs + PV of terminal value
    enterprise_value = sum(pv_fcfs) + pv_terminal

    # Step 6: Equity Value = EV - Debt + Cash (bridge from firm to equity holders)
    equity_value = enterprise_value - (total_debt or 0) + (cash or 0)
    equity_value = max(equity_value, 0)  # Can't be negative for our purposes

    # Step 7: Price per share
    price_per_share = equity_value / shares if shares > 0 else 0

    return DCFResult(
        wacc=wacc,
        terminal_growth_rate=terminal_growth,
        projected_fcf_year1=fcf_years[0],
        projected_fcf_year2=fcf_years[1],
        projected_fcf_year3=fcf_years[2],
        projected_fcf_year4=fcf_years[3],
        projected_fcf_year5=fcf_years[4],
        terminal_value=terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        dcf_price_per_share=price_per_share,
        shares_outstanding=shares,
    )


def run_dcf(
    financial_data: FinancialData,
    shares_outstanding: float,
    # These assumptions come FROM the LLM (analyst agent) — not hardcoded
    growth_rate: float = 0.10,      # e.g. 0.10 = 10% annual FCF growth
    wacc: float = 0.09,             # Discount rate (risk-free rate + equity risk premium)
    terminal_growth: float = 0.025, # Long-run GDP-level growth (~2-3%)
) -> DCFResult:
    """
    Runs the full DCF valuation.

    WHAT IS DCF?
      Discounted Cash Flow says: a company is worth the sum of all its future
      cash flows, discounted back to today's dollars. A dollar in 10 years is
      worth less than a dollar today because of inflation and risk.

    INPUTS FROM LLM (assumptions):
      - growth_rate:       How fast will FCF grow? LLM reasons about this from news/trends
      - wacc:              What discount rate to use? LLM estimates from beta + sector
      - terminal_growth:   What's the long-run steady state growth? Usually ~2.5%

    INPUTS FROM YFINANCE (data):
      - free_cash_flow:    Actual trailing FCF (our baseline for projections)
      - total_debt, cash:  Balance sheet items for the equity bridge
    """

    # Use FCF as base. If missing, try to approximate from net income.
    fcf_base = financial_data.free_cash_flow
    if not fcf_base or fcf_base <= 0:
        # Rough approximation: FCF ≈ Net Income (not great but better than nothing)
        fcf_base = financial_data.net_income_ttm
    if not fcf_base or fcf_base <= 0:
        raise ValueError(f"Cannot run DCF for {financial_data.ticker}: no valid FCF or net income data")

    return _dcf_single(
        fcf_base=fcf_base,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        total_debt=financial_data.total_debt or 0,
        cash=financial_data.cash_and_equivalents or 0,
        shares=shares_outstanding,
    )


def run_multiples(
    financial_data: FinancialData,
    shares_outstanding: float,
) -> MultiplesResult:
    """
    Comparable multiples valuation.

    WHAT IS MULTIPLES VALUATION?
      Instead of projecting cash flows, we ask: "What multiple does the market
      assign to similar companies?" Then apply that multiple to our company's
      current earnings/EBITDA to get an implied value.

      Example: If tech peers trade at 20x EBITDA, and our company has $1B EBITDA,
      implied Enterprise Value = $20B.

    We use sector-average multiples from our SECTOR_MULTIPLES table above.
    """

    sector  = financial_data.sector or "DEFAULT"
    benchmarks = SECTOR_MULTIPLES.get(sector, SECTOR_MULTIPLES["DEFAULT"])

    ev_multiple = benchmarks["ev_ebitda"]
    pe_multiple  = benchmarks["pe"]

    implied_price_ev_ebitda = None
    implied_price_pe        = None

    # ── EV/EBITDA Method ──────────────────────────────────────────────────────
    # Implied EV = EBITDA × sector multiple
    # Implied Equity Value = Implied EV - Debt + Cash
    # Implied Price = Equity Value / Shares
    if financial_data.ebitda_ttm and financial_data.ebitda_ttm > 0:
        implied_ev = financial_data.ebitda_ttm * ev_multiple
        implied_equity = (implied_ev
                          - (financial_data.total_debt or 0)
                          + (financial_data.cash_and_equivalents or 0))
        implied_price_ev_ebitda = implied_equity / shares_outstanding if shares_outstanding > 0 else None

    # ── P/E Method ────────────────────────────────────────────────────────────
    # Implied Price = EPS × sector P/E multiple
    # EPS = Net Income / Shares Outstanding
    if financial_data.net_income_ttm and financial_data.net_income_ttm > 0:
        eps = financial_data.net_income_ttm / shares_outstanding
        implied_price_pe = eps * pe_multiple

    # ── Blend the two estimates ───────────────────────────────────────────────
    prices = [p for p in [implied_price_ev_ebitda, implied_price_pe] if p is not None]
    blended = sum(prices) / len(prices) if prices else None

    return MultiplesResult(
        ev_ebitda_multiple_used=ev_multiple,
        pe_multiple_used=pe_multiple,
        implied_price_ev_ebitda=implied_price_ev_ebitda,
        implied_price_pe=implied_price_pe,
        blended_target_price=blended,
    )


def run_sensitivity(
    financial_data: FinancialData,
    shares_outstanding: float,
    base_growth_rate: float,
    base_wacc: float,
    base_terminal_growth: float,
) -> list[SensitivityRow]:
    """
    BONUS: Sensitivity analysis — how much does the price change if we're
    slightly wrong about our key assumptions?

    We vary WACC by -1%, 0%, +1% and growth rate by -1%, 0%, +1%,
    creating a 3×3 grid of 9 scenarios.

    WHY THIS MATTERS: DCF is extremely sensitive to WACC and growth rate.
    A 1% change in WACC can swing the price by 20-30%. This table shows the
    range of reasonable outcomes given our uncertainty.
    """

    fcf_base = financial_data.free_cash_flow or financial_data.net_income_ttm
    if not fcf_base or fcf_base <= 0:
        return []

    rows = []
    for wacc_delta in [-0.01, 0.0, 0.01]:       # -1%, 0%, +1%
        for growth_delta in [-0.01, 0.0, 0.01]:  # -1%, 0%, +1%
            adj_wacc   = base_wacc + wacc_delta
            adj_growth = base_growth_rate + growth_delta

            # Skip if WACC would go negative (unrealistic)
            if adj_wacc <= 0:
                continue

            try:
                result = _dcf_single(
                    fcf_base=fcf_base,
                    growth_rate=adj_growth,
                    wacc=adj_wacc,
                    terminal_growth=base_terminal_growth,
                    total_debt=financial_data.total_debt or 0,
                    cash=financial_data.cash_and_equivalents or 0,
                    shares=shares_outstanding,
                )
                rows.append(SensitivityRow(
                    wacc_delta=wacc_delta,
                    growth_delta=growth_delta,
                    dcf_price=round(result.dcf_price_per_share, 2),
                ))
            except Exception:
                continue  # Skip bad parameter combos silently

    return rows


def validate_margins(financial_data: FinancialData) -> list[str]:
    """
    Basic rule-based sanity checks on the raw data BEFORE we start valuing.
    Returns a list of warning strings. Empty list = all clear.

    These are deterministic checks — no LLM needed.
    The Critic agent will add deeper qualitative checks on top of these.
    """

    warnings = []

    # Net margin > 100% is physically impossible (you can't earn more than revenue)
    if financial_data.net_margin and financial_data.net_margin > 1.0:
        warnings.append(
            f"CRITICAL: Net margin of {financial_data.net_margin:.1%} exceeds 100% — data error"
        )

    # Negative revenue is a red flag (could be a data glitch)
    if financial_data.revenue_ttm and financial_data.revenue_ttm < 0:
        warnings.append("WARNING: Negative revenue reported — possible data error")

    # Very high P/E ratios (>100x) are unusual and worth flagging
    if financial_data.pe_ratio and financial_data.pe_ratio > 100:
        warnings.append(
            f"NOTE: P/E ratio of {financial_data.pe_ratio:.1f}x is very high — "
            "could indicate losses or speculative valuation"
        )

    # Very high EV/EBITDA
    if financial_data.ev_ebitda and financial_data.ev_ebitda > 50:
        warnings.append(
            f"NOTE: EV/EBITDA of {financial_data.ev_ebitda:.1f}x is very high for most sectors"
        )

    # Negative FCF might mean the DCF isn't valid
    if financial_data.free_cash_flow and financial_data.free_cash_flow < 0:
        warnings.append(
            f"WARNING: Negative FCF ({financial_data.free_cash_flow:,.0f}) — "
            "DCF result may not be meaningful"
        )

    # Missing critical fields
    if not financial_data.free_cash_flow and not financial_data.net_income_ttm:
        warnings.append("CRITICAL: No FCF or net income available — DCF cannot be calculated")

    if not financial_data.ebitda_ttm:
        warnings.append("NOTE: EBITDA not available — EV/EBITDA multiples method skipped")

    return warnings
