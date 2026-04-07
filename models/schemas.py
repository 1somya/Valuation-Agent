# models/schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# WHY PYDANTIC?
#   Pydantic enforces strict data shapes at runtime. If an agent returns a
#   number where a string is expected (or vice-versa), Pydantic raises an
#   error immediately rather than letting bad data silently corrupt later steps.
#
#   Think of these classes as "contracts" between our three agents:
#     Collector  →  FinancialData
#     Analyst    →  ValuationResult
#     Critic     →  CritiqueResult
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional


class FinancialData(BaseModel):
    """
    Raw financial data fetched by the Collector agent.
    All Optional fields mean the data might be missing — that's fine.
    The Critic will flag those gaps later.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    ticker: str                          # e.g. "AAPL"
    company_name: str                    # e.g. "Apple Inc."
    sector: Optional[str] = None        # e.g. "Technology"
    industry: Optional[str] = None      # e.g. "Consumer Electronics"

    # ── Income Statement metrics ──────────────────────────────────────────────
    revenue_ttm: Optional[float] = None          # Trailing 12-month revenue (USD)
    net_income_ttm: Optional[float] = None       # Trailing 12-month net income
    ebitda_ttm: Optional[float] = None           # EBITDA (earnings before interest/tax/depreciation)
    gross_margin: Optional[float] = None         # e.g. 0.43 = 43%
    net_margin: Optional[float] = None           # Net income / Revenue
    revenue_growth_yoy: Optional[float] = None  # Year-over-year revenue growth %

    # ── Balance Sheet / Cash Flow metrics ────────────────────────────────────
    free_cash_flow: Optional[float] = None       # FCF = Operating CF - CapEx
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None

    # ── Market Data ───────────────────────────────────────────────────────────
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None             # Price / Earnings
    forward_pe: Optional[float] = None           # Based on next year's estimated earnings
    ev_ebitda: Optional[float] = None            # Enterprise Value / EBITDA
    price_to_sales: Optional[float] = None       # Market cap / Revenue
    beta: Optional[float] = None                 # Volatility vs market (1.0 = same as market)

    # ── Contextual / Qualitative ──────────────────────────────────────────────
    recent_news: list[str] = Field(default_factory=list)   # List of news headlines/summaries
    news_sources: list[str] = Field(default_factory=list)  # Corresponding URLs


class DCFResult(BaseModel):
    """
    Output of the deterministic DCF (Discounted Cash Flow) calculation.
    These numbers come from Python math — NOT from the LLM.
    """
    wacc: float                          # Weighted Average Cost of Capital (discount rate)
    terminal_growth_rate: float          # Long-run growth rate (usually ~2-3%, GDP level)
    projected_fcf_year1: float           # Forecasted Free Cash Flow year 1
    projected_fcf_year2: float
    projected_fcf_year3: float
    projected_fcf_year4: float
    projected_fcf_year5: float
    terminal_value: float                # Value of all cash flows after year 5
    enterprise_value: float             # Sum of PV of FCFs + terminal value
    equity_value: float                  # Enterprise value - debt + cash
    dcf_price_per_share: float          # Equity value / shares outstanding
    shares_outstanding: float


class MultiplesResult(BaseModel):
    """
    Output of the comparable multiples valuation.
    Uses sector-average multiples to estimate fair value.
    """
    method: str = "Comparable Multiples"
    ev_ebitda_multiple_used: Optional[float] = None   # The multiple we applied
    pe_multiple_used: Optional[float] = None
    implied_price_ev_ebitda: Optional[float] = None   # Price implied by EV/EBITDA
    implied_price_pe: Optional[float] = None          # Price implied by P/E
    blended_target_price: Optional[float] = None      # Average of the two methods


class SensitivityRow(BaseModel):
    """One row in the sensitivity table (WACC shift × Growth shift → price)."""
    wacc_delta: float        # e.g. -0.01 means WACC was reduced by 1%
    growth_delta: float      # e.g. +0.01 means growth rate increased by 1%
    dcf_price: float         # Resulting DCF price under those assumptions


class ValuationResult(BaseModel):
    """
    Complete output from the Analyst agent.
    Combines deterministic math results + LLM-generated narrative.
    """
    ticker: str
    company_name: str
    valuation_method: str                         # "DCF + Multiples"

    # ── LLM-generated sections ────────────────────────────────────────────────
    analyst_narrative: str                        # LLM writes this: key assumptions & reasoning
    risk_factors: list[str]                       # LLM identifies these from news + financials
    growth_assumption_rationale: str              # LLM explains why it chose the growth rate

    # ── Deterministic calculation results ─────────────────────────────────────
    dcf_result: Optional[DCFResult] = None
    multiples_result: Optional[MultiplesResult] = None
    sensitivity_table: list[SensitivityRow] = Field(default_factory=list)

    # ── Final output ──────────────────────────────────────────────────────────
    target_price_low: float
    target_price_high: float
    target_price_base: float
    current_price: Optional[float] = None

    # ── Evidence ──────────────────────────────────────────────────────────────
    evidence_sources: list[str] = Field(default_factory=list)
    confidence_score: int = Field(ge=1, le=10)   # ge=1, le=10 enforces 1–10 range


class CritiqueResult(BaseModel):
    """
    Output from the Critic agent.
    Flags problems in the Analyst's work.
    """
    ticker: str
    passed_validation: bool                      # True = no major issues found

    # ── Specific checks ───────────────────────────────────────────────────────
    inconsistencies_found: list[str]             # e.g. "Net margin > 100%"
    missing_data_flags: list[str]                # e.g. "FCF data unavailable"
    evidence_gaps: list[str]                     # e.g. "Growth assumption not supported by news"
    unrealistic_assumptions: list[str]           # e.g. "WACC of 3% is too low for this sector"

    # ── Overall assessment ────────────────────────────────────────────────────
    critique_summary: str                        # LLM writes a paragraph summarizing issues
    revised_confidence_score: int = Field(ge=1, le=10)  # May lower the analyst's score
    recommended_actions: list[str]              # e.g. "Re-check FCF source"


class FinalReport(BaseModel):
    """
    The complete report for one company — analyst output + critic review merged.
    This is what gets written to the markdown file at the end.
    """
    financial_data: FinancialData
    valuation: ValuationResult
    critique: CritiqueResult
