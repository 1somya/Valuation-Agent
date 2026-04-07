# main.py
# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
#
# This is the entry point. It:
#   1. Defines the list of companies to analyze
#   2. Runs each company through the 3-agent pipeline
#   3. Saves all reports to output/reports/
#   4. Prints a summary table at the end
#
# PIPELINE FLOW:
#   For each ticker:
#     Collector Agent  →  FinancialData
#     Analyst Agent    →  ValuationResult
#     Critic Agent     →  CritiqueResult
#     Report Generator →  .md file
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads ANTHROPIC_API_KEY and TAVILY_API_KEY into os.getenv()
load_dotenv()

from agents.collector import run_collector
from agents.analyst   import run_analyst
from agents.critic    import run_critic
from models.schemas   import FinalReport
from tools.report_generator import generate_report


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# These are the companies we'll analyze.
# You can change these to any publicly traded US tickers.
TICKERS = [
    "AAPL",   # Apple — large cap tech, high margins
    "MSFT",   # Microsoft — cloud + AI growth story
    "NVDA",   # Nvidia — high-growth, high-multiple
]

# Where to save reports
OUTPUT_DIR = "output/reports"


def run_pipeline(ticker: str) -> FinalReport:
    """
    Runs the full 3-agent pipeline for a single ticker.
    Returns a FinalReport combining all three agents' outputs.
    """

    print(f"\n{'#'*60}")
    print(f"  STARTING PIPELINE FOR: {ticker}")
    print(f"{'#'*60}")

    # ── PHASE A: Collect ──────────────────────────────────────────────────────
    # Collector fetches financial data from yfinance + news from Tavily
    # OUTPUT: FinancialData object
    financial_data = run_collector(ticker)

    # ── PHASE B & C: Analyze ──────────────────────────────────────────────────
    # Analyst uses Claude to choose assumptions, Python to do the math,
    # then compiles a structured ValuationResult
    # OUTPUT: ValuationResult object
    valuation = run_analyst(financial_data)

    # ── PHASE D: Critique ─────────────────────────────────────────────────────
    # Critic runs rule-based checks + a second LLM call to validate the work
    # OUTPUT: CritiqueResult object
    critique = run_critic(financial_data, valuation)

    # ── Package everything into a FinalReport ─────────────────────────────────
    report = FinalReport(
        financial_data=financial_data,
        valuation=valuation,
        critique=critique,
    )

    # ── Save Markdown report ──────────────────────────────────────────────────
    report_path = generate_report(report, output_dir=OUTPUT_DIR)

    # ── Also save raw JSON for debugging / further processing ─────────────────
    # This lets you inspect every field programmatically
    json_path = report_path.replace(".md", ".json")
    with open(json_path, "w") as fh:
        fh.write(report.model_dump_json(indent=2))

    print(f"\n  ✅ {ticker} complete → {report_path}")

    return report


def print_summary_table(reports: list[FinalReport]) -> None:
    """
    Prints a concise summary table of all companies side-by-side.
    Makes it easy to compare at a glance.
    """
    print(f"\n\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")

    # Header
    print(f"{'Ticker':<8} {'Company':<25} {'Current':<10} {'Target':<10} "
          f"{'Upside':<10} {'Conf':<6} {'Valid':<8}")
    print("-" * 80)

    for r in reports:
        v = r.valuation
        f = r.financial_data
        c = r.critique

        current = f"${f.current_price:.2f}" if f.current_price else "N/A"
        target  = f"${v.target_price_base:.2f}" if v.target_price_base else "N/A"

        if f.current_price and v.target_price_base:
            upside_pct = (v.target_price_base - f.current_price) / f.current_price
            upside = f"{upside_pct:+.1%}"
        else:
            upside = "N/A"

        status  = "✅" if c.passed_validation else "⚠️"
        company = f.company_name[:24]  # Truncate long names

        print(f"{f.ticker:<8} {company:<25} {current:<10} {target:<10} "
              f"{upside:<10} {c.revised_confidence_score:<6} {status:<8}")

    print(f"{'='*80}")
    print(f"\nReports saved to: {OUTPUT_DIR}/")
    print("Each company has a .md (human-readable) and .json (machine-readable) file.")


def main():
    """
    Entry point. Validates config, runs the pipeline, prints summary.
    """

    # ── Validate API keys before starting ─────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your_groq_key_here":
        print("❌ ERROR: GROQ_API_KEY not set in .env file")
        print("   Get your free key from: https://console.groq.com")
        return

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key or tavily_key == "your_tavily_key_here":
        print("⚠️  WARNING: TAVILY_API_KEY not set — news context will be skipped")
        print("   Get a free key from: https://tavily.com/")

    print(f"\n🚀 AI Valuation Agent — Starting analysis for {len(TICKERS)} companies")
    print(f"   Tickers: {', '.join(TICKERS)}")
    print(f"   Output: {OUTPUT_DIR}/\n")

    # ── Run pipeline for each ticker ───────────────────────────────────────────
    reports = []
    failed  = []

    for ticker in TICKERS:
        try:
            report = run_pipeline(ticker)
            reports.append(report)
        except Exception as e:
            print(f"\n❌ Pipeline failed for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(ticker)

    # ── Print summary ──────────────────────────────────────────────────────────
    if reports:
        print_summary_table(reports)

    if failed:
        print(f"\n⚠️  Failed tickers: {', '.join(failed)}")
        print("   Check error messages above for details.")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
