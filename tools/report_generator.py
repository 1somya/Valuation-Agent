# tools/report_generator.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Takes a FinalReport (analyst output + critic review) and writes a clean
#   human-readable Markdown file. This is what you'd actually hand to a client.
#
# WHY MARKDOWN?
#   Easy to read as plain text, renders beautifully on GitHub, and can be
#   converted to PDF/Word if needed. No special tools required.
# ─────────────────────────────────────────────────────────────────────────────

import os
from datetime import datetime
from models.schemas import FinalReport


def generate_report(report: FinalReport, output_dir: str = "output/reports") -> str:
    """
    Generates a markdown valuation memo and saves it to disk.
    Returns the path to the saved file.
    """

    os.makedirs(output_dir, exist_ok=True)

    v = report.valuation    # shorthand
    f = report.financial_data
    c = report.critique

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename  = f"{output_dir}/{f.ticker}_{timestamp}_valuation.md"

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# Valuation Memo: {f.company_name} ({f.ticker})",
        f"**Generated:** {datetime.now().strftime('%B %d, %Y %H:%M')}  ",
        f"**Method:** {v.valuation_method}  ",
        f"**Confidence Score:** {v.confidence_score}/10  ",
        f"**Validation Status:** {'✅ Passed' if c.passed_validation else '⚠️ Issues Found'}",
        "",
    ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    lines += [
        "## Executive Summary",
        "",
        v.analyst_narrative,
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Current Price | ${f.current_price:.2f if f.current_price else 'N/A'} |",
        f"| Target Price (Base) | **${v.target_price_base:.2f}** |",
        f"| Target Price Range | ${v.target_price_low:.2f} – ${v.target_price_high:.2f} |",
    ]

    if f.current_price and v.target_price_base:
        upside = (v.target_price_base - f.current_price) / f.current_price
        lines.append(f"| Implied Upside/Downside | {upside:+.1%} |")

    lines += [
        f"| Sector | {f.sector or 'N/A'} |",
        f"| Market Cap | ${f.market_cap/1e9:.1f}B |" if f.market_cap else "| Market Cap | N/A |",
        "",
    ]

    # ── Financial Snapshot ────────────────────────────────────────────────────
    lines += [
        "## Financial Snapshot",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Revenue (TTM) | ${f.revenue_ttm/1e9:.2f}B |" if f.revenue_ttm else "| Revenue (TTM) | N/A |",
        f"| Net Income (TTM) | ${f.net_income_ttm/1e9:.2f}B |" if f.net_income_ttm else "| Net Income (TTM) | N/A |",
        f"| EBITDA (TTM) | ${f.ebitda_ttm/1e9:.2f}B |" if f.ebitda_ttm else "| EBITDA (TTM) | N/A |",
        f"| Free Cash Flow | ${f.free_cash_flow/1e9:.2f}B |" if f.free_cash_flow else "| Free Cash Flow | N/A |",
        f"| Gross Margin | {f.gross_margin:.1%} |" if f.gross_margin else "| Gross Margin | N/A |",
        f"| Net Margin | {f.net_margin:.1%} |" if f.net_margin else "| Net Margin | N/A |",
        f"| Revenue Growth YoY | {f.revenue_growth_yoy:.1%} |" if f.revenue_growth_yoy else "| Revenue Growth YoY | N/A |",
        f"| P/E Ratio | {f.pe_ratio:.1f}x |" if f.pe_ratio else "| P/E Ratio | N/A |",
        f"| EV/EBITDA | {f.ev_ebitda:.1f}x |" if f.ev_ebitda else "| EV/EBITDA | N/A |",
        f"| Beta | {f.beta:.2f} |" if f.beta else "| Beta | N/A |",
        "",
    ]

    # ── DCF Results ───────────────────────────────────────────────────────────
    if v.dcf_result:
        d = v.dcf_result
        lines += [
            "## DCF Valuation",
            "",
            "### Assumptions (LLM-generated, bounded by rules)",
            f"- **WACC:** {d.wacc:.1%}",
            f"- **Terminal Growth Rate:** {d.terminal_growth_rate:.1%}",
            "",
            "### Growth Rationale",
            v.growth_assumption_rationale,
            "",
            "### Projected Free Cash Flows (Python-calculated)",
            "",
            "| Year | Projected FCF |",
            "|------|--------------|",
            f"| Year 1 | ${d.projected_fcf_year1/1e9:.3f}B |",
            f"| Year 2 | ${d.projected_fcf_year2/1e9:.3f}B |",
            f"| Year 3 | ${d.projected_fcf_year3/1e9:.3f}B |",
            f"| Year 4 | ${d.projected_fcf_year4/1e9:.3f}B |",
            f"| Year 5 | ${d.projected_fcf_year5/1e9:.3f}B |",
            "",
            f"- **Terminal Value:** ${d.terminal_value/1e9:.2f}B",
            f"- **Enterprise Value:** ${d.enterprise_value/1e9:.2f}B",
            f"- **Equity Value:** ${d.equity_value/1e9:.2f}B",
            f"- **DCF Price Per Share:** **${d.dcf_price_per_share:.2f}**",
            "",
        ]

    # ── Multiples Results ─────────────────────────────────────────────────────
    if v.multiples_result:
        m = v.multiples_result
        lines += [
            "## Comparable Multiples Valuation",
            "",
            "| Method | Multiple Used | Implied Price |",
            "|--------|--------------|---------------|",
        ]
        if m.ev_ebitda_multiple_used and m.implied_price_ev_ebitda:
            lines.append(f"| EV/EBITDA | {m.ev_ebitda_multiple_used:.1f}x | ${m.implied_price_ev_ebitda:.2f} |")
        if m.pe_multiple_used and m.implied_price_pe:
            lines.append(f"| P/E | {m.pe_multiple_used:.1f}x | ${m.implied_price_pe:.2f} |")
        if m.blended_target_price:
            lines.append(f"| **Blended** | — | **${m.blended_target_price:.2f}** |")
        lines.append("")

    # ── Sensitivity Table ─────────────────────────────────────────────────────
    if v.sensitivity_table:
        lines += [
            "## Sensitivity Analysis (DCF Price per Share)",
            "",
            "Rows = WACC change | Columns = Growth Rate change",
            "",
        ]

        # Build 3×3 grid
        wacc_deltas   = sorted(set(r.wacc_delta for r in v.sensitivity_table))
        growth_deltas = sorted(set(r.growth_delta for r in v.sensitivity_table))

        # Table header
        header_cols = " | ".join([f"Growth {d:+.0%}" for d in growth_deltas])
        lines.append(f"| WACC | {header_cols} |")
        lines.append("|" + "------|" * (len(growth_deltas) + 1))

        # Table rows
        for wd in wacc_deltas:
            row_prices = []
            for gd in growth_deltas:
                match = next((r.dcf_price for r in v.sensitivity_table
                              if r.wacc_delta == wd and r.growth_delta == gd), None)
                row_prices.append(f"${match:.2f}" if match else "N/A")
            lines.append(f"| WACC {wd:+.0%} | {' | '.join(row_prices)} |")

        lines.append("")

    # ── Risk Factors ──────────────────────────────────────────────────────────
    lines += [
        "## Risk Factors (LLM-identified)",
        "",
    ]
    for risk in v.risk_factors:
        lines.append(f"- {risk}")
    lines.append("")

    # ── Evidence Sources ──────────────────────────────────────────────────────
    lines += [
        "## Evidence Sources",
        "",
    ]
    for src in v.evidence_sources:
        lines.append(f"- {src}")
    lines.append("")

    # ── Critic Review ─────────────────────────────────────────────────────────
    lines += [
        "## Critic Agent Review",
        "",
        f"**Revised Confidence Score:** {c.revised_confidence_score}/10  ",
        f"**Validation Status:** {'✅ Passed' if c.passed_validation else '⚠️ Issues Found'}",
        "",
        "### Critique Summary",
        c.critique_summary,
        "",
    ]

    if c.inconsistencies_found:
        lines += ["### ⚠️ Inconsistencies Detected"]
        for item in c.inconsistencies_found:
            lines.append(f"- {item}")
        lines.append("")

    if c.missing_data_flags:
        lines += ["### 📋 Missing Data"]
        for item in c.missing_data_flags:
            lines.append(f"- {item}")
        lines.append("")

    if c.unrealistic_assumptions:
        lines += ["### 🚩 Unrealistic Assumptions"]
        for item in c.unrealistic_assumptions:
            lines.append(f"- {item}")
        lines.append("")

    if c.evidence_gaps:
        lines += ["### 🔍 Evidence Gaps"]
        for item in c.evidence_gaps:
            lines.append(f"- {item}")
        lines.append("")

    if c.recommended_actions:
        lines += ["### ✅ Recommended Actions"]
        for item in c.recommended_actions:
            lines.append(f"- {item}")
        lines.append("")

    # ── Disclosure ────────────────────────────────────────────────────────────
    lines += [
        "---",
        "*This report was generated by an automated AI valuation agent for educational/research purposes only.*  ",
        "*It does not constitute investment advice. All figures should be independently verified.*",
    ]

    content = "\n".join(lines)

    with open(filename, "w") as fh:
        fh.write(content)

    print(f"  [Report] Saved: {filename}")
    return filename
