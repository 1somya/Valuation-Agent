# AI Valuation Agent

An agentic system that automatically collects financial data, generates structured valuation analyses, and performs self-correction to mitigate hallucinations.

---

## Architecture

```
main.py  (Orchestrator)
    │
    ├── agents/collector.py   →  Phase A: Data Collection
    │       ├── tools/data_fetcher.py  (yfinance + Tavily)
    │       └── Claude API  (news summarization only)
    │
    ├── agents/analyst.py     →  Phase B & C: Valuation + Memo
    │       ├── Claude API  (assumption reasoning, narrative, risks)
    │       └── tools/financial_calc.py  (ALL math — no LLM)
    │
    └── agents/critic.py      →  Phase D: Validation
            ├── Deterministic checks  (margin limits, data flags)
            └── Claude API  (evidence audit, qualitative critique)
```

### LLM vs Deterministic Split

| Task | Handled By |
|------|-----------|
| Choose growth rate & WACC | **Claude** (reasoning) |
| Write narrative & identify risks | **Claude** (language) |
| Distill news into themes | **Claude** (summarization) |
| Audit evidence consistency | **Claude** (reading comprehension) |
| DCF cash flow projections | **Python** (math) |
| Terminal value calculation | **Python** (math) |
| Multiples implied prices | **Python** (math) |
| Sensitivity analysis grid | **Python** (math) |
| Data validation rules | **Python** (deterministic rules) |
| Confidence scoring | **Python** (rubric) |

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add:
- **ANTHROPIC_API_KEY** — from [console.anthropic.com](https://console.anthropic.com/)
- **TAVILY_API_KEY** — from [tavily.com](https://tavily.com/) (free tier: 1000 calls/month)

> The system works without Tavily — news context will just be skipped.

### 3. Run

```bash
python main.py
```

Reports are saved to `output/reports/` as both `.md` and `.json` files.

---

## Customization

**Change companies:** Edit `TICKERS` in `main.py`

```python
TICKERS = ["TSLA", "AMZN", "META"]
```

**Change valuation assumptions range:** Edit `_clamp` logic in `agents/analyst.py`

**Change sector multiples:** Edit `SECTOR_MULTIPLES` in `tools/financial_calc.py`

---

## Output Structure

Each company generates:

### Markdown Report (`TICKER_timestamp_valuation.md`)
Human-readable memo with:
- Executive summary + target price
- Financial snapshot table
- DCF projections (5-year)
- Multiples analysis
- Sensitivity table (3×3 grid)
- Risk factors
- Critic review with flagged issues

### JSON File (`TICKER_timestamp_valuation.json`)
Machine-readable version of the full `FinalReport` object, useful for further processing or backtesting.

---

## Validation Checks

The Critic agent performs:

| Check | Type |
|-------|------|
| Net margin > 100% | Deterministic |
| P/E positive but net income negative | Deterministic |
| DCF implies >200% upside | Deterministic |
| WACC outside 5–20% range | Deterministic |
| Terminal growth > 4% | Deterministic |
| Narrative unsupported by data | LLM |
| Missing risk factors | LLM |
| Internal logical contradictions | LLM |

---

## Sensitivity Analysis

The system automatically generates a 3×3 sensitivity table showing DCF price under 9 combinations of:
- WACC: −1%, base, +1%
- Growth Rate: −1%, base, +1%

This quantifies model uncertainty and shows which assumption has the most impact.

---

## Limitations

- Financial data from Yahoo Finance may have delays or gaps
- Sector multiples are static benchmarks, not live peer comparisons
- DCF assumes constant growth rate (single-stage model)
- News context limited to Tavily search results
