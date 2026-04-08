### LLM vs Deterministic Split

| Task | Handled By |
|------|-----------|
| Choose growth rate & WACC | **Groq/Llama** (reasoning) |
| Write narrative & identify risks | **Groq/Llama** (language) |
| Distill news into themes | **Groq/Llama** (summarization) |
| Audit evidence consistency | **Groq/Llama** (reading comprehension) |
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
- **GROQ_API_KEY** — from [console.groq.com](https://console.groq.com) (free, no credit card)
- **TAVILY_API_KEY** — from [tavily.com](https://tavily.com) (free tier: 1000 calls/month)

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

**Change valuation assumptions range:** Edit clamp logic in `agents/analyst.py`

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
Machine-readable version of the full report, useful for further processing or backtesting.

---

## Valuation Methods

### DCF (Discounted Cash Flow)
Projects free cash flow 5 years forward using a LLM-chosen growth rate, discounts back using WACC, and adds a terminal value using the Gordon Growth Model. All arithmetic is done in pure Python.

### Comparable Multiples
Applies sector-average EV/EBITDA and P/E multiples to current financials to derive an implied price. Blended 50/50, then combined with DCF at 60/40 weighting.

### Sensitivity Analysis
Runs the DCF 9 times across a 3×3 grid of ±1% WACC and growth rate variations to show the range of reasonable outcomes.

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

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Groq API — Llama 3.3 70B (free tier) |
| Financial Data | yfinance (Yahoo Finance) |
| News Retrieval | Tavily Search API |
| Data Validation | Pydantic v2 |
| Calculations | Pure Python (no LLM arithmetic) |

---

## Limitations

- Financial data from Yahoo Finance may have delays or gaps
- Sector multiples are static benchmarks, not live peer comparisons
- DCF assumes constant growth rate (single-stage model)
- News context limited to Tavily search results