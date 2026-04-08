# agents/collector.py
# ─────────────────────────────────────────────────────────────────────────────
# COLLECTOR AGENT — Phase A
#
# RESPONSIBILITY:
#   Gather all the raw data we need before any analysis happens.
#   This agent is mostly deterministic (yfinance calls) with one small LLM
#   call to summarize the news into structured risk themes.
#
# INPUTS:  ticker symbol (e.g. "AAPL")
# OUTPUTS: FinancialData object (fully populated, or partially populated with
#          warnings about what's missing)
#
# AGENT PATTERN:
#   This is a simple "fetch + enrich" agent. It doesn't need a reasoning loop.
#   More complex agents (Analyst, Critic) will use multi-step LLM calls.
# ─────────────────────────────────────────────────────────────────────────────

from groq import Groq
import os
from models.schemas import FinancialData
from tools.data_fetcher import fetch_financial_data, fetch_news


def run_collector(ticker: str) -> FinancialData:
    """
    Main entry point for the Collector agent.
    Fetches financial data + news, then returns a populated FinancialData object.
    """

    print(f"\n{'='*60}")
    print(f"  COLLECTOR AGENT: Starting data collection for {ticker}")
    print(f"{'='*60}")

    # ── Step 1: Fetch structured financial data from Yahoo Finance ────────────
    # This is fully deterministic — no LLM involved
    financial_data = fetch_financial_data(ticker)

    # ── Step 2: Fetch recent news via Tavily search ───────────────────────────
    # Also deterministic — just an API call
    news_summaries, news_urls = fetch_news(ticker, financial_data.company_name)

    # Attach news to our financial data object
    financial_data.recent_news   = news_summaries
    financial_data.news_sources  = news_urls

    # ── Step 3: (Optional LLM) Summarize news into key themes ────────────────
    # We use Claude here ONLY if we got news back.
    # Claude's job: extract the 3 most important themes from raw news snippets.
    # This is much better than passing 5 raw news blobs into the analyst prompt.
    if news_summaries:
        financial_data = _enrich_news_with_llm(financial_data)

    print(f"  [Collector] Done. Collected {len(financial_data.recent_news)} news items.")
    return financial_data


def _enrich_news_with_llm(financial_data: FinancialData) -> FinancialData:
    """
    Uses Groq to distill raw news snippets into 3 concise theme bullets.
    This makes the Analyst's prompt much cleaner and more focused.

    LLM ROLE HERE: Summarization / information distillation
    NOT USED FOR: Any numbers or calculations
    """

    print(f"  [Collector] Using LLM to distill news themes for {financial_data.ticker}...")

    # Format the raw news into a readable block for the prompt
    news_block = "\n".join([f"- {item}" for item in financial_data.recent_news[:5]])

    prompt = f"""You are a financial analyst assistant preprocessing news data.

    Company: {financial_data.company_name} ({financial_data.ticker})
    Sector: {financial_data.sector}

    Here are recent news snippets:
    {news_block}

    Extract the 3 most important themes relevant to valuation (growth drivers, risks, or strategic changes).
    Return ONLY a bullet list of 3 concise themes, each under 30 words. No intro text."""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    distilled = raw
    theme_lines = [line.strip("- ").strip() for line in distilled.split("\n") if line.strip()]

    

    # Keep themes as the first entries, then append any leftover raw news
    financial_data.recent_news = theme_lines

    return financial_data
