# tools/data_fetcher.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   This file is a "tool layer" — it handles all external API calls.
#   Keeping this separate from the agents means:
#     1. Easy to swap out data sources later (e.g. replace yfinance with Bloomberg)
#     2. Agents stay clean — they just call fetch_financial_data() and move on
#     3. Easier to unit test each function in isolation
#
# DATA SOURCES USED:
#   • yfinance  — pulls from Yahoo Finance, no API key needed
#   • Tavily    — AI-optimized search API for recent news
# ─────────────────────────────────────────────────────────────────────────────

import os
import yfinance as yf
from tavily import TavilyClient
from models.schemas import FinancialData


def fetch_financial_data(ticker: str) -> FinancialData:
    """
    Fetches all financial metrics for a given ticker from Yahoo Finance.

    HOW IT WORKS:
      yfinance downloads a "Ticker" object that contains .info (a big dict
      of 100+ fields), .financials (income statement), .cashflow, etc.
      We extract only the fields we need and pack them into our FinancialData schema.

    IMPORTANT: yfinance can return None for many fields — we handle that
    gracefully by using .get() with None as the default everywhere.
    """
    print(f"  [DataFetcher] Fetching Yahoo Finance data for {ticker}...")

    # Download the ticker object — this makes the HTTP request to Yahoo
    stock = yf.Ticker(ticker)
    info = stock.info  # Big dictionary with all available market data

    # ── Extract market data from .info dict ───────────────────────────────────
    # These fields come from Yahoo Finance's summary page for the stock
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap    = info.get("marketCap")
    pe_ratio      = info.get("trailingPE")
    forward_pe    = info.get("forwardPE")
    ev_ebitda     = info.get("enterpriseToEbitda")
    price_to_sales= info.get("priceToSalesTrailing12Months")
    beta          = info.get("beta")

    # ── Extract income statement metrics ──────────────────────────────────────
    revenue_ttm        = info.get("totalRevenue")
    net_income_ttm     = info.get("netIncomeToCommon")
    ebitda_ttm         = info.get("ebitda")
    gross_margin       = info.get("grossMargins")    # Already a decimal (e.g. 0.43)
    net_margin         = info.get("profitMargins")
    revenue_growth_yoy = info.get("revenueGrowth")  # YoY growth, decimal

    # ── Extract balance sheet / cash flow ─────────────────────────────────────
    free_cash_flow        = info.get("freeCashflow")
    total_debt            = info.get("totalDebt")
    cash_and_equivalents  = info.get("totalCash")

    # ── Company identity ──────────────────────────────────────────────────────
    company_name = info.get("longName", ticker)
    sector       = info.get("sector")
    industry     = info.get("industry")

    print(f"  [DataFetcher] Got data for {company_name} | Price: {current_price} | MCap: {market_cap}")

    return FinancialData(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        industry=industry,
        revenue_ttm=revenue_ttm,
        net_income_ttm=net_income_ttm,
        ebitda_ttm=ebitda_ttm,
        gross_margin=gross_margin,
        net_margin=net_margin,
        revenue_growth_yoy=revenue_growth_yoy,
        free_cash_flow=free_cash_flow,
        total_debt=total_debt,
        cash_and_equivalents=cash_and_equivalents,
        current_price=current_price,
        market_cap=market_cap,
        pe_ratio=pe_ratio,
        forward_pe=forward_pe,
        ev_ebitda=ev_ebitda,
        price_to_sales=price_to_sales,
        beta=beta,
    )


def fetch_news(ticker: str, company_name: str) -> tuple[list[str], list[str]]:
    """
    Searches Tavily for recent news about the company.
    Returns (list_of_summaries, list_of_urls).

    WHY TAVILY INSTEAD OF GOOGLE?
      Tavily is purpose-built for LLM pipelines — it returns clean text
      summaries rather than raw HTML, making it much easier to pass into
      a prompt without token bloat.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    # Gracefully skip if no key is configured
    if not api_key or api_key == "your_tavily_key_here":
        print(f"  [DataFetcher] No Tavily key — skipping news for {ticker}")
        return [], []

    print(f"  [DataFetcher] Fetching news for {company_name} via Tavily...")

    client = TavilyClient(api_key=api_key)

    try:
        # Search for recent earnings / analyst news
        results = client.search(
            query=f"{company_name} {ticker} earnings valuation analyst 2025 2026",
            search_depth="basic",   # "advanced" uses more credits
            max_results=5,
        )

        summaries = []
        urls = []

        for r in results.get("results", []):
            # Each result has: title, url, content (snippet), score
            content = r.get("content", "")
            url     = r.get("url", "")
            title   = r.get("title", "")

            if content:
                summaries.append(f"{title}: {content[:300]}")  # Cap at 300 chars per source
            if url:
                urls.append(url)

        print(f"  [DataFetcher] Found {len(summaries)} news items for {company_name}")
        return summaries, urls

    except Exception as e:
        print(f"  [DataFetcher] Tavily error for {ticker}: {e}")
        return [], []


def get_shares_outstanding(ticker: str) -> float:
    """
    Fetches shares outstanding — needed to convert equity value → price per share.
    Falls back to 1B shares if not found (will be flagged by the Critic).
    """
    stock = yf.Ticker(ticker)
    info  = stock.info
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")

    if shares:
        return float(shares)
    else:
        print(f"  [DataFetcher] WARNING: Could not find shares outstanding for {ticker}, using fallback")
        return 1_000_000_000  # 1 billion shares as a rough fallback
