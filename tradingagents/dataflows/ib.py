"""Interactive Brokers data vendor using ib_insync.

Requires TWS or IB Gateway running locally with the API enabled.
Connection parameters are read from the dataflow config:
    ib_host  (default: "127.0.0.1")
    ib_port  (default: 7497  — TWS paper; use 7496 for TWS live, 4002 for Gateway paper)
    ib_client_id (default: 1)

Install: pip install ib_insync
"""

from __future__ import annotations

import os
import contextlib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Annotated

import pandas as pd

from .config import get_config
from .stockstats_utils import StockstatsUtils, _clean_dataframe


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_ib():
    """Return a connected IB instance.  Caller is responsible for disconnect."""
    try:
        from ib_insync import IB
    except ImportError as exc:
        raise ImportError(
            "ib_insync is required for the IB data vendor. "
            "Install it with: pip install ib_insync"
        ) from exc

    cfg = get_config()
    host = cfg.get("ib_host", os.environ.get("IB_HOST", "127.0.0.1"))
    port = int(cfg.get("ib_port", os.environ.get("IB_PORT", 7497)))
    client_id = int(cfg.get("ib_client_id", os.environ.get("IB_CLIENT_ID", 1)))

    ib = IB()
    ib.connect(host, port, clientId=client_id, readonly=True, timeout=20)
    return ib


@contextlib.contextmanager
def _ib_session():
    """Context manager that opens and cleanly closes an IB connection."""
    ib = _get_ib()
    try:
        yield ib
    finally:
        ib.disconnect()


def _resolve_contract(ib, symbol: str):
    """Qualify and return an STK contract for *symbol* on SMART/USD.

    Falls back to the first qualified contract if multiple are returned.
    Supports exchange-suffixed symbols (e.g. RY.TO → TSX).
    """
    from ib_insync import Stock

    parts = symbol.upper().split(".")
    if len(parts) == 2:
        ticker_sym, exchange_suffix = parts
        # Map common yfinance suffixes to IB exchange codes
        _SUFFIX_MAP = {
            "TO": "TSX",
            "L":  "LSE",
            "PA": "SBF",
            "DE": "IBIS",
            "HK": "SEHK",
            "T":  "TSEJ",
            "AX": "ASX",
        }
        exchange = _SUFFIX_MAP.get(exchange_suffix, exchange_suffix)
        contract = Stock(ticker_sym, exchange, "USD")
    else:
        contract = Stock(symbol, "SMART", "USD")

    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise ValueError(f"IB could not qualify contract for symbol '{symbol}'")
    return qualified[0]


# ---------------------------------------------------------------------------
# OHLCV (core_stock_apis)
# ---------------------------------------------------------------------------

def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch OHLCV bar data from IB for the given date range."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    delta_days = (end_dt - start_dt).days + 1
    duration = f"{max(delta_days, 1)} D"

    try:
        with _ib_session() as ib:
            contract = _resolve_contract(ib, symbol)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt.strftime("%Y%m%d %H:%M:%S"),
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

        if not bars:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

        df = pd.DataFrame(
            [
                {
                    "Date": b.date,
                    "Open": round(b.open, 2),
                    "High": round(b.high, 2),
                    "Low": round(b.low, 2),
                    "Close": round(b.close, 2),
                    "Volume": b.volume,
                }
                for b in bars
            ]
        )

        header = (
            f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
            f"# Total records: {len(df)}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + df.to_csv(index=False)

    except Exception as exc:
        return f"Error retrieving stock data for '{symbol}' from IB: {exc}"


# ---------------------------------------------------------------------------
# Technical indicators (technical_indicators)
# Computed locally via stockstats on IB OHLCV — same approach as yfinance.
# ---------------------------------------------------------------------------

def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute technical indicators using IB historical data via stockstats."""
    from stockstats import wrap

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_dt - relativedelta(days=look_back_days + 250)  # extra for warmup

    ohlcv = get_stock_data(symbol, before.strftime("%Y-%m-%d"), curr_date)
    if ohlcv.startswith("Error") or ohlcv.startswith("No data"):
        return ohlcv

    lines = ohlcv.splitlines()
    csv_lines = [l for l in lines if not l.startswith("#") and l.strip()]
    df = pd.read_csv(pd.io.common.StringIO("\n".join(csv_lines)))
    df = _clean_dataframe(df)

    try:
        df_stats = wrap(df)
        df_stats[indicator]
    except Exception as exc:
        return f"Error computing indicator '{indicator}': {exc}"

    before_filter = curr_dt - relativedelta(days=look_back_days)
    ind_string = ""
    for _, row in df_stats.iterrows():
        row_date = pd.to_datetime(row["Date"])
        if row_date < before_filter:
            continue
        val = row.get(indicator)
        val_str = "N/A" if pd.isna(val) else str(val)
        ind_string += f"{row_date.strftime('%Y-%m-%d')}: {val_str}\n"

    return (
        f"## {indicator} values from {before_filter.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
    )


# ---------------------------------------------------------------------------
# Fundamentals (fundamental_data)
# IB provides fundamentals via reqFundamentalData (requires IB research sub).
# Falls back to a helpful message when not subscribed.
# ---------------------------------------------------------------------------

def _parse_ratios_xml(xml_text: str) -> dict:
    """Parse key metrics from IB's ReportRatios XML into a flat dict."""
    import xml.etree.ElementTree as ET

    wanted = {
        "MKTCAP": "Market Cap",
        "TTMPE": "PE Ratio (TTM)",
        "TTMEPS": "EPS (TTM)",
        "PRICE2BK": "Price to Book",
        "TTMDIVYLD": "Dividend Yield",
        "BETA": "Beta",
        "TTMROEPCT": "Return on Equity",
        "TTMROAPCT": "Return on Assets",
        "TTMNPMGN": "Profit Margin",
        "TTMOPMGN": "Operating Margin",
        "TTMREVPS": "Revenue Per Share",
        "QCURRATIO": "Current Ratio",
        "QTOTD2EQ": "Debt to Equity",
    }
    result = {}
    try:
        root = ET.fromstring(xml_text)
        for ratio in root.iter("Ratio"):
            field_id = ratio.get("FieldName", "")
            if field_id in wanted and ratio.text:
                result[wanted[field_id]] = ratio.text.strip()
    except ET.ParseError:
        pass
    return result


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (unused, IB returns latest)"] = None,
) -> str:
    """Get company fundamentals from IB (requires IB research data subscription)."""
    try:
        with _ib_session() as ib:
            contract = _resolve_contract(ib, ticker)
            xml = ib.reqFundamentalData(contract, "ReportRatios")

        if not xml:
            return f"No fundamentals data returned by IB for '{ticker}'. Check your IB research subscription."

        metrics = _parse_ratios_xml(xml)
        if not metrics:
            return f"Could not parse fundamentals XML for '{ticker}'."

        header = (
            f"# Company Fundamentals for {ticker.upper()} (via Interactive Brokers)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        lines = [f"{k}: {v}" for k, v in metrics.items()]
        return header + "\n".join(lines)

    except Exception as exc:
        return f"Error retrieving fundamentals for '{ticker}' from IB: {exc}"


def _parse_financial_statements_xml(xml_text: str, statement: str, freq: str) -> pd.DataFrame:
    """Parse IB ReportFinancialStatements XML into a DataFrame for one statement."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"XML parse error: {exc}") from exc

    # statement: "BAL" | "INC" | "CAS"
    # freq: "Annual" | "Interim"
    period_type = "Annual" if freq.lower() == "annual" else "Interim"
    cols: dict[str, dict] = {}  # period_end -> {item_name: value}

    for fin_stmt in root.iter("FinancialStatement"):
        if fin_stmt.get("Type") != statement:
            continue
        for period in fin_stmt.iter("FiscalPeriod"):
            if period.get("Type") != period_type:
                continue
            end_date = period.get("EndDate", "")
            if end_date not in cols:
                cols[end_date] = {}
            for item in period.iter("lineItem"):
                name = item.get("displayName") or item.get("coaCode", "")
                value = item.text
                if name and value:
                    cols[end_date][name] = value

    if not cols:
        return pd.DataFrame()

    df = pd.DataFrame(cols)
    df.index.name = "Item"
    return df


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get balance sheet from IB financial statements."""
    try:
        with _ib_session() as ib:
            contract = _resolve_contract(ib, ticker)
            xml = ib.reqFundamentalData(contract, "ReportFinancialStatements")

        if not xml:
            return f"No financial statement data from IB for '{ticker}'."

        df = _parse_financial_statements_xml(xml, "BAL", freq)
        if df.empty:
            return f"No balance sheet data found for '{ticker}' ({freq})."

        header = (
            f"# Balance Sheet for {ticker.upper()} ({freq}, via Interactive Brokers)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + df.to_csv()

    except Exception as exc:
        return f"Error retrieving balance sheet for '{ticker}' from IB: {exc}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get cash flow statement from IB financial statements."""
    try:
        with _ib_session() as ib:
            contract = _resolve_contract(ib, ticker)
            xml = ib.reqFundamentalData(contract, "ReportFinancialStatements")

        if not xml:
            return f"No financial statement data from IB for '{ticker}'."

        df = _parse_financial_statements_xml(xml, "CAS", freq)
        if df.empty:
            return f"No cash flow data found for '{ticker}' ({freq})."

        header = (
            f"# Cash Flow for {ticker.upper()} ({freq}, via Interactive Brokers)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + df.to_csv()

    except Exception as exc:
        return f"Error retrieving cash flow for '{ticker}' from IB: {exc}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get income statement from IB financial statements."""
    try:
        with _ib_session() as ib:
            contract = _resolve_contract(ib, ticker)
            xml = ib.reqFundamentalData(contract, "ReportFinancialStatements")

        if not xml:
            return f"No financial statement data from IB for '{ticker}'."

        df = _parse_financial_statements_xml(xml, "INC", freq)
        if df.empty:
            return f"No income statement data found for '{ticker}' ({freq})."

        header = (
            f"# Income Statement for {ticker.upper()} ({freq}, via Interactive Brokers)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + df.to_csv()

    except Exception as exc:
        return f"Error retrieving income statement for '{ticker}' from IB: {exc}"


# ---------------------------------------------------------------------------
# Insider transactions — not available via IB API; return graceful message.
# ---------------------------------------------------------------------------

def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Insider transactions are not available through the IB API."""
    return (
        f"Insider transactions for '{ticker}' are not available via the IB data vendor. "
        "Switch the news_data vendor to 'yfinance' or 'alpha_vantage' for this data."
    )


# ---------------------------------------------------------------------------
# News (news_data)
# IB provides news via reqNewsArticle / reqHistoricalNews (requires subscription).
# ---------------------------------------------------------------------------

def get_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch news headlines for a stock from IB (requires IB news subscription)."""
    try:
        with _ib_session() as ib:
            contract = _resolve_contract(ib, ticker)
            # reqHistoricalNews returns up to 300 items
            headlines = ib.reqHistoricalNews(
                contract.conId,
                providerCodes="BRFG+DJNL",  # Briefing.com + Dow Jones; adjust as needed
                startDateTime=start_date,
                endDateTime=end_date,
                totalResults=50,
            )

        if not headlines:
            return f"No news found for '{ticker}' from {start_date} to {end_date} via IB."

        lines = [
            f"# News for {ticker.upper()} from {start_date} to {end_date} (via Interactive Brokers)",
            f"# Total articles: {len(headlines)}",
            "",
        ]
        for h in headlines:
            lines.append(f"[{h.time}] {h.providerCode}: {h.headline}")

        return "\n".join(lines)

    except Exception as exc:
        return f"Error retrieving news for '{ticker}' from IB: {exc}"


def get_global_news(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Global market news is not available as a standalone feed via the IB API."""
    return (
        "Global news is not available via the IB data vendor. "
        "Switch the news_data vendor to 'yfinance' or 'alpha_vantage' for global news."
    )
