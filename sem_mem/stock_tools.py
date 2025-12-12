"""
Stock market data tools for Sema (Semantic Memory Agent).

Provides stock price retrieval using Polygon.io API:
- Latest trade price
- Previous day OHLC data
- Price change calculations

Requires POLYGON_API_KEY environment variable.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


# Polygon API configuration
POLYGON_BASE_URL = "https://api.polygon.io"


def get_polygon_api_key() -> Optional[str]:
    """Get Polygon API key from environment."""
    # Try to load .env if not already loaded
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        for path in [Path(__file__).parent.parent / ".env", Path(".env")]:
            if path.exists():
                load_dotenv(path)
                break
    except ImportError:
        pass

    return os.getenv("POLYGON_API_KEY")


@dataclass
class StockPriceResult:
    """Result from fetching stock price."""
    ticker: str
    price: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    prev_close: Optional[float]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    volume: Optional[int]
    timestamp: Optional[str]
    success: bool
    error: Optional[str] = None


def fetch_stock_price(
    ticker: str,
    api_key: Optional[str] = None,
) -> StockPriceResult:
    """
    Fetch current stock price and previous day data from Polygon.io.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        api_key: Polygon API key (uses env var if not provided)

    Returns:
        StockPriceResult with price data or error
    """
    import requests

    # Get API key
    if not api_key:
        api_key = get_polygon_api_key()

    if not api_key:
        return StockPriceResult(
            ticker=ticker.upper(),
            price=None,
            change=None,
            change_percent=None,
            prev_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            timestamp=None,
            success=False,
            error="POLYGON_API_KEY not configured. Set it in .env file.",
        )

    ticker = ticker.upper().strip()

    # Fetch previous day data (has OHLC)
    prev_url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"

    try:
        prev_response = requests.get(
            prev_url,
            params={"apiKey": api_key, "adjusted": "true"},
            timeout=10,
        )
        prev_response.raise_for_status()
        prev_data = prev_response.json()

        if prev_data.get("status") == "ERROR":
            return StockPriceResult(
                ticker=ticker,
                price=None,
                change=None,
                change_percent=None,
                prev_close=None,
                open=None,
                high=None,
                low=None,
                volume=None,
                timestamp=None,
                success=False,
                error=prev_data.get("error", "Unknown error from Polygon API"),
            )

        results = prev_data.get("results", [])
        if not results:
            return StockPriceResult(
                ticker=ticker,
                price=None,
                change=None,
                change_percent=None,
                prev_close=None,
                open=None,
                high=None,
                low=None,
                volume=None,
                timestamp=None,
                success=False,
                error=f"No data found for ticker: {ticker}",
            )

        bar = results[0]
        close_price = bar.get("c")
        open_price = bar.get("o")
        high_price = bar.get("h")
        low_price = bar.get("l")
        volume = bar.get("v")
        timestamp_ms = bar.get("t")

        # Convert timestamp
        timestamp_str = None
        if timestamp_ms:
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            timestamp_str = dt.strftime("%Y-%m-%d")

        # Try to get latest trade for more current price
        latest_price = close_price
        try:
            trade_url = f"{POLYGON_BASE_URL}/v2/last/trade/{ticker}"
            trade_response = requests.get(
                trade_url,
                params={"apiKey": api_key},
                timeout=10,
            )
            if trade_response.status_code == 200:
                trade_data = trade_response.json()
                trade_result = trade_data.get("results", {})
                if trade_result and "p" in trade_result:
                    latest_price = trade_result["p"]
                    # Update timestamp if we got latest trade
                    trade_ts = trade_result.get("t")
                    if trade_ts:
                        # Polygon uses nanoseconds for trade timestamp
                        dt = datetime.fromtimestamp(trade_ts / 1_000_000_000)
                        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # Fall back to close price if latest trade fails
            pass

        # Calculate change from previous close (open) to current
        change = None
        change_percent = None
        if latest_price and open_price:
            change = latest_price - open_price
            change_percent = (change / open_price) * 100 if open_price else None

        return StockPriceResult(
            ticker=ticker,
            price=latest_price,
            change=round(change, 2) if change else None,
            change_percent=round(change_percent, 2) if change_percent else None,
            prev_close=close_price,
            open=open_price,
            high=high_price,
            low=low_price,
            volume=int(volume) if volume else None,
            timestamp=timestamp_str,
            success=True,
        )

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return StockPriceResult(
                ticker=ticker,
                price=None,
                change=None,
                change_percent=None,
                prev_close=None,
                open=None,
                high=None,
                low=None,
                volume=None,
                timestamp=None,
                success=False,
                error="Invalid Polygon API key. Check your POLYGON_API_KEY.",
            )
        elif e.response.status_code == 404:
            return StockPriceResult(
                ticker=ticker,
                price=None,
                change=None,
                change_percent=None,
                prev_close=None,
                open=None,
                high=None,
                low=None,
                volume=None,
                timestamp=None,
                success=False,
                error=f"Ticker not found: {ticker}",
            )
        else:
            return StockPriceResult(
                ticker=ticker,
                price=None,
                change=None,
                change_percent=None,
                prev_close=None,
                open=None,
                high=None,
                low=None,
                volume=None,
                timestamp=None,
                success=False,
                error=f"API error: {e.response.status_code}",
            )
    except requests.exceptions.Timeout:
        return StockPriceResult(
            ticker=ticker,
            price=None,
            change=None,
            change_percent=None,
            prev_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            timestamp=None,
            success=False,
            error="Request timed out",
        )
    except Exception as e:
        return StockPriceResult(
            ticker=ticker,
            price=None,
            change=None,
            change_percent=None,
            prev_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            timestamp=None,
            success=False,
            error=f"Error fetching stock price: {str(e)}",
        )


def format_stock_price_result(result: StockPriceResult) -> str:
    """
    Format a stock price result as context for the LLM.

    Args:
        result: StockPriceResult from fetch_stock_price()

    Returns:
        Formatted string for injection into prompt context
    """
    if not result.success:
        return f"Failed to get stock price for {result.ticker}: {result.error}"

    lines = [f"Stock data for {result.ticker}:"]

    if result.price is not None:
        lines.append(f"  Current price: ${result.price:.2f}")

    if result.change is not None and result.change_percent is not None:
        direction = "+" if result.change >= 0 else ""
        lines.append(f"  Change: {direction}${result.change:.2f} ({direction}{result.change_percent:.2f}%)")

    if result.open is not None:
        lines.append(f"  Open: ${result.open:.2f}")

    if result.high is not None and result.low is not None:
        lines.append(f"  Day range: ${result.low:.2f} - ${result.high:.2f}")

    if result.prev_close is not None:
        lines.append(f"  Previous close: ${result.prev_close:.2f}")

    if result.volume is not None:
        lines.append(f"  Volume: {result.volume:,}")

    if result.timestamp:
        lines.append(f"  As of: {result.timestamp}")

    return "\n".join(lines)


def get_stock_price_tool_definition(api_format: str = "responses") -> Dict[str, Any]:
    """
    Get the OpenAI function tool definition for stock_price.

    This allows the LLM to proactively request stock prices.

    Args:
        api_format: Either "responses" (OpenAI Responses API) or "completions" (Chat Completions).

    Returns:
        Tool definition dict for the specified API format.
    """
    description = (
        "Get the current stock price and trading data for a ticker symbol. "
        "Returns the latest price, daily change, open/high/low, and volume. "
        "Use this when the user asks about stock prices, market data, or wants "
        "to know how a stock is performing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft, 'GOOGL' for Alphabet)",
            },
        },
        "required": ["ticker"],
    }

    if api_format == "responses":
        # OpenAI Responses API format (flat structure)
        return {
            "type": "function",
            "name": "stock_price",
            "description": description,
            "parameters": parameters,
            "strict": False,
        }
    else:
        # Chat Completions API format (nested under "function")
        return {
            "type": "function",
            "function": {
                "name": "stock_price",
                "description": description,
                "parameters": parameters,
            },
        }
