import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent.investment_agent import run_investment_agent


DEFAULT_QUERY = "Should I buy Nvidia if my portfolio is already concentrated in technology?"
DEFAULT_PORTFOLIO = {
    "AAPL": 0.45,
    "MSFT": 0.35,
    "SPY": 0.20
}


def _parse_portfolio(value: str | None) -> dict:
    if not value:
        return DEFAULT_PORTFOLIO

    candidate_path = Path(value)
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    portfolio = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Portfolio items must be in TICKER=WEIGHT format.")
        ticker, weight = item.split("=", 1)
        portfolio[ticker.strip().upper()] = float(weight)

    return portfolio


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo investment-agent query.")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="Either a JSON file path or comma-separated TICKER=WEIGHT pairs."
    )
    args = parser.parse_args()

    portfolio = _parse_portfolio(args.portfolio)
    result = run_investment_agent(args.query, portfolio)

    print("QUERY")
    print(result["query"])
    print("=" * 60)

    print("\nDECISIONS")
    print(result["decisions"])
    print("=" * 60)

    print("\nRETRIEVED SOURCES")
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(f"{i}. {chunk['metadata']['source']} - {chunk['id']}")
    print("=" * 60)

    print("\nPORTFOLIO ANALYSIS")
    print(result["portfolio_analysis"])
    print("=" * 60)

    print("\nANSWER")
    print(result["answer"])


if __name__ == "__main__":
    main()
