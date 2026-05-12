import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent.agent_controller import decide_tools
from src.agent.agent_prompt import build_agent_prompt
from src.rag.retrieval import retrieve_relevant_chunks
from src.rag.slm import generate_answer
from src.tools.analyze_portfolio import analyze_portfolio

DEFAULT_QUERY = "Should I buy Nvidia if my portfolio is already concentrated in technology?"
DEFAULT_PORTFOLIO = {
    "AAPL": 0.45,
    "MSFT": 0.35,
    "SPY": 0.20
}

DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "evaluation_cases.json"


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


def _load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Cases file must contain a list of cases.")
    return data


def _run_pipeline(query: str, portfolio: dict | None, run_simulation: bool) -> dict:
    timings = {}
    start_total = time.perf_counter()

    step_start = time.perf_counter()
    decisions = decide_tools(query)
    timings["decide_tools_ms"] = (time.perf_counter() - step_start) * 1000

    retrieved_chunks = []
    portfolio_analysis = None

    if decisions.get("use_rag"):
        step_start = time.perf_counter()
        retrieved_chunks = retrieve_relevant_chunks(query)
        timings["rag_ms"] = (time.perf_counter() - step_start) * 1000
    else:
        timings["rag_ms"] = 0.0

    if decisions.get("use_portfolio_tools") and portfolio is not None:
        step_start = time.perf_counter()
        portfolio_analysis = analyze_portfolio(portfolio, run_simulation=run_simulation)
        timings["portfolio_ms"] = (time.perf_counter() - step_start) * 1000
    else:
        timings["portfolio_ms"] = 0.0

    step_start = time.perf_counter()
    prompt = build_agent_prompt(query, retrieved_chunks, portfolio_analysis)
    timings["prompt_ms"] = (time.perf_counter() - step_start) * 1000

    step_start = time.perf_counter()
    answer = generate_answer(prompt)
    timings["answer_ms"] = (time.perf_counter() - step_start) * 1000

    timings["total_ms"] = (time.perf_counter() - start_total) * 1000

    return {
        "query": query,
        "decisions": decisions,
        "retrieved_chunks": retrieved_chunks,
        "portfolio_analysis": portfolio_analysis,
        "prompt": prompt,
        "answer": answer,
        "timings": timings
    }


def _print_summary(result: dict) -> None:
    decisions = result.get("decisions", {})
    retrieved_chunks = result.get("retrieved_chunks", [])
    portfolio_analysis = result.get("portfolio_analysis")
    timings = result.get("timings", {})

    print("\nSUMMARY")
    print(f"Use RAG: {decisions.get('use_rag')}")
    print(f"Use portfolio tools: {decisions.get('use_portfolio_tools')}")
    print(f"Retrieved chunks: {len(retrieved_chunks)}")

    if portfolio_analysis:
        diversification = portfolio_analysis.get("diversification", {})
        score = diversification.get("score")
        label = diversification.get("label")
        concentration = portfolio_analysis.get("concentration_risk") or []
        sector_over = portfolio_analysis.get("sector_overexposure") or []
        print(
            "Diversification: "
            f"{score:.4f} ({label})" if isinstance(score, (int, float)) else "Diversification: n/a"
        )
        print(f"Concentration flags: {len(concentration)}")
        print(f"Sector overexposure flags: {len(sector_over)}")
    else:
        print("Diversification: n/a")
        print("Concentration flags: n/a")
        print("Sector overexposure flags: n/a")

    if timings:
        print("Timing (ms):")
        print(
            "  decide_tools={decide_tools_ms:.1f} rag={rag_ms:.1f} portfolio={portfolio_ms:.1f} "
            "prompt={prompt_ms:.1f} answer={answer_ms:.1f} total={total_ms:.1f}".format(**timings)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo investment-agent query.")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="Either a JSON file path or comma-separated TICKER=WEIGHT pairs."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="JSON file containing a list of demo cases."
    )
    parser.add_argument(
        "--case-id",
        "--case_id",
        "--case",
        dest="case_id",
        type=str,
        default=None,
        help="Run a single case from the cases file by ID."
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Optional path to save results as JSONL."
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip Monte Carlo simulation to speed up the demo."
    )
    args = parser.parse_args()

    cases_path = Path(args.cases) if args.cases else DEFAULT_CASES_PATH
    use_cases = args.cases is not None or (args.cases is None and cases_path.exists())
    results = []

    if use_cases:
        cases = _load_cases(cases_path)
        if args.case_id:
            cases = [case for case in cases if case.get("id") == args.case_id]
            if not cases:
                raise ValueError(f"No case found with id '{args.case_id}'.")

        for case in cases:
            case_id = case.get("id", "unknown")
            query = case.get("query", DEFAULT_QUERY)
            portfolio = case.get("portfolio") or DEFAULT_PORTFOLIO

            print("=" * 80)
            print(f"CASE: {case_id}")

            result = _run_pipeline(query, portfolio, run_simulation=not args.skip_simulation)
            result["case_id"] = case_id
            results.append(result)

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
            print("=" * 60)

            _print_summary(result)
    else:
        portfolio = _parse_portfolio(args.portfolio)
        result = _run_pipeline(args.query, portfolio, run_simulation=not args.skip_simulation)
        results.append(result)

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
        print("=" * 60)

        _print_summary(result)

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
