import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent.agent_prompt import build_agent_prompt
from src.rag.retrieval import retrieve_relevant_chunks
from src.rag.slm import generate_answer
from src.tools.analyze_portfolio import analyze_portfolio


CASES_PATH = Path("data/evaluation_cases.json")
OUTPUTS_DIR = Path("outputs")
MODES = ["model_only", "rag_only", "rag_tools"]


@dataclass
class CaseResult:
    case_id: str
    mode: str
    answer: str
    metrics: dict
    retrieved_count: int
    retrieved_sources: list[str]
    portfolio_analysis_present: bool
    errors: list[str]


def _extract_decision(answer: str) -> str | None:
    match = re.search(r"Recommendation:\s*(Buy|Do\s+Not\s+Buy|Neutral)", answer, re.IGNORECASE)
    if not match:
        return None
    decision = match.group(1)
    return re.sub(r"\s+", " ", decision).strip().title()


def _evaluate_answer(answer: str, expected_decision: str | None) -> dict:
    has_recommendation = _extract_decision(answer) is not None
    has_justification = "Justification:" in answer
    has_portfolio_impact = "Portfolio impact:" in answer
    has_uncertainty = "Uncertainty:" in answer
    mentions_diversification = bool(re.search(r"diversif", answer, re.IGNORECASE))
    mentions_concentration = bool(re.search(r"concentration", answer, re.IGNORECASE))
    mentions_sector = bool(re.search(r"sector", answer, re.IGNORECASE))

    expected_match = None
    if expected_decision:
        expected_match = _extract_decision(answer) == expected_decision.title()

    return {
        "has_recommendation": has_recommendation,
        "has_justification": has_justification,
        "has_portfolio_impact": has_portfolio_impact,
        "has_uncertainty": has_uncertainty,
        "mentions_diversification": mentions_diversification,
        "mentions_concentration": mentions_concentration,
        "mentions_sector": mentions_sector,
        "expected_decision_match": expected_match
    }


def _run_case(case: dict[str, Any], mode: str) -> CaseResult:
    errors: list[str] = []
    retrieved_chunks: list[dict] = []
    portfolio_analysis = None

    if mode in {"rag_only", "rag_tools"}:
        try:
            retrieved_chunks = retrieve_relevant_chunks(case["query"])
        except Exception as exc:
            errors.append(f"retrieval_error: {exc}")
            retrieved_chunks = []

    if mode == "rag_tools":
        try:
            portfolio_analysis = analyze_portfolio(case["portfolio"], run_simulation=False)
        except Exception as exc:
            errors.append(f"portfolio_analysis_error: {exc}")
            portfolio_analysis = None

    prompt = build_agent_prompt(case["query"], retrieved_chunks, portfolio_analysis)

    try:
        answer = generate_answer(prompt)
    except Exception as exc:
        errors.append(f"generation_error: {exc}")
        answer = ""

    retrieved_sources = [chunk["metadata"]["source"] for chunk in retrieved_chunks]

    metrics = _evaluate_answer(answer, case.get("expected_decision"))

    return CaseResult(
        case_id=case["id"],
        mode=mode,
        answer=answer,
        metrics=metrics,
        retrieved_count=len(retrieved_chunks),
        retrieved_sources=retrieved_sources,
        portfolio_analysis_present=portfolio_analysis is not None,
        errors=errors
    )


def _load_cases() -> list[dict[str, Any]]:
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Evaluation cases not found at {CASES_PATH}.")

    with CASES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize_results(results: list[CaseResult]) -> dict:
    summary = {
        "total": len(results),
        "by_mode": {},
    }

    for result in results:
        mode_entry = summary["by_mode"].setdefault(result.mode, {
            "count": 0,
            "has_recommendation": 0,
            "has_justification": 0,
            "has_portfolio_impact": 0,
            "has_uncertainty": 0,
            "mentions_diversification": 0,
            "mentions_concentration": 0,
            "mentions_sector": 0,
            "expected_decision_match": 0,
            "errors": 0,
            "avg_retrieved": 0.0
        })

        mode_entry["count"] += 1
        mode_entry["has_recommendation"] += int(result.metrics["has_recommendation"])
        mode_entry["has_justification"] += int(result.metrics["has_justification"])
        mode_entry["has_portfolio_impact"] += int(result.metrics["has_portfolio_impact"])
        mode_entry["has_uncertainty"] += int(result.metrics["has_uncertainty"])
        mode_entry["mentions_diversification"] += int(result.metrics["mentions_diversification"])
        mode_entry["mentions_concentration"] += int(result.metrics["mentions_concentration"])
        mode_entry["mentions_sector"] += int(result.metrics["mentions_sector"])

        if result.metrics["expected_decision_match"] is True:
            mode_entry["expected_decision_match"] += 1

        if result.errors:
            mode_entry["errors"] += 1

        mode_entry["avg_retrieved"] += result.retrieved_count

    for mode_entry in summary["by_mode"].values():
        if mode_entry["count"]:
            mode_entry["avg_retrieved"] = mode_entry["avg_retrieved"] / mode_entry["count"]

    return summary


def main() -> None:
    cases = _load_cases()
    results: list[CaseResult] = []

    for case in cases:
        for mode in MODES:
            results.append(_run_case(case, mode))

    summary = _summarize_results(results)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUTS_DIR / f"eval_results_{timestamp}.jsonl"
    summary_path = OUTPUTS_DIR / f"eval_summary_{timestamp}.json"

    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.__dict__, ensure_ascii=True) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=True))

    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
