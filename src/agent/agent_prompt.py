def format_retrieved_context(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "No retrieved context available."

    parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["metadata"]["source"]
        text = chunk["text"]
        parts.append(f"Source {i}: {source}\n{text}")

    return "\n\n".join(parts)


def format_portfolio_analysis(analysis: dict | None) -> str:
    if analysis is None:
        return "No portfolio analysis available."

    lines = []

    lines.append("Portfolio weights:")
    for ticker, weight in analysis["portfolio_weights"].items():
        lines.append(f"- {ticker}: {weight:.2%}")

    lines.append("\nDiversification:")
    lines.append(f"- Score: {analysis['diversification']['score']:.4f}")
    lines.append(f"- Label: {analysis['diversification']['label']}")

    lines.append("\nConcentration risk:")
    if analysis["concentration_risk"]:
        for risk in analysis["concentration_risk"]:
            lines.append(f"- {risk['ticker']}: {risk['weight']:.2%}")
    else:
        lines.append("- No asset above threshold.")

    lines.append("\nSector exposure:")
    for sector, weight in analysis["sector_exposure"].items():
        lines.append(f"- {sector}: {weight:.2%}")

    lines.append("\nSector overexposure:")
    if analysis["sector_overexposure"]:
        for risk in analysis["sector_overexposure"]:
            lines.append(f"- {risk['sector']}: {risk['weight']:.2%}")
    else:
        lines.append("- No sector above threshold.")

    if "return_simulation" in analysis:
        lines.append("\nMonte Carlo simulation:")
        for key, value in analysis["return_simulation"].items():
            lines.append(f"- {key}: {value:.2%}")

    return "\n".join(lines)


def build_agent_prompt(query: str, retrieved_chunks: list[dict], portfolio_analysis: dict | None) -> str:
    retrieved_context = format_retrieved_context(retrieved_chunks)
    analysis_text = format_portfolio_analysis(portfolio_analysis)

    prompt = f"""
You are a portfolio-aware financial education assistant.

Use ONLY the retrieved financial knowledge and portfolio analysis below.
If the available information is insufficient, say "insufficient data".
Do not invent facts.
Do not give guaranteed predictions.
Do not present this as personalized financial advice.

Decision rubric (pick ONE):
- Do Not Buy: concentration risk or sector overexposure is flagged, or diversification is Low.
- Buy: diversification improves meaningfully and concentration/sector risks are not elevated.
- Neutral: mixed signals without a clear risk or improvement.

User question:
{query}

Retrieved financial knowledge:
{retrieved_context}

Portfolio analysis:
{analysis_text}

Return the answer in exactly this structure and use one of the three labels:

Recommendation: Buy / Do Not Buy / Neutral

Justification:
Explain the reasoning using the retrieved knowledge and portfolio analysis.

Portfolio impact:
Explain how the investment may affect diversification, concentration, sector exposure, and risk.

Uncertainty:
Mention limitations of the analysis.
""".strip()

    return prompt