from src.agent.agent_controller import decide_tools
from src.agent.agent_prompt import build_agent_prompt
from src.rag.retrieval import retrieve_relevant_chunks
from src.rag.slm import generate_answer
from src.tools.analyze_portfolio import analyze_portfolio


def run_investment_agent(query: str, portfolio: dict | None = None) -> dict:
    decisions = decide_tools(query)

    retrieved_chunks = []
    portfolio_analysis = None

    if decisions["use_rag"]:
        retrieved_chunks = retrieve_relevant_chunks(query)

    if decisions["use_portfolio_tools"] and portfolio is not None:
        portfolio_analysis = analyze_portfolio(portfolio)

    prompt = build_agent_prompt(query, retrieved_chunks, portfolio_analysis)
    answer = generate_answer(prompt)

    return {
        "query": query,
        "decisions": decisions,
        "retrieved_chunks": retrieved_chunks,
        "portfolio_analysis": portfolio_analysis,
        "prompt": prompt,
        "answer": answer
    }


def main():
    query = "Should I buy Nvidia if my portfolio is already concentrated in technology?"
    portfolio = {
        "AAPL": 0.40,
        "SPY": 0.45,
        "TSLA": 0.15
    }

    result = run_investment_agent(query, portfolio)

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