from src.agent.investment_agent import run_investment_agent
from src.agent.portfolio_memory import collect_portfolio


def main():
    portfolio = collect_portfolio()

    print("\nEnter your investment question.")
    query = input("Question: ").strip()

    result = run_investment_agent(query, portfolio)

    print("\nANSWER")
    print(result["answer"])

    print("\nRETRIEVED SOURCES")
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(f"{i}. {chunk['metadata']['source']} - {chunk['id']}")

    print("\nPORTFOLIO TOOL OUTPUT")
    print(result["portfolio_analysis"])


if __name__ == "__main__":
    main()