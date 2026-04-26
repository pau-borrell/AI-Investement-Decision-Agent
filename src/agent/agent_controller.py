def decide_tools(query: str) -> dict:
    query_lower = query.lower()

    use_portfolio_tools = any(
        word in query_lower
        for word in [
            "portfolio",
            "diversification",
            "concentration",
            "risk",
            "sector",
            "exposure",
            "buy",
            "etf",
            "stock"
        ]
    )

    use_rag = True

    return {
        "use_rag": use_rag,
        "use_portfolio_tools": use_portfolio_tools
    }


def main():
    query = "Should I buy Nvidia if I already own a lot of technology stocks?"
    decision = decide_tools(query)
    print(decision)


if __name__ == "__main__":
    main()