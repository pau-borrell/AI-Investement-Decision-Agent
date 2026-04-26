from src.tools.portfolio_input import normalize_portfolio


def collect_portfolio() -> dict:
    holdings = []

    print("Enter your current portfolio.")
    print("Use ticker and amount/percentage.")
    print("Example: AAPL 3000 or SPY 45")
    print("Type 'done' when finished.\n")

    while True:
        user_input = input("Holding: ").strip()

        if user_input.lower() == "done":
            break

        parts = user_input.split()

        if len(parts) != 2:
            print("Invalid format. Use: TICKER AMOUNT")
            continue

        ticker, amount = parts

        try:
            amount = float(amount)
        except ValueError:
            print("Amount must be a number.")
            continue

        holdings.append({"ticker": ticker, "amount": amount})

    return normalize_portfolio(holdings)


def main():
    portfolio = collect_portfolio()
    print(portfolio)


if __name__ == "__main__":
    main()