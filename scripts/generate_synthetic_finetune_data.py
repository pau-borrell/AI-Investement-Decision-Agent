import argparse
import json
import random
from pathlib import Path

SECTORS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "ADBE", "CRM"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Industrials": ["CAT", "GE", "BA", "HON", "MMM"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Utilities": ["NEE", "DUK", "SO"],
    "Real Estate": ["AMT", "PLD", "O"],
}

ETFS = {
    "Broad Market": ["SPY", "VOO", "VTI"],
    "Tech ETF": ["QQQ", "XLK"],
    "Dividend ETF": ["VYM", "SCHD"],
    "International ETF": ["VXUS", "VEA"],
}

KNOWLEDGE_SNIPPETS = [
    "Diversification reduces unsystematic risk by spreading exposure across assets.",
    "Sector overexposure can amplify drawdowns when a sector underperforms.",
    "Broad-market ETFs provide diversified exposure across multiple sectors.",
    "High concentration in a single stock increases idiosyncratic risk.",
    "Rebalancing can improve risk-adjusted returns over long horizons.",
]

QUERY_TEMPLATES = [
    "Should I buy {target} if I already own a lot of {sector} stocks?",
    "Is it a good idea to add {target} to my portfolio?",
    "Would buying {target} improve diversification?",
    "Should I avoid {target} given my current portfolio exposure?",
    "Is {target} a reasonable addition if I want to reduce risk?",
]


def build_ticker_map():
    ticker_to_sector = {}
    for sector, tickers in SECTORS.items():
        for ticker in tickers:
            ticker_to_sector[ticker] = sector
    for _, tickers in ETFS.items():
        for ticker in tickers:
            ticker_to_sector[ticker] = "ETF"
    return ticker_to_sector


def sample_portfolio(rng, ticker_to_sector, min_items=3, max_items=6):
    all_tickers = list(ticker_to_sector.keys())
    num_items = rng.randint(min_items, max_items)
    chosen = rng.sample(all_tickers, num_items)
    weights = [rng.random() for _ in chosen]
    total = sum(weights)
    normalized = [w / total for w in weights]
    portfolio = []
    for ticker, weight in zip(chosen, normalized):
        portfolio.append({
            "ticker": ticker,
            "weight": weight,
            "sector": ticker_to_sector[ticker]
        })
    return portfolio


def portfolio_to_text(portfolio):
    parts = []
    for item in portfolio:
        pct = item["weight"] * 100
        parts.append(f"{item['ticker']} {pct:.1f}%")
    return ", ".join(parts)


def sector_exposure(portfolio):
    exposure = {}
    for item in portfolio:
        sector = item["sector"]
        exposure[sector] = exposure.get(sector, 0.0) + item["weight"]
    return exposure


def decide_recommendation(target_sector, exposure, query):
    sector_weight = exposure.get(target_sector, 0.0)

    if "reduce risk" in query or "diversification" in query:
        if target_sector == "ETF" or sector_weight < 0.25:
            return "Buy"

    if target_sector != "ETF":
        if sector_weight >= 0.35:
            return "Do Not Buy"
        if sector_weight >= 0.25:
            return "Neutral"
        return "Buy"

    if sector_weight >= 0.45:
        return "Neutral"
    return "Buy"


def build_example(rng, ticker_to_sector):
    portfolio = sample_portfolio(rng, ticker_to_sector)
    exposure = sector_exposure(portfolio)

    use_etf = rng.random() < 0.25
    if use_etf:
        etf_group = rng.choice(list(ETFS.values()))
        target = rng.choice(etf_group)
        target_sector = "ETF"
        sector_hint = "broad market"
    else:
        target_sector = rng.choice(list(SECTORS.keys()))
        target = rng.choice(SECTORS[target_sector])
        sector_hint = target_sector.lower()

    query_template = rng.choice(QUERY_TEMPLATES)
    query = query_template.format(target=target, sector=sector_hint)

    recommendation = decide_recommendation(target_sector, exposure, query.lower())
    knowledge = rng.choice(KNOWLEDGE_SNIPPETS)

    portfolio_text = portfolio_to_text(portfolio)

    justification = (
        f"Your current exposure to {target_sector} is {exposure.get(target_sector, 0.0) * 100:.1f}%, "
        f"which influences whether adding {target} improves balance. {knowledge}"
    )

    portfolio_impact = (
        f"Adding {target} would {'increase' if target_sector != 'ETF' else 'maintain'} "
        f"exposure to {target_sector}. This could {'raise' if recommendation != 'Buy' else 'reduce'} "
        "concentration risk depending on position size."
    )

    uncertainty = (
        "This is based on portfolio weights and general principles only. "
        "Market conditions and personal goals could change the outcome."
    )

    text = (
        "Instruction: " + query + "\n"
        "Context: Portfolio " + portfolio_text + "\n"
        "Retrieved knowledge: " + knowledge + "\n"
        "Answer:\n"
        f"Recommendation: {recommendation}\n\n"
        "Justification: " + justification + "\n\n"
        "Portfolio impact: " + portfolio_impact + "\n\n"
        "Uncertainty: " + uncertainty
    )

    return {"text": text}


def generate_dataset(seed, train_size, val_size, output_dir):
    rng = random.Random(seed)
    ticker_to_sector = build_ticker_map()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as train_file:
        for _ in range(train_size):
            example = build_example(rng, ticker_to_sector)
            train_file.write(json.dumps(example, ensure_ascii=True) + "\n")

    with val_path.open("w", encoding="utf-8") as val_file:
        for _ in range(val_size):
            example = build_example(rng, ticker_to_sector)
            val_file.write(json.dumps(example, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning data.")
    parser.add_argument("--output-dir", default="data/finetune", help="Output folder.")
    parser.add_argument("--train-size", type=int, default=200, help="Number of training examples.")
    parser.add_argument("--val-size", type=int, default=40, help="Number of validation examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generate_dataset(args.seed, args.train_size, args.val_size, output_dir)
    print(f"Wrote {args.train_size} train and {args.val_size} val examples to {output_dir}.")


if __name__ == "__main__":
    main()
