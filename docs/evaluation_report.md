# Evaluation and Demo

## Scope
Evaluation and demo of the AI Investment Decision Agent.

## Environment
- Backend: Ollama (fine-tuned model: mistral-lora)
- Vector DB: ChromaDB at data/chroma_db

## Evaluation Design

### Functional Evaluation
Checks that the system executes each stage:
- RAG retrieval returns chunks for relevant queries.
- Portfolio tools run and produce diversification, concentration, and sector exposure.
- The final response follows the required structure.

### Qualitative Evaluation
Manual review of:
- Recommendation usefulness.
- Clarity and completeness of reasoning.
- Uncertainty and limitations statements.

### Ablation Study
Three modes are compared:
- model_only: LLM without RAG or tools.
- rag_only: LLM with retrieved context, no tools.
- rag_tools: LLM with retrieved context and portfolio analysis.

## Metrics
Automated metrics computed by the evaluation script:
- Structure compliance: presence of Recommendation, Justification, Portfolio impact, Uncertainty.
- Mentions of diversification, concentration, and sector exposure.
- Average number of retrieved chunks.
- Expected decision match (optional per case).

## How To Run

1) Build the vector DB if not already built:

```bash
python -m src.rag.build_index
```

2) Start the model backend (fine-tuned):

```bash
ollama run mistral-lora
```

3) Run the evaluation:

```bash
python -m src.evaluation.evaluate
```

Outputs are written to the outputs/ folder:
- eval_results_YYYYMMDD_HHMMSS.jsonl
- eval_summary_YYYYMMDD_HHMMSS.json

## Demo Script
A demo script is provided to show a single end-to-end query:

```bash
python scripts/demo.py
```

Optional portfolio input:

```bash
python scripts/demo.py --portfolio "AAPL=0.4,SPY=0.6"
```

Or with a JSON file:

```bash
python scripts/demo.py --portfolio data/portfolio_demo.json
```

## Results

### Automated Summary
Run: 2026-05-10

- Model only: 3/3 structured outputs, avg retrieved = 0.0
	- Mentions: diversification 3/3, concentration 3/3, sector 3/3
	- Errors: 0
- RAG only: 3/3 structured outputs, avg retrieved = 5.0
	- Mentions: diversification 3/3, concentration 3/3, sector 3/3
	- Errors: 0
- RAG + tools: 3/3 structured outputs, avg retrieved = 5.0
	- Mentions: diversification 3/3, concentration 2/3, sector 3/3
	- Errors: 0

### Qualitative Notes
- Responses consistently follow the required structure and cite diversification/sector effects.
- RAG + tools provides portfolio-specific reasoning (weights, sector exposure), which is more grounded than model_only.
- One RAG + tools case omitted explicit concentration wording despite portfolio analysis, suggesting prompt compliance is good but not perfect.

## Demo Notes
Run command: python scripts/demo.py

Demo highlights:
- Input: tech-heavy portfolio (AAPL 45%, MSFT 35%, SPY 20%)
- Output: Recommendation = Neutral
- Rationale: warns about tech sector overexposure (80%) and concentration risk
- Includes portfolio analysis details and uncertainty statement
