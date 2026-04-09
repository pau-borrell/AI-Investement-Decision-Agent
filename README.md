# AI-Investement-Decision-Agent
AI Stock/Fund Buy Decision Agent that evaluates whether a user should purchase a given asset based on their current portfolio.


Team Erasmus:
Natalia Patallo Suarez, Pau Borrel Bullich, Guilherme Chagas Silva, Norbert Bara, Ilayda Tiryaki


This project builds an AI agent capable of advising whether a user should buy a stock or fund.
The system combines portfolio-aware reasoning, quantitative analysis tools, and Retrieval-Augmented Generation (RAG) over financial knowledge.
The agent integrates external tools such as diversification analysis and valuation signals to produce structured investment recommendations grounded in retrieved financial principles.


User Query
   ↓
Agent Controller (LLM)
   ↓
Tool Selection
   ↓
RAG Retrieval (financial knowledge)
   ↓
Analytical Tools (portfolio analysis)
   ↓
Final Recommendation Generator
