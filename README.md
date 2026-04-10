# AI Investment Decision Agent

## Team Name
Erasmus

## Team Members
Natalia Patallo Suarez  
Pau Borrell Bullich  
Guilherme Chagas Silva  
Norbert Bara  
Ilayda Tiryaki  

## Project Description

This project develops an AI agent capable of assisting investors in deciding whether to purchase a specific stock or fund. The system produces personalized recommendations by considering the user's current portfolio, financial knowledge retrieved from external sources, and quantitative portfolio analysis tools.

Unlike traditional financial assistants that provide generic advice, this system evaluates investment decisions from the perspective of the individual investor. The agent analyzes how a potential purchase would affect diversification, concentration risk, and portfolio balance.

The system combines several modern AI techniques including Retrieval-Augmented Generation (RAG), tool-based reasoning, and fine-tuning of a Small Language Model (SLM). Financial knowledge will be stored in a vector database and retrieved during inference to ground the model's reasoning and reduce hallucinations.

## Architecture Overview

The system follows an agent-based architecture composed of several components:

User Query  
↓  
LLM Agent Controller  
↓  
Tool Selection  
↓  
Portfolio Memory + Analytical Tools  
↓  
Financial Knowledge Retrieval (RAG)  
↓  
Final Structured Recommendation  

The agent integrates retrieved financial knowledge with outputs from analytical tools to generate a recommendation including:

• Final decision (Buy / Do Not Buy / Neutral)  
• Explanation of reasoning  
• Portfolio impact analysis  
• Diversification assessment  
• Suggested alternatives

## Technologies

Python  
Mistral 7B (fine-tuned SLM)  
LangChain / LlamaIndex  
ChromaDB (vector database)  
Sentence Transformers (embeddings)  
HuggingFace Transformers  
PEFT / LoRA fine-tuning  
yfinance (market data)
