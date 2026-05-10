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

This project develops an AI agent that assists investors in deciding whether to buy a particular stock or fund based on their existing investment portfolio.

Traditional financial assistants usually provide general market summaries or sentiment analysis but do not personalize recommendations according to the investor’s portfolio composition. Because of this, the same investment recommendation may not be appropriate for different investors.

The goal of this project is to build a portfolio-aware AI decision agent that combines financial knowledge retrieval, quantitative portfolio analysis tools, and reasoning capabilities from a fine-tuned Small Language Model (SLM).

The system will integrate several components including Retrieval-Augmented Generation (RAG), portfolio analysis tools, and an agent architecture capable of selecting tools and combining their outputs. Financial knowledge will be stored in a vector database and retrieved during inference to ground the model’s reasoning and reduce hallucinations.

The agent will produce structured recommendations including:
- A final decision (Buy / Do Not Buy / Neutral)
- Key reasoning
- Portfolio diversification analysis
- Concentration risk evaluation
- Possible alternative investments

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

## Technologies

Python  
Mistral 7B (fine-tuned SLM)  
LangChain / LlamaIndex  
ChromaDB (vector database)  
Sentence Transformers (embeddings)  
HuggingFace Transformers  
PEFT / LoRA fine-tuning  
yfinance (market data)

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/pau-borrell/AI-Investment-Decision-Agent
cd AI-Investment-Decision-Agent
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Running the Project

### 1. Build the RAG Vector Database

This step processes the financial knowledge files, creates embeddings, and stores them in ChromaDB.

```bash
python -m src.rag.build_index
```

You should see output indicating:

* documents loaded
* chunks created
* embeddings generated
* database stored successfully


### 2. Start the Language Model

By default, the project uses Ollama. Make sure Ollama is installed and running locally.

Start the model:

```bash
ollama run mistral
```

Leave this running in the background.

If you want to use a fine-tuned HuggingFace + LoRA adapter instead of Ollama, set:

```bash
set SLM_BACKEND=hf
set HF_BASE_MODEL_PATH=C:\path\to\mistral-7b-instruct
set HF_LORA_ADAPTER_PATH=C:\path\to\AI-Investment-Decision-Agent\models\mistral7b-lora
```

Then run the agent as usual. This runs locally and can be slow on CPU.


### 3. Run the Investment Agent

```bash
python -m src.agent.run_agent
```

## Fine-Tune the SLM (LoRA)

Generate a synthetic dataset:

```bash
python scripts/generate_synthetic_finetune_data.py --train-size 2000 --val-size 200
```

Then follow the Kaggle workflow in:

* docs/kaggle_finetune_notebook.md

When training finishes, download the adapter folder and place it at:

* models/mistral7b-lora

Finally, set the HF environment variables shown above and run the agent.

## Use the LoRA Model in Ollama (Merge + GGUF)

Ollama only runs GGUF models. To use the fine-tuned LoRA in Ollama, first merge the
adapter into the base model on a GPU machine, then convert to GGUF.

### 1. Merge LoRA into Base (GPU)

On Kaggle or another GPU machine, upload:

* Base Mistral model folder (HF format)
* LoRA adapter folder (models/mistral7b-lora)

Then run:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/kaggle/input/your-base-model-folder"
ADAPTER = "/kaggle/input/your-adapter-folder"
OUT = "mistral7b-merged"

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)

model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
tokenizer.save_pretrained(OUT)
```

Download the `mistral7b-merged` folder.

### 2. Convert to GGUF (Local)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python -m pip install -r requirements.txt
python convert_hf_to_gguf.py ..\mistral7b-merged --outfile ..\mistral7b-merged.gguf

# Quantize (required in newer llama.cpp versions)
cmake -S . -B build
cmake --build build --config Release
build\bin\quantize ..\mistral7b-merged.gguf ..\mistral7b-merged-q4_0.gguf q4_0

# Linux/macOS
# ./build/bin/quantize ../mistral7b-merged.gguf ../mistral7b-merged-q4_0.gguf q4_0
```

### 3. Create an Ollama Model

Create a Modelfile:

```
FROM ./mistral7b-merged.gguf
PARAMETER temperature 0.2
PARAMETER top_p 0.9
```

Then build and run:

```bash
ollama create mistral-lora -f Modelfile
ollama run mistral-lora
```

### The fine-tuned model

If you have an Ollama account, you can pull the model so no need to do the fine-tuning yourself:

```bash
ollama pull baranorbi12/mistral-lora
ollama run baranorbi12/mistral-lora
```

Model page:
https://ollama.com/baranorbi12/mistral-lora

## Example Usage

Enter your portfolio:

```text
Holding: AAPL 3000
Holding: SPY 5000
Holding: done
```

Then enter your question:

```text
Question: Should I buy Nvidia if I already own a lot of technology stocks?
```

## Expected Output

The agent will return:

* Recommendation (Buy / Do Not Buy / Neutral)
* Justification based on retrieved knowledge
* Portfolio impact analysis
* Uncertainty explanation

It will also display:

* Retrieved knowledge sources
* Portfolio analysis results


## Repository

GitHub Repository:
https://github.com/pau-borrell/AI-Investement-Decision-Agent
