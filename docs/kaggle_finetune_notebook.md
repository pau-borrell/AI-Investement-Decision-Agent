# Kaggle LoRA Fine-Tuning Notebook Template

Use this as a ready-to-run Kaggle notebook. Copy each section into its own notebook cell.

## 0. Notebook Setup

- Enable GPU: Notebook Settings -> Accelerator -> GPU
- Upload your dataset files (train.jsonl, val.jsonl) to the notebook

If you do not have dataset files yet, you can generate them locally:

```bash
python scripts/generate_synthetic_finetune_data.py --train-size 2000 --val-size 200
```

Then upload:

- data/finetune/train.jsonl
- data/finetune/val.jsonl

---

## 1. Install Dependencies

```bash
pip install -q transformers datasets peft accelerate bitsandbytes
```

If Kaggle cannot reach PyPI for bitsandbytes, use this fallback:

```bash
pip install -q transformers datasets peft accelerate
```

---

## 2. Verify GPU

```bash
nvidia-smi
```

---

## 3. Load Dataset

```python
from datasets import load_dataset

DATA_FILES = {
    "train": "/kaggle/input/your-dataset/train.jsonl",
    "validation": "/kaggle/input/your-dataset/val.jsonl",
}

dataset = load_dataset("json", data_files=DATA_FILES)
print(dataset)
```

Replace the path with the dataset location in your Kaggle notebook.

---

## 4. QLoRA Fine-Tuning (Mistral 7B, no TRL)

This uses 4-bit loading to fit in Kaggle T4 GPUs.

```python
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# If you uploaded the model to Kaggle as a dataset, use the local path instead:
# MODEL_NAME = "/kaggle/input/models/mistral-ai/mistral/pytorch/7b-instruct-v0.1-hf/1"
OUTPUT_DIR = "mistral7b-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    learning_rate=2e-4,
    warmup_steps=50,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

---

## 4b. Fallback (No bitsandbytes, slower)

Use this if bitsandbytes cannot be installed. It runs full precision and will be slower.

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# If you uploaded the model to Kaggle as a dataset, use the local path instead:
# MODEL_NAME = "/kaggle/input/models/mistral-ai/mistral/pytorch/7b-instruct-v0.1-hf/1"
OUTPUT_DIR = "mistral7b-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    learning_rate=2e-4,
    warmup_steps=50,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

---

## 5. Quick In-Notebook Check

```python
sample = dataset["validation"][0]["text"]
prompt = sample.split("Answer:\n", 1)[0] + "Answer:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 6. Export Adapter

```python
import os
import shutil

print(os.listdir("."))          # should show "mistral7b-lora"
shutil.make_archive("mistral7b-lora", "zip", "mistral7b-lora")
```

In Kaggle, the adapter will be saved in the notebook output folder. Download the folder
`mistral7b-lora` and copy it into your project (for example `models/mistral7b-lora`).

You can then load it locally using PEFT if you want to run it without Ollama.

---