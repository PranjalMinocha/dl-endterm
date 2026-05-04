import os
import pandas as pd
from tqdm.auto import tqdm
import re

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from utils import ScienceQADataset, CHOICE_LETTERS, parse_choices_column

# Basic Settings
IMG_SIZE = 336

LORA_DIR = "outputs/checkpoint-2723"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load CSVs
test_csv = "test.csv" if os.path.exists("test.csv") else os.path.join("given", "test.csv")
submission_csv = (
    "sample_submission.csv"
    if os.path.exists("sample_submission.csv")
    else os.path.join("given", "sample_submission.csv")
)
test_df = pd.read_csv(test_csv)
submission_df = pd.read_csv(submission_csv, index_col="id")

# Parse JSON strings
test_df = parse_choices_column(test_df)

print(f"Test: {len(test_df)}, Submission: {len(submission_df)}")

test_ds = ScienceQADataset(test_df, img_size=IMG_SIZE)

print(f"Test dataset created: {len(test_ds)} rows")

# Define hyperparameters
BATCH_SIZE = 64 
NUM_WORKERS = 4 

def custom_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "image": [item["image"] for item in batch], 
        "text": [item["text"] for item in batch],
        "answer": [item["answer"] for item in batch]
    }

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate_fn
)

# Model
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Load Processor directly from Hugging Face (safest method for inference)
processor = AutoProcessor.from_pretrained(MODEL_ID)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# Left-padding is required for batched generation
processor.tokenizer.padding_side = "left"

print(f'EOS token: {processor.tokenizer.eos_token}, Pad token: {processor.tokenizer.pad_token}')

# Load Base Model
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=dtype, # FIX 2: Corrected from dtype=dtype
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
)

# Load and Merge LoRA Adapters
if os.path.exists(LORA_DIR):
    print(f"\nFound LoRA adapters at: {LORA_DIR}")
    model = PeftModel.from_pretrained(model, LORA_DIR)

    print("Merging LoRA weights into base model for fast inference...")
    model = model.merge_and_unload()
else:
    print(f"\nERROR: Could not find {LORA_DIR}. Check your folder path!")

if not torch.cuda.is_available():
    model.to(device)

model.eval()

cnt_regex_failures = 0
debug_data = []

# Inference Loop
for batch in tqdm(test_loader, desc="Running Inference"):
    # Prepare inputs
    inputs = processor(
        text=batch["text"],
        images=batch["image"],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.15
        )

    # Decode only the newly generated tokens
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    decoded_outputs = processor.batch_decode(new_tokens, skip_special_tokens=True)

    # Parse and Store
    for i in range(len(decoded_outputs)):
        q_id = batch["id"][i]
        generated_text = decoded_outputs[i].strip()

        # Try to find the exact phrase it was trained on
        match = re.search(r"choice is ([A-J])", generated_text, re.IGNORECASE)

        if match:
            pred_letter = match.group(1).upper()
            pred_index = CHOICE_LETTERS.index(pred_letter)
        else:
            # Fallback: find the very last A-J letter in the ramble
            matches = re.findall(r"\b([A-J])\b", generated_text)
            if matches:
                pred_letter = matches[-1].upper()
                pred_index = CHOICE_LETTERS.index(pred_letter)
            else:
                # Absolute failure fallback
                cnt_regex_failures += 1
                pred_index = 0

        if q_id in submission_df.index:
            submission_df.loc[q_id, "answer"] = pred_index

        debug_data.append({
            "id": q_id,
            "predicted_index": pred_index,
            "raw_output": generated_text
        })

# Save the final file
submission_df.to_csv("submission.csv")
print("Inference complete. Saved to submission.csv. Total regex failures: ", cnt_regex_failures)

debug_df = pd.DataFrame(debug_data)
debug_df.to_csv("model_responses.csv", index=False)
print("Saved full raw model outputs to model_responses.csv for inspection.")