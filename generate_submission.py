import os
import pandas as pd
from tqdm.auto import tqdm
import re

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

from utils import ScienceQADataset, CHOICE_LETTERS, parse_choices_column

# Basic Settings
IMG_SIZE = 336

LORA_DIR = os.path.join("outputs", "lora")
PROCESSOR_DIR = os.path.join("outputs", "processor")

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

# The 'choices' column is a JSON string, so we parse it
test_df = parse_choices_column(test_df)

print(f"Test: {len(test_df)}, Submission: {len(submission_df)}")

test_ds = ScienceQADataset(test_df, img_size=IMG_SIZE, is_train=False)

print(f"Test dataset created: {len(test_ds)} rows")

# Define hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 4  # Adjust based on your CPU cores

def custom_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "image": [item["image"] for item in batch], # Keeps these as a list of PIL Images
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

# Load Model and Processor
processor_path = PROCESSOR_DIR if os.path.exists(PROCESSOR_DIR) else MODEL_ID
processor = AutoProcessor.from_pretrained(processor_path)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

processor.tokenizer.padding_side = "left"

print(f'EOS token: {processor.tokenizer.eos_token}, Pad token: {processor.tokenizer.pad_token}')

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
)
if os.path.exists(LORA_DIR):
    model = PeftModel.from_pretrained(model, LORA_DIR)
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
            max_new_tokens=50,
            do_sample=False,
        )

    # Decode 
    decoded_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Parse and Store in submission_df
    for i in range(len(decoded_outputs)):
        q_id = batch["id"][i]
        full_text = decoded_outputs[i]
        
        # Parse the output: Look for the first letter A-J after "Answer:"
        match = re.search(r"Answer:\s*([A-J])", full_text)
        
        if match:
            pred_letter = match.group(1)
            pred_index = CHOICE_LETTERS.index(pred_letter)
        else:
            # Fallback in case the model outputs something unexpected
            cnt_regex_failures += 1
            pred_index = 0 
            
        # Update the specific row in our submission dataframe
        if q_id in submission_df.index:
            submission_df.loc[q_id, "answer"] = pred_index
        
        debug_data.append({
            "id": q_id,
            "predicted_index": pred_index,
            "raw_output": full_text # Saving just the generated part makes it easier to read!
        })

# Save the final file
submission_df.to_csv("submission.csv")
print("Inference complete. Saved to submission.csv. Total regex failures: ", cnt_regex_failures)

debug_df = pd.DataFrame(debug_data)
debug_df.to_csv("model_responses.csv", index=False)
print("Saved full raw model outputs to model_responses.csv for inspection.")
