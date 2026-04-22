import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq

# Basic Settings
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load CSVs
test_df  = pd.read_csv("test.csv")
submission_df = pd.read_csv("sample_submission.csv", index_col="id")

# The 'choices' column is a JSON string, so we parse it
test_df["choices"] = test_df["choices"].apply(json.loads)

print(f"Test: {len(test_df)}, Submission: {len(submission_df)}")

# Prompt Engineering
CHOICE_LETTERS = "ABCDEFGHIJ"

def build_prompt(row: pd.Series, include_answer: bool = False) -> str:
    """
    Builds the text prompt for the Vision Language Model.
    The <image> token is required for the model to process the image.
    """
    context_parts = []
    lecture = row.get("lecture", "")
    hint    = row.get("hint", "")
    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)

    choices = row["choices"]
    choices_str = "\n".join(
        f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)
    )

    prompt = "<image>\n"
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    prompt += "Answer:"

    if include_answer:
        answer_idx = int(row['answer'])
        prompt += f" {CHOICE_LETTERS[answer_idx]}"

    return prompt


# PyTorch Dataset
class ScienceQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 224, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return img

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])

        if self.is_train:
            return {
                "id": row["id"],
                "image": img,
                "text": build_prompt(row, include_answer=True),
                "answer": int(row["answer"]),
            }
        else:
            return {
                "id": row["id"],
                "image": img,
                "text": build_prompt(row, include_answer=False),
                "answer": int(row["answer"]) if "answer" in row else -1,
            }

test_ds  = ScienceQADataset(test_df, img_size=IMG_SIZE, is_train=False)

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
processor = AutoProcessor.from_pretrained(MODEL_ID)
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
if not torch.cuda.is_available():
    model.to(device)
model.eval()

cnt_regex_failures = 0

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

# Save the final file
submission_df.to_csv("submission.csv")
print("Inference complete. Saved to submission.csv. Total regex failures: ", cnt_regex_failures)