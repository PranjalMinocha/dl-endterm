import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import re

from utils import ScienceQADataset, build_prompt, CHOICE_LETTERS, parse_choices_column


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMG_SIZE = 336

TRAIN_CSV = os.path.join("given", "train.csv")
VAL_CSV = os.path.join("given", "val.csv")
OUTPUT_DIR = "outputs"

SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScienceQATrainDataset(ScienceQADataset):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img = self._load_image("images/" + row["image_path"])
        prompt = build_prompt(row, include_answer=False)
        label_text = f" {CHOICE_LETTERS[int(row['answer'])]}"
        return {
            "id": row["id"],
            "image": img,
            "prompt": prompt,
            "label_text": label_text,
        }


@dataclass
class DataCollatorForSmolVLM:
    processor: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [f["image"] for f in features]
        # Combine prompt and label into one string for the causal decoder
        full_texts = [f["prompt"] + f["label_text"] for f in features]
        prompt_texts = [f["prompt"] for f in features]

        # Tokenize the full sequence (Prompt + Answer)
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Tokenize the prompt alone to find where the "Answer" starts
        prompt_encodings = self.processor.tokenizer(
            prompt_texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # Create labels by copying input_ids
        labels = batch["input_ids"].clone()

        # Mask the prompt tokens: Find the length of each prompt and set labels to -100
        for i in range(len(features)):
            # Find how many tokens were padding in the prompt (if left padded)
            prompt_len = prompt_encodings["attention_mask"][i].sum().item()

            # Mask the prompt part (0 to prompt_len)
            labels[i, :prompt_len] = self.label_pad_token_id

        # Mask the padding tokens in the labels
        labels[labels == self.processor.tokenizer.pad_token_id] = self.label_pad_token_id

        batch["labels"] = labels
        return batch


def parse_pred_letter(text: str) -> Optional[str]:
    # This specifically looks for the A-J letter right after "Answer:"
    match = re.search(r"Answer:\s*([A-J])", text)
    if match:
        return match.group(1)
    return None


def preprocess_logits_for_metrics(logits, labels):
    """
    Shrinks the massive logits tensor into just the predicted token IDs
    batch-by-batch so the GPU memory doesn't explode during validation.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    # Do the argmax here, before the batches are gathered!
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    pred_ids, labels = eval_pred
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    label_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    correct = 0
    total = 0
    for pred_text, label_text in zip(pred_texts, label_texts):
        pred_letter = parse_pred_letter(pred_text.strip())
        label_letter = parse_pred_letter(label_text.strip())
        if pred_letter is None or label_letter is None:
            total += 1
            continue
        if pred_letter == label_letter:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc}


if __name__ == "__main__":
    set_seed(SEED)

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_df = parse_choices_column(train_df)
    val_df = parse_choices_column(val_df)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
        use_cache=False,
    )
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    train_ds = ScienceQATrainDataset(train_df, img_size=IMG_SIZE, is_train=True)
    val_ds = ScienceQATrainDataset(val_df, img_size=IMG_SIZE, is_train=False)

    data_collator = DataCollatorForSmolVLM(processor=processor)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=50,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        learning_rate=2e-4,
        num_train_epochs=10,
        warmup_steps=0.03,
        bf16=True,
        tf32=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Starting training...")
    trainer.train(ignore_keys_for_eval=["past_key_values"])

    print("Saving LoRA adapters...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "processor"))

    print("Done.")
