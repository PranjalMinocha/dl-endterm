import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMG_SIZE = 336
CHOICE_LETTERS = "ABCDEFGHIJ"

TRAIN_CSV = os.path.join("given", "train.csv")
VAL_CSV = os.path.join("given", "val.csv")
OUTPUT_DIR = "outputs"

SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(row: pd.Series, include_answer: bool = False) -> str:
    context_parts: List[str] = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
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
        answer_idx = int(row["answer"])
        prompt += f" {CHOICE_LETTERS[answer_idx]}"

    return prompt


class ScienceQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])
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
        prompts = [f["prompt"] for f in features]
        label_texts = [f["label_text"] for f in features]

        prompt_inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels = self.processor.tokenizer(
                label_texts,
                padding=True,
                return_tensors="pt",
            )["input_ids"]

        labels = labels.masked_fill(
            labels == self.processor.tokenizer.pad_token_id,
            self.label_pad_token_id,
        )

        prompt_inputs["labels"] = labels
        return prompt_inputs


def parse_pred_letter(text: str) -> Optional[str]:
    for ch in text:
        if ch in CHOICE_LETTERS:
            return ch
    return None


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    pred_ids = np.argmax(preds, axis=-1)
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

    train_df["choices"] = train_df["choices"].apply(json.loads)
    val_df["choices"] = val_df["choices"].apply(json.loads)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)

    train_ds = ScienceQADataset(train_df, img_size=IMG_SIZE, is_train=True)
    val_ds = ScienceQADataset(val_df, img_size=IMG_SIZE, is_train=False)

    data_collator = DataCollatorForSmolVLM(processor=processor)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=4,
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
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
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapters...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "processor"))

    print("Done.")
