import json
from typing import List, Dict, Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


CHOICE_LETTERS = "ABCDEFGHIJ"


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
    prompt += "Solution:\n"

    return prompt


def parse_choices_column(df: pd.DataFrame) -> pd.DataFrame:
    df["choices"] = df["choices"].apply(json.loads)
    return df


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
        img = self._load_image("images/" + row["image_path"])

        return {
            "id": row["id"],
            "image": img,
            "text": build_prompt(row),
            "answer": int(row["answer"]) if "answer" in row else -1,
        }
