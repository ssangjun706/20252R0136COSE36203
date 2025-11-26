# src/configs/lemonmint.py
from ._config import CFG

LEMONMINT_EN2KO = CFG(
    SOURCE="hf",
    FORMAT="auto",
    SRC_FIELD="english",
    TGT_FIELD="korean",

    HF_SPLIT="train",
    HF_DATASET_NAME="lemon-mint/korean_english_parallel_wiki_augmented_v1",
    MODEL_NAME="facebook/nllb-200-distilled-600M",
)