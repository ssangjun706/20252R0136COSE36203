# src/configs/lemonmint_koen.py
from ._config import CFG

LEMONMINT_EN2KO = CFG(
    SOURCE="hf",
    FORMAT="auto",
    SRC_FIELD="en",
    TGT_FIELD="ko",

    HF_SPLIT="test",
    HF_DATASET_NAME="lemon-mint/korean-english-parallel-datasets",
    MODEL_NAME="facebook/nllb-200-distilled-600M",
)