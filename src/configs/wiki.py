# src/configs/wiki.py
from ._config import CFG

WIKI_EN2KO = CFG(
    SOURCE="local",
    FORMAT="jsonl",
    SRC_FIELD="english",
    TGT_FIELD="korean",

    RAW_DIR="cose362/data/wiki_augmented_v1",
    MODEL_NAME="facebook/nllb-200-distilled-600M",
)