# src/configs/wiki.py
"""
lemonmint_wiki 데이터를 문장 단위로 가공한 데이터 셋
"""
from ._config import CFG

WIKI_EN2KO = CFG(
    SOURCE="local",
    FORMAT="jsonl",
    
    IDX_FIELD="idx",
    SRC_FIELD="english",
    TGT_FIELD="korean",

    RAW_DIR="cose362/data/wiki_augmented_v1",
    MODEL_NAME="facebook/nllb-200-distilled-600M",
)