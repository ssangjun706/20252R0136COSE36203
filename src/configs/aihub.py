# src/configs/aihub.py
from ._config import CFG
# from ...data import aihub_news

AIHUB_EN2KO = CFG(
    SOURCE="local",
    FORMAT="xlsx",
    SRC_FIELD="en",
    TGT_FIELD="ko",

    RAW_DIR="cose362/data/aihub_news",
    MODEL_NAME="facebook/nllb-200-distilled-600M",
    RUN_NAME="default",
)