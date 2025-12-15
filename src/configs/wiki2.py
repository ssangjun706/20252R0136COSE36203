# src/configs/wiki2.py
"""
wiki2.py
  - MAX_GROUPS=20000 으로 늘림.
"""
from ._config import CFG

WIKI_EN2KO = CFG(
    SOURCE="local",
    FORMAT="jsonl",
    
    IDX_FIELD="idx",
    SRC_FIELD="english",
    TGT_FIELD="korean",

    MAX_GROUPS=20000,

    RAW_DIR="cose362/data/wiki_augmented_v1",
    MODEL_NAME="facebook/nllb-200-distilled-600M",
    SRC_LANG   = "eng_Latn",
    TGT_LANG   = "kor_Hang",
    RUN_NAME="lr5e4_bs32_e10_g20k",
    
    # 학습 파라미터
    LR=5e-4,            # learning rate
    WD=0.01,            # weight decay
    EPOCHS=10,           
    TRAIN_BS=32,        # 한 step에 들어가는 샘플 수 (gpu 성능 따라 정할것.)
    EVAL_BS=32,         # 평가 시 사용하는 batch size
    LOG_STEPS=100,      # 몇 step 마다 loss/lr 등을 로그로 찍을지
    SAVE_STEPS=1000,    # 몇 step 마다 checkpoint를 저장할지
    EVAL_STEPS=1000,    # 몇 step 마다 validation을 돌릴지
)