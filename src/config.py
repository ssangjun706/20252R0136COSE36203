# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class CFG:
    # 데이터
    RAW_DIR: str = "cose362/data/aihub_news"        # .xlsx가 있는 루트
    CACHE_DIR: str = "data/aihub_news_parsed"  # 파서 출력(평행 파일)
    DIRECTION: str = "en2ko"                 # or "ko2en"

    # 모델
    # MODEL_NAME: str = "Helsinki-NLP/opus-mt-tc-big-en-ko"  # ko2en이면 opus-mt-ko-en
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    MAX_SRC: int = 128
    MAX_TGT: int = 128

    # 언어 코드
    # NLLB: eng_Latn / kor_Hang
    SRC_LANG: str = "eng_Latn"
    TGT_LANG: str = "kor_Hang"

    GEN_BEAMS: int = 4
    MAX_SRC: int = 128
    MAX_TGT: int = 128

    # 학습
    OUTPUT_DIR: str = "out/en2ko_baseline"
    LR: float = 2e-5
    WD: float = 0.01
    EPOCHS: int = 2
    TRAIN_BS: int = 32
    EVAL_BS: int = 32
    FP16: bool = False
    BF16: bool = True
    GEN_BEAMS: int = 4
    LOG_STEPS: int = 200
    SAVE_STEPS: int = 2000
    EVAL_STEPS: int = 2000
    SEED: int = 1337
