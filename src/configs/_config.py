# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Literal

# 지원 언어 매핑
LANG_MAP = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "english": "eng_Latn",

    "ko": "kor_Hang",
    "kor": "kor_Hang",
    "korean": "kor_Hang",
}

@dataclass
class CFG:
    # -------------------------
    # 1) 데이터 소스 및 포맷
    # -------------------------
    SOURCE: Literal["hf", "local"] = "local"
    FORMAT: Literal[
        "auto", "xlsx", "json", "csv", "tsv",
        "txt", "parquet", "arrow"
    ] = "auto"

    # -------------------------
    # 2) 데이터 식별자
    # -------------------------
    RAW_DIR: Optional[str] = None # xlsx/json은 파일경로, 디렉토리, 어떤 형태든 문자열이면 됨
    HF_DATASET_NAME: Optional[str] = None # hf만 사용
    HF_DATASET_CONFIG: Optional[str] = None

    HF_SPLIT: str = "test"
    CACHE_DIR: Optional[str] = None  # 파서 출력(평행 파일)

    # -------------------------
    # 3) 컬럼 정보
    # -------------------------
    IDX_FIELD: Optional[str] = None  # 동일 맥락 문장 식별용
    SRC_FIELD: str = "src"
    TGT_FIELD: str = "tgt"
    SRC_LANG: Optional[str] = None
    TGT_LANG: Optional[str] = None
    
    # -------------------------
    # 4) 모델 정보
    # -------------------------
    # MODEL_NAME: str = "Helsinki-NLP/opus-mt-tc-big-en-ko"  # ko2en이면 opus-mt-ko-en
    MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
    MAX_SRC: int = 128
    MAX_TGT: int = 128
    GEN_BEAMS: int = 4
    DIRECTION: Optional[str] = None  # ex) en2ko

    # -------------------------
    # 5) 학습/평가
    # -------------------------
    MAX_GROUPS: Optional[int] = None
    OUTPUT_DIR: Optional[str] = None
    SEED: int = 1337
    SENTENCE_LEVEL: bool = True        # 샘플을 쪼갠 후, 문장 단위 번역 여부 (실제 측정 시 반드시 켜야함)
    EVAL_MAX_SAMPLES: Optional[int] = 2000  # 랜덤 샘플링 개수 (None이면 전체 사용)

    # 학습 파라미터
    LR: float = 2e-5
    WD: float = 0.01
    EPOCHS: int = 2
    TRAIN_BS: int = 32
    EVAL_BS: int = 32
    FP16: bool = False
    BF16: bool = True
    LOG_STEPS: int = 200
    SAVE_STEPS: int = 2000
    EVAL_STEPS: int = 2000

    # -------------------------
    # 6) POST INIT
    # -------------------------

    def __post_init__(self):
        # SOURCE 검증
        if self.SOURCE == "hf":
            if not self.HF_DATASET_NAME:
                raise ValueError("HF 데이터셋 사용 시 HF_DATASET_NAME이 필요합니다.")
        elif self.SOURCE == "local":
            if not self.RAW_DIR:
                raise ValueError("LOCAL 데이터셋 사용 시 RAW_DIR이 필요합니다.")
        else:
            raise ValueError(f"지원하지 않는 SOURCE: {self.SOURCE}")
        
        # FORMAT 자동 판별
        if self.FORMAT == "auto" and self.SOURCE == "local" and self.RAW_DIR:
            if "." in self.RAW_DIR:
                ext = self.RAW_DIR.split(".")[-1].lower()
                self.FORMAT = {
                    "xlsx": "xlsx",
                    "json": "json",
                    "jsonl": "json",
                    "csv": "csv",
                    "tsv": "tsv",
                    "txt": "txt",
                    "parquet": "parquet",
                    "arrow": "arrow"
                }.get(ext, "txt")  # default text
            else:
                # RAW_DIR이 폴더라면 → XLSX 파일 묶음으로 판단
                self.FORMAT = "xlsx"

        # 언어 코드 자동 설정
        if self.SRC_LANG is None:
            key = self.SRC_FIELD.lower()
            if key not in LANG_MAP:
                raise ValueError(f"Unsupported SRC_FIELD '{self.SRC_FIELD}'")
            self.SRC_LANG = LANG_MAP[key]

        if self.TGT_LANG is None:
            key = self.TGT_FIELD.lower()
            if key not in LANG_MAP:
                raise ValueError(f"Unsupported TGT_FIELD '{self.TGT_FIELD}'")
            self.TGT_LANG = LANG_MAP[key]

        # OUTPUT_DIR 및 DIRECTION 자동 설정
        if self.DIRECTION is None:
            self.DIRECTION = f"{self.SRC_FIELD}2{self.TGT_FIELD}"

        if self.OUTPUT_DIR is None:
            if self.SOURCE == "hf":
                if not self.HF_DATASET_NAME:
                    raise ValueError("HF_DATASET_NAME required for DATA_TYPE='hf'")
                dataset = self.HF_DATASET_NAME.replace("/", "_")

            else:  # local
                if not self.RAW_DIR:
                    raise ValueError("RAW_DIR required for DATA_TYPE='xlsx' or 'jsonl'")
                dataset = self.RAW_DIR.replace("/", "_").replace("\\", "_")

            self.OUTPUT_DIR = f"out/{dataset}/{self.DIRECTION}"