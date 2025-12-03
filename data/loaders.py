# -*- coding: utf-8 -*-
"""
loaders.py

데이터 타입(SOURCE: hf/local, FORMAT: xlsx/json/csv/...)
에 관계없이 항상 HF Dataset 객체
컬럼명은 "src" / "tgt" 로 통일해 반환한다.
"""

import os, random
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from typing  import Dict
from ._xlsx import build_hfds_from_xlsx

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# HF_LOCAL_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "hf"))


# --------------------------------------------------------
# HF DATASET LOADER
# --------------------------------------------------------
def _load_hf_dataset(cfg, logger=None):
    """
    HF dataset을 /root/data/에 캐싱하고,
    테스트 split을 Dataset 형태로 반환.

    CFG 필수:
      - HF_DATASET_NAME
      - HF_DATASET_CONFIG(optional)
      - HF_SPLIT
      - SRC_FIELD / TGT_FIELD
    """
    name = cfg.HF_DATASET_NAME.replace("/", "_")
    cfg_name = cfg.HF_DATASET_CONFIG or "default"
    split = cfg.HF_SPLIT

    local_dir = os.path.join(THIS_DIR, name, cfg_name, split)

    # --- 1) load from local cache -------------------------
    if os.path.isdir(local_dir):
        if logger:
            logger.info(f"[HF] 캐시에서 로드: {local_dir}")
        ds = load_from_disk(local_dir)

    # --- 2) download from HF Hub --------------------------
    else:
        if logger:
            logger.info(f"[HF] 다운로드 중: {cfg.HF_DATASET_NAME}, config={cfg.HF_DATASET_CONFIG}, split={split}")

        if cfg.HF_DATASET_CONFIG:
            ds = load_dataset(cfg.HF_DATASET_NAME, cfg.HF_DATASET_CONFIG, split=split)
        else:
            ds = load_dataset(cfg.HF_DATASET_NAME, split=split)

        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        ds.save_to_disk(local_dir)

        if logger:
            logger.info(f"[HF] 캐시 저장 완료: {local_dir}")

    # --- 3) 컬럼명 통일 (src/tgt) --------------------------
    rename_map = {}
    if cfg.SRC_FIELD != "src":
        rename_map[cfg.SRC_FIELD] = "src"
    if cfg.TGT_FIELD != "tgt":
        rename_map[cfg.TGT_FIELD] = "tgt"
    if rename_map:
        ds = ds.rename_columns(rename_map)

    return ds


# --------------------------------------------------------
# LOCAL XLSX LOADER
# --------------------------------------------------------
def _load_local_xlsx(cfg):
    """
    엑셀을 파싱하여 DatasetDict(train/validation/test)를 만든 뒤,
    test split을 Dataset으로 반환한다.
    """
    ds_dict: DatasetDict = build_hfds_from_xlsx(cfg.RAW_DIR, cfg.DIRECTION)

    test_ds = ds_dict["test"]

    # # src/tgt 통일 (XLSX는 이미 src/tgt로 반환됨. 하지만 일반화 위해 유지) -> 에러 생겨서 일단 주석
    # rename_map = {}
    # if cfg.SRC_FIELD != "src":
    #     rename_map[cfg.SRC_FIELD] = "src"
    # if cfg.TGT_FIELD != "tgt":
    #     rename_map[cfg.TGT_FIELD] = "tgt"

    # if rename_map:
    #     test_ds = test_ds.rename_columns(rename_map)

    return test_ds


# --------------------------------------------------------
# LOCAL JSON LOADER
# --------------------------------------------------------
def _load_local_json(cfg):
    """
    JSON 파일을 list[dict] 형태로 읽은 뒤 Dataset으로 변환.
    """
    import json
    
    path = cfg.RAW_DIR
    if os.path.isdir(path):
        # 디렉토리 내에 존재하는 모든 *.json 파일 이름 저장
        files = [f for f in os.listdir(path) if f.endswith(".json")]
        if not files:
            raise FileNotFoundError(f"No .json file found in {path}")
    
    idx_list = []
    src_list = []
    tgt_list = []

    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            idx_list.append(item[cfg.IDX_FIELD] if cfg.IDX_FIELD in item else len(idx_list))
            src_list.append(item[cfg.SRC_FIELD])
            tgt_list.append(item[cfg.TGT_FIELD])

    return Dataset.from_dict({"idx": idx_list, "src": src_list, "tgt": tgt_list})


# --------------------------------------------------------
# LOCAL JSONL LOADER
# --------------------------------------------------------
def _load_local_jsonl(cfg):
    """
    JSONL 파일을 줄 단위로 읽어 list[dict] 로 변환한 뒤 Dataset으로 변환.
    """
    import json

    path = cfg.RAW_DIR
    if os.path.isdir(path):
        # 디렉토리 내에 존재하는 모든 *.jsonl 파일 이름 저장
        files = [f for f in os.listdir(path) if f.endswith(".jsonl")]
        if not files:
            raise FileNotFoundError(f"No .jsonl file found in {path}")

    idx_list = []
    src_list = []
    tgt_list = []

    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 빈 줄 건너뛰기
                
                obj = json.loads(line)  # 한 줄 = 하나의 JSON 객체

                idx_list.append(obj[cfg.IDX_FIELD] if cfg.IDX_FIELD in obj else len(idx_list))
                src_list.append(obj[cfg.SRC_FIELD])
                tgt_list.append(obj[cfg.TGT_FIELD])

    return Dataset.from_dict({"idx": idx_list, "src": src_list, "tgt": tgt_list})


# --------------------------------------------------------
# LOCAL CSV/TSV LOADER
# --------------------------------------------------------
def _load_local_csv_tsv(cfg):
    import pandas as pd

    if cfg.FORMAT == "csv":
        df = pd.read_csv(cfg.RAW_DIR)
    else:
        df = pd.read_csv(cfg.RAW_DIR, sep="\t")

    src_list = df[cfg.SRC_FIELD].astype(str).tolist()
    tgt_list = df[cfg.TGT_FIELD].astype(str).tolist()

    return Dataset.from_dict({"src": src_list, "tgt": tgt_list})


# --------------------------------------------------------
# LOCAL TXT LOADER (parallel files)
# --------------------------------------------------------
def _load_local_txt(cfg):
    """
    RAW_DIR이 디렉토리인 경우:
      - RAW_DIR/train.src, RAW_DIR/train.tgt 등이 있을 수도 있으나
      - eval이라면 test.src/test.tgt만 필요

    RAW_DIR이 파일이면:
      - 한 줄마다 src\ttgt 형식이라고 가정(옵션)
    """
    path = cfg.RAW_DIR

    # 1) 파일 하나: src <tab> tgt 구조 가정
    if os.path.isfile(path):
        src, tgt = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "\t" not in line:
                    continue
                s, t = line.strip().split("\t", 1)
                src.append(s)
                tgt.append(t)
        return Dataset.from_dict({"src": src, "tgt": tgt})

    # 2) 디렉토리: test.src/test.tgt
    src_file = os.path.join(path, "test.src")
    tgt_file = os.path.join(path, "test.tgt")

    if os.path.isfile(src_file) and os.path.isfile(tgt_file):
        src = [l.rstrip("\n") for l in open(src_file, encoding="utf-8")]
        tgt = [l.rstrip("\n") for l in open(tgt_file, encoding="utf-8")]
        return Dataset.from_dict({"src": src, "tgt": tgt})

    raise FileNotFoundError(f"TXT loader는 {path}에서 test.src/test.tgt 또는 TSV 라인이 필요합니다.")


# --------------------------------------------------------
# MAIN DISPATCHER
# --------------------------------------------------------
def load_eval_dataset(cfg, logger=None):
    """
    Always returns datasets.Dataset
    with columns "src" and "tgt".
    """
    if cfg.IDX_FIELD is None:
        # raise Warning("CFG.IDX_FIELD가 존재하지 않습니다. 이거 없으면 시발 train하면 안됨.")
        print("[WARNING] CFG.IDX_FIELD가 존재하지 않습니다. 이거 없으면 시발 train하면 안됨.")

    if cfg.SOURCE == "hf":
        return _load_hf_dataset(cfg, logger)

    elif cfg.SOURCE == "local":
        # FORMAT에 따라 선택
        fmt = cfg.FORMAT.lower()

        if fmt == "xlsx":
            return _load_local_xlsx(cfg)

        elif fmt == "json":
            return _load_local_json(cfg)
        
        elif fmt == "jsonl":
            return _load_local_jsonl(cfg)

        elif fmt in ("csv", "tsv"):
            return _load_local_csv_tsv(cfg)

        elif fmt in ("txt", "text"):
            return _load_local_txt(cfg)

        elif fmt in ("parquet", "arrow"):
            # parquet/arrow는 pandas 또는 pyarrow로 처리 가능
            import pandas as pd
            df = pd.read_parquet(cfg.RAW_DIR)
            src_list = df[cfg.SRC_FIELD].astype(str).tolist()
            tgt_list = df[cfg.TGT_FIELD].astype(str).tolist()
            return Dataset.from_dict({"src": src_list, "tgt": tgt_list})

        else:
            raise RuntimeError(f"지원하지 않는 FORMAT: {cfg.FORMAT}")

    else:
        raise RuntimeError(f"지원하지 않는 SOURCE: {cfg.SOURCE}")

def load_train_val_test(cfg, logger=None) -> Dict[str, Dataset]:
    """
    cfg에 따라 HF/xlsx/json 등에서 병렬 말뭉치를 불러오고,
    train/validation을 나눠서 반환.
    - 이미 train/validation split이 있으면 그대로 반환
    - 없으면 train_test_split으로 쪼개기 등
    """
    ...