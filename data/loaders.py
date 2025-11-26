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
from .xlsx import build_hfds_from_xlsx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HF_LOCAL_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "hf_cache"))


# --------------------------------------------------------
# HF DATASET LOADER
# --------------------------------------------------------
def _load_hf_dataset(cfg, logger=None):
    """
    HF dataset을 /root/data/hf 에 캐시하고,
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

    local_dir = os.path.join(HF_LOCAL_ROOT, name, cfg_name, split)

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

    with open(cfg.RAW_DIR, "r", encoding="utf-8") as f:
        raw = json.load(f)

    src_list = [item[cfg.SRC_FIELD] for item in raw]
    tgt_list = [item[cfg.TGT_FIELD] for item in raw]

    return Dataset.from_dict({"src": src_list, "tgt": tgt_list})


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

    if cfg.SOURCE == "hf":
        return _load_hf_dataset(cfg, logger)

    elif cfg.SOURCE == "local":
        # FORMAT에 따라 선택
        fmt = cfg.FORMAT.lower()

        if fmt == "xlsx":
            return _load_local_xlsx(cfg)

        elif fmt == "json":
            return _load_local_json(cfg)

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
