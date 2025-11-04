# -*- coding: utf-8 -*-
import os, re, random
from typing import List, Tuple, Dict
import pandas as pd
from datasets import Dataset, DatasetDict
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer

mpn_en = MosesPunctNormalizer(lang="en")
mtok_en = MosesTokenizer(lang="en")
# detok은 BLEU용 복원 때만 필요. 한국어 detok은 굳이 안 써도 됨.

def moses_tokenize_en(s: str) -> str:
    s = mpn_en.normalize(s)
    return " ".join(mtok_en.tokenize(s, return_str=False))

HANGUL_RE = re.compile(r"[가-힣]")

def _detect_lang(text: str) -> str:
    return "ko" if text and HANGUL_RE.search(str(text)) else "en"

def _find_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    # 명시 헤더 우선
    if "원문" in cols and "번역문" in cols: return "원문", "번역문"
    for s, t in [("source","target"),("src","tgt"),("en","ko"),("english","korean"),("Origin","Translation")]:
        if s in cols and t in cols: return s, t
    # 휴리스틱
    text_cols = [c for c in cols if df[c].dtype == "object"]
    if len(text_cols) < 2: raise ValueError(f"텍스트 컬럼 부족: {cols}")
    scores = {}
    for c in text_cols:
        ser = df[c].dropna().astype(str).head(200)
        scores[c] = ser.apply(lambda x: 1 if HANGUL_RE.search(x) else 0).mean()
    ko_col = max(scores, key=scores.get)
    en_col = min(scores, key=scores.get)
    return ko_col, en_col

def load_xlsx_pairs(root: str, direction: str="en2ko") -> List[Tuple[str,str]]:
    pairs: List[Tuple[str,str]] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".xlsx"): continue
            fp = os.path.join(dirpath, fn)
            try:
                xls = pd.ExcelFile(fp, engine="openpyxl")
            except Exception as e:
                print(f"[skip] {fp}: {e}"); continue
            for sh in xls.sheet_names:
                try:
                    df = xls.parse(sheet_name=sh, dtype=str)
                except Exception as e:
                    print(f"[skip] {fp}:{sh}: {e}"); continue
                if df.empty: continue
                try:
                    ko_col, en_col = _find_cols(df)
                except Exception as e:
                    print(f"[warn] col detect fail {fp}:{sh}: {e}"); continue
                df = df[[ko_col, en_col]].rename(columns={ko_col:"ko", en_col:"en"}).dropna()
                df["ko"] = df["ko"].astype(str).str.strip()
                df["en"] = df["en"].astype(str).str.strip()
                df = df[(df["ko"]!="") & (df["en"]!="")]
                if direction == "en2ko":
                    cur = list(df[["en","ko"]].itertuples(index=False, name=None))
                else:
                    cur = list(df[["ko","en"]].itertuples(index=False, name=None))
                pairs.extend(cur)
    # 중복 제거
    pairs = list(dict.fromkeys(pairs))
    return pairs

def split_pairs(pairs: List[Tuple[str,str]], seed=1337, train_ratio=0.98, val_ratio=0.01):
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = pairs[:n_train]
    val = pairs[n_train:n_train+n_val]
    test = pairs[n_train+n_val:]
    return train, val, test

def save_parallel(out_dir: str, name: str, data: List[Tuple[str,str]]):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.src"), "w", encoding="utf-8") as fs, \
         open(os.path.join(out_dir, f"{name}.tgt"), "w", encoding="utf-8") as ft:
        for s,t in data:
            fs.write(s.replace("\n"," ").strip()+"\n")
            ft.write(t.replace("\n"," ").strip()+"\n")

def build_hfds_from_xlsx(root: str, direction: str, cache_dir: str=None) -> DatasetDict:
    pairs = load_xlsx_pairs(root, direction)
    if len(pairs) < 10: raise RuntimeError(f"pair 부족: {len(pairs)}")
    train, val, test = split_pairs(pairs)
    if cache_dir:
        save_parallel(cache_dir, "train", train)
        save_parallel(cache_dir, "val", val)
        save_parallel(cache_dir, "test", test)
    def to_ds(split): return Dataset.from_dict({"src":[s for s,_ in split], "tgt":[t for _,t in split]})
    return DatasetDict({"train":to_ds(train), "validation":to_ds(val), "test":to_ds(test)})
