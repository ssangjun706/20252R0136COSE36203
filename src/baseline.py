# -*- coding: utf-8 -*-
from transformers import MarianTokenizer, MarianMTModel, DataCollatorForSeq2Seq
from datasets import DatasetDict
from typing import Tuple
from .config import CFG
from .utils import build_hfds_from_xlsx

def build_data(cfg: CFG) -> DatasetDict:
    return build_hfds_from_xlsx(cfg.RAW_DIR, cfg.DIRECTION, cache_dir=cfg.CACHE_DIR)

def build_tok_model(cfg: CFG) -> Tuple[MarianTokenizer, MarianMTModel]:
    tok = MarianTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = MarianMTModel.from_pretrained(cfg.MODEL_NAME)
    return tok, model

def preprocess_builder(tok, cfg: CFG):
    max_src, max_tgt = cfg.MAX_SRC, cfg.MAX_TGT
    def _pp(batch):
        model_inputs = tok(batch["src"], max_length=max_src, truncation=True)
        with tok.as_target_tokenizer():
            labels = tok(batch["tgt"], max_length=max_tgt, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _pp

def build_collator(tok, model):
    return DataCollatorForSeq2Seq(tok, model=model)
