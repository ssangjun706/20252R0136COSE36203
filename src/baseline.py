# -*- coding: utf-8 -*-
"""
src/baseline.py

백본 모델
- HuggingFace seq2seq 백본 모델/토크나이저 빌더
- 모델 입력(feature) 전처리 함수 빌더   # cf) loaders.py에서는 데이터 형식 통일 전처리 수행. (로더는 모델과 독립적.)
- DataCollator 빌더
"""

from typing import Tuple, Callable, Dict, Any, List, Optional, Sequence
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from .configs._config import CFG


def build_backbone(cfg: CFG):
    """
    기본 HF seq2seq 백본 모델과 토크나이저를 생성.
    - MT 모델 (Marian, NLLB, M2M 등 상관 없음)
    """
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    # 언어 코드 지정 (NLLB/M2M 등)
    if hasattr(tok, "src_lang") and cfg.SRC_LANG:
        tok.src_lang = cfg.SRC_LANG
    if hasattr(tok, "tgt_lang") and getattr(cfg, "TGT_LANG", None):
        tok.tgt_lang = cfg.TGT_LANG

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL_NAME)
    return tok, model


def build_preprocess_fn(tok, cfg: CFG) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Dataset의 {"src", "tgt"} → 모델 입력/레이블 토크나이즈를 수행하는 함수 생성.
    Trainer의 .map()에 넘겨서 사용.
    """
    max_src, max_tgt = cfg.MAX_SRC, cfg.MAX_TGT

    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        # 입력 (encoder)
        model_inputs = tok(
            examples["src"],
            max_length=max_src,
            truncation=True,
        )

        # 타깃 (decoder labels)
        labels = tok(
            text_target=examples["tgt"],
            max_length=max_tgt,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


class PrevSentenceLookup:
    """
    (idx, sent_idx) → 해당 샘플의 토크나이즈 결과를 찾아주는 간단한 룩업 테이블.
    이전 문장을 효율적으로 조회하기 위해 사용한다.
    """

    def __init__(
        self,
        datasets: Optional[Sequence] = None,
        idx_field: str = "idx",
        sent_idx_field: str = "sent_idx",
    ):
        self.idx_field = idx_field
        self.sent_idx_field = sent_idx_field
        self._datasets: List = []
        self._index: Dict[Tuple[int, int], Tuple[int, int]] = {}

        datasets = datasets or []
        for ds in datasets:
            self._register_dataset(ds)

    def _register_dataset(self, dataset):
        if dataset is None:
            return
        if self.idx_field not in dataset.column_names:
            return

        idx_column = dataset[self.idx_field]
        sent_idx_column = (
            dataset[self.sent_idx_field]
            if self.sent_idx_field in dataset.column_names
            else None
        )

        dataset_id = len(self._datasets)
        self._datasets.append(dataset)

        for row_idx, idx_val in enumerate(idx_column):
            sent_val = (
                sent_idx_column[row_idx]
                if sent_idx_column is not None
                else row_idx
            )
            key = (int(idx_val), int(sent_val))
            self._index[key] = (dataset_id, row_idx)

    def fetch_prev(self, idx_val, sent_idx_val):
        if idx_val is None or sent_idx_val is None:
            return None
        sent_idx_val = int(sent_idx_val)
        if sent_idx_val <= 0:
            return None

        key = (int(idx_val), sent_idx_val - 1)
        pointer = self._index.get(key)
        if pointer is None:
            return None

        ds_id, row_idx = pointer
        dataset = self._datasets[ds_id]
        row = dataset[row_idx]
        return row.get("input_ids"), row.get("attention_mask")


class ContextDataCollator:
    """
    기본 Seq2Seq data collator + 이전 문장 입력을 구성하는 래퍼.
    """

    def __init__(
        self,
        tokenizer,
        model=None,
        prev_lookup: Optional[PrevSentenceLookup] = None,
        idx_field: str = "idx",
        sent_idx_field: str = "sent_idx",
    ):
        self.tokenizer = tokenizer
        self.idx_field = idx_field
        self.sent_idx_field = sent_idx_field
        self.prev_lookup = prev_lookup
        self.base_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if self.pad_token_id is None:
            self.pad_token_id = 0

        self._meta_keys = {
            self.idx_field,
            self.sent_idx_field,
            "prev_input_ids",
            "prev_attention_mask",
        }
        # DataCollatorForSeq2Seq의 자동 decoder_input_ids 생성을 방지하기 위해
        # decoder_inputs_embeds를 추가로 필터링
        self._collator_keys_to_remove = {
            "idx",
            "sent_idx", 
            "prev_input_ids",
            "prev_attention_mask",
            "decoder_inputs_embeds",  # 자동 생성되는 경우 제거
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prev_inputs: List[Optional[List[int]]] = []
        prev_masks: List[Optional[List[int]]] = []
        filtered_features: List[Dict[str, Any]] = []

        for feat in features:
            idx_val = feat.get(self.idx_field)
            sent_idx_val = feat.get(self.sent_idx_field)
            provided_prev = feat.get("prev_input_ids")
            provided_prev_mask = feat.get("prev_attention_mask")

            prev_ids, prev_mask = self._resolve_prev_inputs(
                idx_val,
                sent_idx_val,
                provided_prev,
                provided_prev_mask,
            )
            prev_inputs.append(prev_ids)
            prev_masks.append(prev_mask)

            # 메타데이터 제거
            filtered = {k: v for k, v in feat.items() if k not in self._meta_keys}
            # decoder_inputs_embeds도 제거 (있을 경우)
            filtered.pop("decoder_inputs_embeds", None)
            filtered_features.append(filtered)

        # base collator에서 decoder_input_ids 자동 생성이 발생하면 제거
        batch = self.base_collator(filtered_features)
        
        # 혹시 base_collator에서 생성한 decoder_inputs_embeds 제거
        batch.pop("decoder_inputs_embeds", None)
        
        prev_batch = self._build_prev_batch(prev_inputs, prev_masks)
        batch.update(prev_batch)
        return batch

    def _resolve_prev_inputs(
        self,
        idx_val,
        sent_idx_val,
        provided_prev,
        provided_prev_mask,
    ):
        prev_ids = self._to_list(provided_prev)
        prev_mask = self._to_list(provided_prev_mask)

        if prev_ids is not None:
            if prev_mask is None:
                prev_mask = [1] * len(prev_ids)
            return prev_ids, prev_mask

        if self.prev_lookup is None:
            return None, None

        lookup = self.prev_lookup.fetch_prev(idx_val, sent_idx_val)
        if lookup is None:
            return None, None

        prev_ids, prev_mask = lookup
        prev_ids = self._to_list(prev_ids)
        prev_mask = self._to_list(prev_mask)
        if prev_ids is not None and prev_mask is None:
            prev_mask = [1] * len(prev_ids)

        return prev_ids, prev_mask

    def _build_prev_batch(self, prev_inputs, prev_masks):
        pad_token_id = self.pad_token_id
        valid_lengths = [len(seq) for seq in prev_inputs if seq]
        max_len = max(valid_lengths) if valid_lengths else 1

        padded_ids = []
        padded_masks = []

        for seq, mask in zip(prev_inputs, prev_masks):
            if not seq:
                seq = [pad_token_id]
                mask = [0]
            elif mask is None:
                mask = [1] * len(seq)

            pad_len = max_len - len(seq)
            if pad_len > 0:
                seq = seq + [pad_token_id] * pad_len
                mask = mask + [0] * pad_len

            padded_ids.append(seq)
            padded_masks.append(mask)

        return {
            "prev_input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "prev_attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }

    @staticmethod
    def _to_list(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except TypeError:
                pass
        if isinstance(value, (list, tuple)):
            return list(value)
        return [int(value)]


def build_collator(
    tok,
    model,
    datasets: Optional[Sequence] = None,
    cfg: Optional[CFG] = None,
):
    """
    Seq2Seq Trainer용 데이터 콜레이터.
    padding/label shifting 외에도, 이전 문장 입력 텐서를 구성해 모델에 전달한다.
    """
    idx_field = getattr(cfg, "IDX_FIELD", "idx") if cfg else "idx"
    sent_idx_field = (
        getattr(cfg, "SENT_IDX_FIELD", "sent_idx") if cfg else "sent_idx"
    )

    lookup = PrevSentenceLookup(
        datasets=datasets,
        idx_field=idx_field,
        sent_idx_field=sent_idx_field,
    )
    return ContextDataCollator(
        tok,
        model=model,
        prev_lookup=lookup,
        idx_field=idx_field,
        sent_idx_field=sent_idx_field,
    )
