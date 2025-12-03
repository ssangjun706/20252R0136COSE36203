# -*- coding: utf-8 -*-
"""
src/baseline.py

백본 모델
- HuggingFace seq2seq 백본 모델/토크나이저 빌더
- 모델 입력(feature) 전처리 함수 빌더   # cf) loaders.py에서는 데이터 형식 통일 전처리 수행. (로더는 모델과 독립적.)
- DataCollator 빌더
"""

from typing import Tuple, Callable, Dict, Any
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
        if hasattr(tok, "as_target_tokenizer"):
            # 허깅페이스 최신 스타일 (NLLB 등)
            with tok.as_target_tokenizer():
                labels = tok(
                    examples["tgt"],
                    max_length=max_tgt,
                    truncation=True,
                )
        else:
            # 구형/기타 토크나이저 호환용
            labels = tok(
                text_target=examples["tgt"],
                max_length=max_tgt,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def build_collator(tok, model) -> DataCollatorForSeq2Seq:
    """
    Seq2Seq Trainer용 데이터 콜레이터.
    padding, label shifting 등을 내부에서 처리한다.
    """
    return DataCollatorForSeq2Seq(tok, model=model)
