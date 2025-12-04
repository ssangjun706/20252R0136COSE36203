# -*- coding: utf-8 -*-
"""
src/train.py

트레이닝 엔트리 포인트.
CFG → 데이터 → (baseline/context) 모델 → Trainer
- CFG 선택
- 데이터셋 로딩
- (baseline or context-aware) 모델 구성
- Trainer 실행
"""

import os
import contextlib

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

from .configs import CONFIGS
from .configs._config import CFG
from .baseline import build_backbone, build_preprocess_fn, build_collator
from ..prototype.model import build_context_model
from ..data.loaders import load_train_val_datasetdict
from .utils import *


def main():
    logger = setup_logger()

    # 1) 데이터셋 선택 (aihub_en2ko / lemonmint_en2ko / wiki_en2ko ...)
    cfg: CFG = CONFIGS["wiki_en2ko"] # 일단 하드코딩 해둠. 추후 필요시: argparse 등으로 개선
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    set_seed(cfg.SEED)

    # 2) 데이터셋 로딩 (train/validation)
    logger.info(f"[TRAIN] 데이터셋 로딩... SOURCE={cfg.SOURCE}, FORMAT={cfg.FORMAT}")
    ds_dict = load_train_val_datasetdict(cfg, logger)
    train_ds = ds_dict["train"]
    val_ds   = ds_dict["validation"]
    logger.info(f"[TRAIN] train={len(train_ds):,}, val={len(val_ds):,}")

    # 3) 백본 모델 + 토크나이저
    tok, base_model = build_backbone(cfg)

    inspect_dataset(train_ds, tok, logger)

    # 4) context-aware 모델 래핑
    model = build_context_model(base_model, cfg)

    # 5) 전처리/콜레이터
    preprocess = build_preprocess_fn(tok, cfg)

    # src/tgt → tokenized
    tokenized_train = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
    )
    tokenized_val = val_ds.map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
    )

    collator = build_collator(
        tok,
        model,
        datasets=[tokenized_train, tokenized_val],
        cfg=cfg,
    )

    # 6) TrainingArguments
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,

        # 평가 / 저장 / 로깅 전략
        eval_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        save_strategy="steps",
        save_steps=cfg.SAVE_STEPS,
        logging_strategy="steps",
        logging_steps=cfg.LOG_STEPS,

        # 학습 하이퍼파라미터
        per_device_train_batch_size=cfg.TRAIN_BS,
        per_device_eval_batch_size=cfg.EVAL_BS,
        learning_rate=cfg.LR,
        weight_decay=cfg.WD,
        num_train_epochs=cfg.EPOCHS,

        # seq2seq 관련
        predict_with_generate=True,
        generation_max_length=cfg.MAX_TGT,

        # mixed precision
        fp16=getattr(cfg, "FP16", False),
        bf16=getattr(cfg, "BF16", False),

        report_to="none",
        seed=cfg.SEED,
    )

    # 7) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tok,
    )

    # 8) 학습
    logger.info("[TRAIN] 학습 시작")
    trainer.train()

    # 9) 저장 (context 모델도 base와 동일하게 save_pretrained 사용 가능)
    final_dir = os.path.join(cfg.OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)

    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    logger.info(f"[TRAIN] 모델 저장 완료: {final_dir}")


if __name__ == "__main__":
    main()
