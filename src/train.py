# -*- coding: utf-8 -*-
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from .config import CFG
from .utils import build_hfds_from_xlsx

def main():
    cfg = CFG()
    set_seed(cfg.SEED)

    ds = build_hfds_from_xlsx(cfg.RAW_DIR, cfg.DIRECTION, cache_dir=cfg.CACHE_DIR)

    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    # 언어 코드 지정
    if hasattr(tok, "src_lang"):
        tok.src_lang = cfg.SRC_LANG

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL_NAME)

    max_src, max_tgt = cfg.MAX_SRC, cfg.MAX_TGT
    def preprocess(examples):
        # 입력
        model_inputs = tok(examples["src"], max_length=max_src, truncation=True)
        # 타겟
        with tok.as_target_tokenizer() if hasattr(tok, "as_target_tokenizer") else contextlib.nullcontext():
            labels = tok(text_target=examples["tgt"], max_length=max_tgt, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=["src","tgt"])
    collator = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        save_steps=cfg.SAVE_STEPS,
        logging_steps=cfg.LOG_STEPS,
        per_device_train_batch_size=cfg.TRAIN_BS,
        per_device_eval_batch_size=cfg.EVAL_BS,
        learning_rate=cfg.LR,
        weight_decay=cfg.WD,
        num_train_epochs=cfg.EPOCHS,
        predict_with_generate=True,
        generation_max_length=cfg.MAX_TGT,
        fp16=getattr(cfg, "FP16", False),
        bf16=getattr(cfg, "BF16", False),
        report_to="none",
        seed=cfg.SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(f"{cfg.OUTPUT_DIR}/final")
    tok.save_pretrained(f"{cfg.OUTPUT_DIR}/final")

if __name__ == "__main__":
    import contextlib  # 위에서 사용
    main()
