# -*- coding: utf-8 -*-
import os, contextlib
import torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import CFG
from .utils import build_hfds_from_xlsx

def _forced_bos_id(tok, cfg):
    name = cfg.MODEL_NAME.lower()
    if "nllb" in name:
        # NLLB: tokenizer 토큰을 직접 id로 변환
        return tok.convert_tokens_to_ids(cfg.TGT_LANG)
    if "m2m100" in name:
        # M2M100: 전용 헬퍼
        return tok.get_lang_id(cfg.TGT_LANG)
    return None  # 그 외엔 사용 안 함

def main():
    cfg = CFG()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model_dir = f"{cfg.OUTPUT_DIR}/final"
    load_dir = model_dir if os.path.isdir(model_dir) else cfg.MODEL_NAME

    ds = build_hfds_from_xlsx(cfg.RAW_DIR, cfg.DIRECTION)  # test split
    tok = AutoTokenizer.from_pretrained(load_dir)

    # 언어 코드 설정
    if hasattr(tok, "src_lang"):
        tok.src_lang = cfg.SRC_LANG

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if dev.type == "cuda":
        if getattr(cfg, "BF16", False) and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif getattr(cfg, "FP16", False):
            dtype = torch.float16
    model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(load_dir, **model_kwargs).to(dev)
    model.eval()

    forced_id = _forced_bos_id(tok, cfg)

    hyps, refs = [], []
    bs = cfg.EVAL_BS

    def amp_context():
        if dev.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", dtype=dtype)
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    for i in range(0, len(ds["test"]), bs):
        batch = ds["test"][i:i+bs]
        enc = tok(batch["src"], return_tensors="pt", padding=True, truncation=True, max_length=cfg.MAX_SRC)
        enc = {k: v.to(dev) for k, v in enc.items()}
        gen_kwargs = dict(max_length=cfg.MAX_TGT, num_beams=cfg.GEN_BEAMS)
        if forced_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_id
        with torch.inference_mode():
            with amp_context():
                gen = model.generate(**enc, **gen_kwargs)
        out = tok.batch_decode(gen, skip_special_tokens=True)
        hyps.extend(out)
        refs.extend(batch["tgt"])

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"BLEU: {bleu.score:.2f}")
    with open(f"{cfg.OUTPUT_DIR}/hyp.txt","w",encoding="utf-8") as f: f.write("\n".join(hyps))
    with open(f"{cfg.OUTPUT_DIR}/ref.txt","w",encoding="utf-8") as f: f.write("\n".join(refs))

if __name__ == "__main__":
    main()
