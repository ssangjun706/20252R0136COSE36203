# -*- coding: utf-8 -*-
'''
eval.py

학습/베이스라인 모델 로드
-> test split 번역 생성
-> BLEU 측정
-> 결과 저장
'''

import os, contextlib, logging
from math import ceil
import torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..data.loaders import load_eval_dataset
from .configs import CONFIGS
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs): return x


def setup_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)

def _forced_bos_id(tok, cfg):
    """
    모델 타입에 따라 강제 BOS(토큰 id) 결정.
      - NLLB: tokenizer 토큰을 직접 id로 변환
      - M2M100: 전용 헬퍼(get_lang_id) 사용
      - 그 외: 지원 안 함
    """
    name = cfg.MODEL_NAME.lower()
    if "nllb" in name:
        # NLLB: tokenizer 토큰을 직접 id로 변환
        return tok.convert_tokens_to_ids(cfg.TGT_LANG)
    if "m2m100" in name:
        # M2M100: 전용 헬퍼
        return tok.get_lang_id(cfg.TGT_LANG)
    return None  # 그 외엔 사용 안 함

def _get_device_and_dtype(cfg):
    """디바이스(cuda/cpu) 및 데이터 타입(float16/bfloat16/None) 결정"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if dev.type == "cuda":
        if getattr(cfg, "BF16", False) and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif getattr(cfg, "FP16", False):
            dtype = torch.float16
    return dev, dtype

def _amp_context(dev, dtype):
    """
    AMP(auto mixed precision) 컨텍스트 헬퍼.
    GPU + FP16/BF16인 경우에만 활성화.
    """
    if dev.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=dtype)
        return torch.cuda.amp.autocast(dtype=dtype)
    return contextlib.nullcontext()

def _build_generate_kwargs(cfg, forced_id):
    """
    generation 관련 하이퍼파라미터를 모아 dict로 생성.
      - CFG.GEN_MODE: "beam"(기본) or "sample"
        - beam 모드: num_beams, max_length 사용
        - sample 모드: do_sample, top_k, top_p, temperature 사용
    """
    gen_mode = getattr(cfg, "GEN_MODE", "beam").lower()
    max_length = getattr(cfg, "MAX_TGT", 256)
    kwargs = {"max_length": max_length}

    if gen_mode == "sample":
        kwargs.update(
            do_sample=True,
            top_k=getattr(cfg, "TOP_K", 0),
            top_p=getattr(cfg, "TOP_P", 0.9),
            temperature=getattr(cfg, "TEMPERATURE", 1.0),
            num_beams=1,
        )
    else:
        kwargs.update(
            num_beams=getattr(cfg, "GEN_BEAMS", 5),
            do_sample=False,
        )
    if forced_id is not None:
        kwargs["forced_bos_token_id"] = forced_id

    return kwargs


# -----------------------
# main()
# -----------------------
def main():
    logger = setup_logger()
    cfg = CONFIGS["lemonmint_en2ko"]  # 원하는 설정으로 변경 가능 (일단 하드코딩 해둠. TODO: argparse 등으로 개선 가능)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # model_dir = os.path.join(cfg.OUTPUT_DIR, "final")
    # load_dir = model_dir if os.path.isdir(model_dir) else cfg.MODEL_NAME

    # 데이터셋 로딩
    logger.info(f"데이터셋 로딩... SOURCE={cfg.SOURCE}, FORMAT={cfg.FORMAT}")
    test_ds = load_eval_dataset(cfg, logger)
    logger.info(f"테스트셋 크기: {len(test_ds)}")

    # 모델 및 토크나이저 로딩
    final_model_dir = os.path.join(cfg.OUTPUT_DIR, "final")
    load_dir = final_model_dir if os.path.isdir(final_model_dir) else cfg.MODEL_NAME
    logger.info(f"모델 및 토크나이저 로딩... from {load_dir}")
    tok = AutoTokenizer.from_pretrained(load_dir)

    # 언어 코드 설정
    if hasattr(tok, "src_lang") and cfg.SRC_LANG:
        tok.src_lang = cfg.SRC_LANG

    dev, dtype = _get_device_and_dtype(cfg)
    logger.info(f"디바이스: {dev}, dtype: {dtype}")

    model_kwargs = {"torch_dtype": dtype} if dtype else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(load_dir, **model_kwargs).to(dev)
    model.eval()

    forced_id = _forced_bos_id(tok, cfg)
    gen_kwargs = _build_generate_kwargs(cfg, forced_id)
    logger.info(f"생성 옵션: {gen_kwargs}")

    bs = getattr(cfg, "EVAL_BS", 16)
    num_batches = ceil(len(test_ds) / bs)

    hyps, refs = [], []
    amp_context = _amp_context(dev, dtype)

    # 번역 생성 루프
    logger.info("번역 생성 시작...")
    with torch.inference_mode():
        for start in tqdm(range(0, len(test_ds), bs), total=num_batches, desc="Evaluating"):
            batch = test_ds[start:start+bs]

            enc = tok(
                batch["src"], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=getattr(cfg, "MAX_SRC", 256),
            )
            enc = {k: v.to(dev) for k, v in enc.items()}

            with amp_context:
                gen = model.generate(**enc, **gen_kwargs)

            out = tok.batch_decode(gen, skip_special_tokens=True)
            hyps.extend(out) # 모델 번역
            refs.extend(batch["tgt"]) # 정답 번역

    # BLEU 계산
    logger.info("BLEU 계산 중...")
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    logger.info(f"BLEU: {bleu.score:.2f}")

    with open(f"{cfg.OUTPUT_DIR}/hyp.txt","w",encoding="utf-8") as f:
        f.write("\n".join(hyps))
    with open(f"{cfg.OUTPUT_DIR}/ref.txt","w",encoding="utf-8") as f:
        f.write("\n".join(refs))

if __name__ == "__main__":
    main()