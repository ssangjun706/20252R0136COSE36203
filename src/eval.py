# -*- coding: utf-8 -*-
'''
src/eval.py

학습/베이스라인 모델 로드
-> test split 번역 생성
-> BLEU 측정
-> 결과 저장
'''
import os
from math import ceil

import torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..data.loaders import load_eval_dataset
from .configs import CONFIGS
from .utils import * # main을 제외하고 모두 utils로 보내서 정리함.
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs): return x

# -----------------------
# main()
# -----------------------
def main():
    logger = setup_logger()

    # 데이터셋 선택 (aihub_en2ko / lemonmint_en2ko / ...)
    cfg = CONFIGS["lemonmint_en2ko"]  # 일단 하드코딩 해둠. 추후 필요시: argparse 등으로 개선
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # model_dir = os.path.join(cfg.OUTPUT_DIR, "final")
    # load_dir = model_dir if os.path.isdir(model_dir) else cfg.MODEL_NAME

    # 모델 및 토크나이저 로딩
    final_model_dir = os.path.join(cfg.OUTPUT_DIR, "final")
    load_dir = final_model_dir if os.path.isdir(final_model_dir) else cfg.MODEL_NAME
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    logger.info(f"모델 소스: {load_dir}")

    # 데이터셋 로딩
    logger.info(f"데이터셋 로딩... SOURCE={cfg.SOURCE}, FORMAT={cfg.FORMAT}")
    test_ds = load_eval_dataset(cfg, logger)
    inspect_dataset(test_ds, tok, cfg, logger) # 데이터셋에 대한 전반적인 로그 출력 함수    

    # (옵션) 랜덤 서브샘플링
    test_ds = maybe_subsample_dataset(test_ds, cfg, logger)
    logger.info(f"실제 평가에 사용할 샘플 수: {len(test_ds):,}")

    # 언어 코드 설정
    if hasattr(tok, "src_lang") and cfg.SRC_LANG:
        tok.src_lang = cfg.SRC_LANG

    dev, dtype = get_device_and_dtype(cfg)
    logger.info(f"디바이스: {dev}, dtype: {dtype}")

    model_kwargs = {"torch_dtype": dtype} if dtype else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(load_dir, **model_kwargs).to(dev)
    model.eval()

    forced_id = forced_bos_id(tok, cfg)
    gen_kwargs = build_generate_kwargs(cfg, forced_id)
    logger.info(f"생성 옵션: {gen_kwargs}")

    amp_ctx = amp_context(dev, dtype)

    # ----------------
    # #1) 샘플 단위 번역 모드: 샘플 전체를 한 번에 번역 (기존 방식임.)
    # #2) 문장 단위 번역 모드: 샘플을 문장으로 잘라서 문장 단위 번역 후 다시 합치기
    # ----------------
    sentence_level = getattr(cfg, "SENTENCE_LEVEL", True) # !! 실전 측정시 반드시 사용 필요 !!

    # #1) 샘플 단위 번역 모드
    if not sentence_level:
        logger.info("[EVAL] 샘플 단위 번역 모드로 평가합니다.")
        src_texts = test_ds["src"]
        refs = test_ds["tgt"]

        hyps = translate_batch_texts(
            model, tok, dev, amp_ctx, gen_kwargs, src_texts, cfg
        )

    # #2) 문장 단위 번역 모드
    else:
        logger.info("[EVAL] 문장 단위 번역 모드로 평가합니다.")
        # 1) 샘플별로 문장 분절
        all_sent_src = []
        mapping = []  # (sample_idx, sent_idx)
        for i in range(len(test_ds)):
            s = test_ds[i]["src"]
            sents = split_into_sentences(s)
            for j, sent in enumerate(sents):
                all_sent_src.append(sent)
                mapping.append((i, j))

        logger.info(f"[SENT] 문장 단위 총 개수: {len(all_sent_src):,}")

        # 2) 문장 단위로 번역 수행 TODO: 추후 모델 사용시 파인튜닝 파라미터 ON/OFF 구현 필요
        all_sent_hyp = translate_batch_texts(
            model, tok, dev, amp_ctx, gen_kwargs, all_sent_src, cfg
        )

        assert len(all_sent_hyp) == len(mapping)

        # 3) 샘플 단위로 다시 묶기
        num_samples = len(test_ds)
        grouped_hyp = [[] for _ in range(num_samples)]
        for (i, j), hyp in zip(mapping, all_sent_hyp):
            grouped_hyp[i].append(hyp)

        # 샘플 단위로 이어 붙이기 (간단하게 공백으로 연결)
        hyps = [" ".join(sents) for sents in grouped_hyp]
        refs = test_ds["tgt"]


    # BLEU 계산
    logger.info("BLEU 계산 중...")
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    logger.info(f"BLEU: {bleu.score:.2f}")

    with open(f"{cfg.OUTPUT_DIR}/hyp.txt","w",encoding="utf-8") as f:
        f.write("\n".join(hyps)) # 모델 번역
    with open(f"{cfg.OUTPUT_DIR}/ref.txt","w",encoding="utf-8") as f:
        f.write("\n".join(refs)) # 정답 번역


if __name__ == "__main__":
    main()