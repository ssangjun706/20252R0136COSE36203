# -*- coding: utf-8 -*-
'''
src/eval.py

학습/베이스라인 모델 로드
-> test split 번역 생성
-> BLEU 측정
-> 결과 저장

TODO:
  - 문장 단위 번역 모드 구현 (인덱스 사용하도록)

'''
import os
import argparse
from math import ceil

import torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..prototype.model import ContextAwareMTWrapper
from ..data.loaders import load_dataset
from .configs import CONFIGS # -> __init__.py: CONFIGS 딕셔너리
from .utils import * # main을 제외하고 모두 utils로 보내서 정리함.
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs): return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-baseline", action="store_true", help="Skip context wrapper and run baseline backbone")
    args = parser.parse_args()

    logger = setup_logger()

    # 데이터셋 선택 (aihub_en2ko / lemonmint_en2ko / wiki_en2ko...)
    # 일단 하드코딩 해둠. 추후 필요시: argparse 등으로 개선
    cfg = CONFIGS["wiki_en2ko"]  # 각 dataset.py의 CFG 객체
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    '''
    eval 프로세스 개요:
    디바이스 설정 -> 모델 로딩 -> 데이터셋 로딩 -> 번역 생성 -> BLEU 계산
    '''

    # 디바이스 및 dtype 설정
    dev, dtype = get_device_and_dtype(cfg)
    logger.info(f"디바이스: {dev}, dtype: {dtype}")
    model_kwargs = {"dtype": dtype} if dtype else {}

    # 모델 및 토크나이저 로딩 (학습된 가중치가 있으면 그것을 우선 사용)
    final_model_dir = os.path.join(cfg.OUTPUT_DIR, "final")
    if args.use_baseline:
        load_dir = cfg.MODEL_NAME
        model = AutoModelForSeq2SeqLM.from_pretrained(load_dir, **model_kwargs).to(dev)
    else:
        if os.path.isdir(final_model_dir):
            logger.info(f"학습된 컨텍스트 모델을 로드합니다: {final_model_dir}")
            load_dir = final_model_dir
        else:
            raise RuntimeError(f"학습된 모델 디렉토리가 존재하지 않습니다. train.py를 먼저 실행하세요.")
        model = ContextAwareMTWrapper.from_pretrained(load_dir, **model_kwargs).to(dev)
        
    tok = AutoTokenizer.from_pretrained(load_dir)
    logger.info(f"모델 소스: {load_dir} (baseline={args.use_baseline})")

    model.eval() # 평가 모드로 전환

    # 데이터셋 로딩
    logger.info(f"데이터셋 로딩... SOURCE={cfg.SOURCE}, FORMAT={cfg.FORMAT}")
    test_ds = load_dataset(cfg, logger)
    inspect_dataset(test_ds, tok, logger) # 데이터셋에 대한 전반적인 로그 출력 함수

    # 언어 코드 설정
    if hasattr(tok, "src_lang") and cfg.SRC_LANG:
        tok.src_lang = cfg.SRC_LANG
    if hasattr(tok, "tgt_lang") and cfg.TGT_LANG:
        tok.tgt_lang = cfg.TGT_LANG
    
    forced_id = forced_bos_id(tok, cfg)
    gen_kwargs = build_generate_kwargs(cfg, forced_id)
    logger.info(f"생성 옵션: {gen_kwargs}")

    amp_ctx = amp_context(dev, dtype)

    # (옵션) 랜덤 서브샘플링: 문맥(idx) 단위로 샘플링해 동일 문맥 보존
    test_ds = maybe_subsample_by_idx(test_ds, cfg, logger)
    logger.info(f"실제 평가에 사용할 샘플 수: {len(test_ds):,}")

    # ----------------
    # #1) 샘플 단위 번역 모드: 샘플 전체를 한 번에 번역 (기존 방식임.)
    # #2) 문장 단위 번역 모드: 샘플을 문장으로 잘라서 문장 단위 번역 후 다시 합치기
    # ----------------
    # 베이스라인은 기존 설정 사용, 컨텍스트 모델은 문장 단위로 고정해 연속 문맥을 유지
    if args.use_baseline:
        sentence_level = getattr(cfg, "SENTENCE_LEVEL", True)
    else:
        sentence_level = True

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