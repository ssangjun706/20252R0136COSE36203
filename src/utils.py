# -*- coding: utf-8 -*-
'''
src/utils.py

'''
import logging, torch, contextlib, random, re
from tqdm import tqdm
from math import ceil


# --------------- 생성 관련 헬퍼 --------------------
def forced_bos_id(tok, cfg):
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

def get_device_and_dtype(cfg):
    """디바이스(cuda/cpu) 및 데이터 타입(float16/bfloat16/None) 결정"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if dev.type == "cuda":
        if getattr(cfg, "BF16", False) and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif getattr(cfg, "FP16", False):
            dtype = torch.float16
    return dev, dtype

def amp_context(dev, dtype):
    """
    AMP(auto mixed precision) 컨텍스트 헬퍼.
    GPU + FP16/BF16인 경우에만 활성화.
    """
    if dev.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=dtype)
        return torch.cuda.amp.autocast(dtype=dtype)
    return contextlib.nullcontext()

def build_generate_kwargs(cfg, forced_id):
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

# ---------------- 문장 분절 ----------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_into_sentences(text: str) -> list[str]:
    """
    매우 단순한 영어/기본용 문장 분절기.
    - '.', '!', '?' 뒤의 공백 기준으로 분절
    - 추후 한국어/정교한 splitter로 교체 가능
    """
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]


# ---------------- 랜덤 샘플링 ----------------
def maybe_subsample_dataset(ds, cfg, logger, max_samples=2000):
    n = len(ds)
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return ds

    logger.info(f"[EVAL] 전체 {n:,}개 중 {max_samples:,}개를 랜덤 샘플링하여 평가합니다.")
    rng = random.Random(getattr(cfg, "SEED", 1337))
    indices = list(range(n))
    rng.shuffle(indices)
    indices = sorted(indices[:max_samples])

    return ds.select(indices)


# ---------------- 문장 단위 번역용 helper ----------------
def translate_batch_texts(model, tok, dev, amp_ctx, gen_kwargs, texts, cfg):
    """
    texts: List[str]
    반환: List[str] (동일 길이)
    """
    bs = getattr(cfg, "EVAL_BS", 16)
    num_batches = ceil(len(texts) / bs)
    hyps = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), bs), total=num_batches, desc="Evaluating"):
            batch_src = texts[start:start+bs]   # ← 행 단위 슬라이싱
            enc = tok(
                batch_src,                      # ← 문자열 리스트 ["문장1", "문장2", ...]
                return_tensors="pt",
                padding=True,
                truncation=True,                # ← 입력 문장의 토큰이 MAX_SRC 보다 길면, 나머지는 버림.
                max_length=getattr(cfg, "MAX_SRC", 256),
            )
            enc = {k: v.to(dev) for k, v in enc.items()}

            with amp_ctx:
                gen = model.generate(**enc, **gen_kwargs)

            out = tok.batch_decode(gen, skip_special_tokens=True)
            hyps.extend(out)
    return hyps


# ---------------- LOGGER SETUP ----------------
def setup_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)

def inspect_dataset(ds, tokenizer, cfg, logger, sample_limit=50000):
    """
    데이터셋 분석(로그출력용):
      - 총 문장 수
      - 총 토큰 수
      - 평균/최대 토큰 길이
    sample_limit: 속도 제한 (대규모 HF dataset 대응)
    """
    N_full = len(ds)
    logger.info(f"[데이터셋 분석] 총 샘플 수: {N_full:,}")

    # 샘플이 너무 크면 일부만 샘플링
    if N_full > sample_limit:
        logger.info(f"[데이터셋 분석] {sample_limit:,}개 샘플링을 통해 전체 토큰 통계를 추정합니다.")
        ds = ds.select(range(sample_limit))
        N = sample_limit
        option = "추정 "
    else:
        N = N_full
        option = ""

    lengths = []
    total_tokens = 0

    for text in ds["src"]:
        # 모델 tokenizer 기준 토큰화 길이 측정
        ids = tokenizer.encode(
            text,
            truncation=False,
            add_special_tokens=False
        )
        L = len(ids)
        lengths.append(L)
        total_tokens += L

    avg_len = total_tokens / N
    max_len = max(lengths) if lengths else 0

    logger.info(f"[데이터 분석 결과]")
    logger.info(f"  - 표본 샘플 수: {N:,}")
    logger.info(f"  - 평균 토큰 길이: {avg_len:,.2f}")
    logger.info(f"  - 최장 토큰 길이: {max_len:,}")
    logger.info(f"  - {option}전체 토큰 길이: {total_tokens * N_full:,.0f}")

    return {
        "num_samples": N,
        "total_tokens": total_tokens,
        "avg_len": avg_len,
        "max_len": max_len,
    }
