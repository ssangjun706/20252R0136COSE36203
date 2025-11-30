# -*- coding: utf-8 -*-
"""
prototype/model.py

컨텍스트-aware 구조 정의 + wrapper 팩토리
- Adapter, FiLM, ContextProjector
- ContextAwareMTWrapper
- ContextConfig, ContextCache
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# 1) 기본 모듈
# ------------------------

class Adapter(nn.Module):
    """Bottleneck Adapter: h -> h + h', where h' = U @ f(D @ h)."""
    def __init__(self, d_model: int, bottleneck: int = 96):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=True)
        self.up   = nn.Linear(bottleneck, d_model, bias=True)
        nn.init.zeros_(self.up.weight)   # 안정 수렴
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(F.relu(self.down(h)))


class FiLM(nn.Module):
    """FiLM: h -> γ ⊙ h + β, where [γ; β] = A @ ct."""
    def __init__(self, d_model: int, r: int):
        super().__init__()
        self.proj = nn.Linear(r, 2 * d_model, bias=True)

    def forward(self, h: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        # h: [B, L, d], ct: [B, r]
        gamma_beta = self.proj(ct)      # [B, 2d]
        d = h.size(-1)
        gamma = gamma_beta[..., :d]     # [B, d]
        beta  = gamma_beta[..., d:]     # [B, d]
        gamma = gamma.unsqueeze(1)      # [B, 1, d]
        beta  = beta.unsqueeze(1)       # [B, 1, d]
        return gamma * h + beta


class ContextProjector(nn.Module):
    """
    직전 문장 encoder hidden을 '평균풀링→정규화→선형변환'을 통해 r차원으로 축약.
    H_prev: [B, L_prev, d] -> context: [B, r]
    """
    def __init__(self, d_model: int, r: int = 96):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, r, bias=True)

    def forward(self, H_prev: torch.Tensor) -> torch.Tensor:
        c = H_prev.mean(dim=1)          # [B, d]
        c = self.ln(c)
        context = self.proj(c)          # [B, r]
        return context


# class MiniPrefixKV(nn.Module):
#     """
#     미니 프리픽스 K/V 생성기. ct -> Kctx, Vctx
#     """
#     def __init__(self, r: int, d_k: int, d_v: int, m: int = 8):
#         super().__init__()
#         self.m = m
#         self.Kp = nn.Linear(r, m * d_k, bias=True)
#         self.Vp = nn.Linear(r, m * d_v, bias=True)

#     def forward(self, ct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # ct: [B, r] -> Kctx: [B, m, d_k], Vctx: [B, m, d_v]
#         B = ct.size(0)
#         Kctx = self.Kp(ct).view(B, self.m, -1)
#         Vctx = self.Vp(ct).view(B, self.m, -1)
#         return Kctx, Vctx


class ScalarGate(nn.Module):
    """레이어별 스칼라 게이트 g = σ(w^T ct + b)."""
    def __init__(self, r: int):
        super().__init__()
        self.w = nn.Linear(r, 1, bias=True)

    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        # ct: [B, r] -> g: [B, 1, 1]
        g = torch.sigmoid(self.w(ct)).unsqueeze(1)
        return g


# ------------------------
# 2) context-aware 래퍼
# ------------------------

@dataclass
class ContextConfig:
    r: int = 96                 # context bottleneck
    bottleneck: int = 96        # adapter bottleneck
    prefix_m: int = 8           # mini prefix length
    topk_decoder_layers_for_prefix: int = 3 # 상위 N 레이어에만 prefix/gate
    apply_on_encoder_final: bool = True     # True: 최종은닉에 적용, False: 레이어별 hook
    gate_with_similarity: bool = False      # 유사도 감쇠 옵션
    sim_clip_min: float = 0.0
    sim_clip_max: float = 1.0


def infer_dims_from_base(base) -> Tuple[int, int, int]:
    """
    HF base 모델(config)에서 d_model, d_k, d_v를 추론.
    - d_k, d_v는 보통 d_model / num_heads 로 둔다.
    """
    if not hasattr(base, "config"):
        raise ValueError("Base model has no config")

    d_model = base.config.d_model
    if hasattr(base.config, "encoder_attention_heads"):
        n_heads = base.config.encoder_attention_heads
    else:
        n_heads = base.config.num_attention_heads

    d_k = d_v = d_model // n_heads
    return d_model, d_k, d_v

class ContextAwareMTWrapper(nn.Module):
    """
    HuggingFace encoder-decoder 모델을 감싼 컨텍스트-어댑터 래퍼.
    - encoder: FiLM + Adapter
    - decoder: encoder_hidden_states 앞에 prefix_hidden 붙여서 "문맥 토큰" 제공
              (정교한 K/V hook 버전은 추후 확장)
    """
    def __init__(self, base_model: nn.Module, cfg: ContextConfig):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False  # 백본 freeze

        # ---- unfreeze 파라미터 정의 ----
        self.cfg = cfg

        d_model, d_k, d_v = infer_dims_from_base(base_model)

        self.ctx_proj = ContextProjector(d_model, cfg.r)
        self.film = FiLM(d_model, cfg.r)
        self.adapter = Adapter(d_model, cfg.bottleneck)

        # 간단 버전: prefix_hidden을 d_model 차원으로 생성
        self.prefix_proj = nn.Linear(cfg.r, cfg.prefix_m * d_model, bias=True)

        # 디코더용 글로벌 스칼라 게이트 (고급: 레이어별로 분리 가능)
        self.scalar_gate = ScalarGate(cfg.r)

    # ------ encoder/decoder helper ------

    def _get_encoder(self):
        if hasattr(self.base, "model") and hasattr(self.base.model, "encoder"):
            return self.base.model.encoder
        if hasattr(self.base, "encoder"):
            return self.base.encoder
        raise ValueError("Encoder not found")

    def _get_decoder(self):
        if hasattr(self.base, "model") and hasattr(self.base.model, "decoder"):
            return self.base.model.decoder
        if hasattr(self.base, "decoder"):
            return self.base.decoder
        raise ValueError("Decoder not found")

    # ------ context 추출 ------

    @torch.no_grad()
    def encode_prev_and_cache_ct(self, prev_input_ids, prev_attention_mask) -> torch.Tensor:
        enc = self._get_encoder()(
            input_ids=prev_input_ids,
            attention_mask=prev_attention_mask,
            return_dict=True,
        )
        H_prev = enc.last_hidden_state  # [B, L_prev, d]
        ct = self.ctx_proj(H_prev)      # [B, r]
        return ct

    # ------ forward ------

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        prev_ct: Optional[torch.Tensor] = None,
        prev_input_ids=None,
        prev_attention_mask=None,
        **kwargs
    ):
        """
        두 모드:
        - prev_ct 제공: 실시간 루프에서 캐시 사용
        - prev_input_ids 제공: 학습 중 직전문장으로부터 ct 계산
        """
        if prev_ct is None:
            if prev_input_ids is None or prev_attention_mask is None:
                raise ValueError("Provide prev_ct or prev_input_ids+prev_attention_mask")
            with torch.no_grad():
                prev_ct = self.encode_prev_and_cache_ct(prev_input_ids, prev_attention_mask)

        encoder = self._get_encoder()
        enc_out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        H = enc_out.last_hidden_state  # [B, L, d]

        # encoder 은닉에 FiLM + Adapter
        H = self.film(H, prev_ct)
        H = self.adapter(H)

        B, L, d_model = H.size()

        # context 기반 prefix hidden 생성
        prefix_hidden = self.prefix_proj(prev_ct).view(B, self.cfg.prefix_m, d_model)  # [B, m, d]
        enc_hidden_with_prefix = torch.cat([prefix_hidden, H], dim=1)                  # [B, m+L, d]

        if attention_mask is not None:
            prefix_mask = torch.ones(B, self.cfg.prefix_m, dtype=attention_mask.dtype, device=attention_mask.device)
            enc_mask_with_prefix = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            enc_mask_with_prefix = None

        decoder = self._get_decoder()

        # 간단 버전: decoder는 그대로 사용, encoder_hidden_states만 확장
        if decoder_input_ids is not None:
            dec_out = decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=enc_hidden_with_prefix,
                encoder_attention_mask=enc_mask_with_prefix,
                return_dict=True,
            )
        else:
            # generate 시에는 base.generate()를 쓰는 게 맞으므로,
            # 이 경로는 학습/Teacher forcing에 주로 사용됨.
            dec_out = decoder(
                input_ids=None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=enc_hidden_with_prefix,
                encoder_attention_mask=enc_mask_with_prefix,
                return_dict=True,
            )

        # LM head 호출
        if hasattr(self.base, "lm_head"):
            logits = self.base.lm_head(dec_out.last_hidden_state)
        elif hasattr(self.base, "model") and hasattr(self.base.model, "lm_head"):
            logits = self.base.model.lm_head(dec_out.last_hidden_state)
        else:
            raise ValueError("LM head not found on base model")

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "encoder_hidden_states": H, "ct": prev_ct}


# ------------------------
# 3) 유틸: ContextCache
# ------------------------

class ContextCache:
    """직전 ct 캐시. 스트림 ID 별로 관리."""
    def __init__(self):
        self._mem: Dict[Any, torch.Tensor] = {}

    def get(self, stream_id) -> Optional[torch.Tensor]:
        return self._mem.get(stream_id, None)

    def update(self, stream_id, ct: torch.Tensor):
        self._mem[stream_id] = ct

    def clear(self, stream_id=None):
        if stream_id is None:
            self._mem.clear()
        else:
            self._mem.pop(stream_id, None)


# ------------------------
# 4) 팩토리: cfg + base → context 모델
# ------------------------

def build_context_model(base_model: nn.Module, cfg) -> ContextAwareMTWrapper:
    """
    CFG 안에 컨텍스트 관련 하이퍼파라미터가 있다면 여기서 ContextConfig로 옮겨 담을 수 있음.
    일단은 기본값만 사용하고, 추후 CFG.CONTEXT_* 필드 추가해서 연결하는 구조로 설계.
    """
    ctx_cfg = ContextConfig(
        r=getattr(cfg, "CTX_R", 96),
        bottleneck=getattr(cfg, "CTX_BOTTLENECK", 96),
        prefix_m=getattr(cfg, "CTX_PREFIX_M", 8),
        topk_decoder_layers_for_prefix=getattr(cfg, "CTX_TOPK_DEC_LAYERS", 3),
    )
    return ContextAwareMTWrapper(base_model, ctx_cfg)