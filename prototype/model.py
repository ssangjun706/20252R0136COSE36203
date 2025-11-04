# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# 1) 기본 모듈
# ------------------------

class Adapter(nn.Module):
    """Bottleneck Adapter: h -> h + U σ(D h)."""
    def __init__(self, d_model: int, bottleneck: int = 96):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=True)
        self.up   = nn.Linear(bottleneck, d_model, bias=True)
        nn.init.zeros_(self.up.weight)   # 안정 수렴
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(F.relu(self.down(h)))


class FiLM(nn.Module):
    """FiLM: h -> γ ⊙ h + β, where [γ, β] = A @ ct."""
    def __init__(self, d_model: int, r: int):
        super().__init__()
        self.proj = nn.Linear(r, 2 * d_model, bias=True)

    def forward(self, h: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        # h: [B, L, d], ct: [B, r]
        gamma_beta = self.proj(ct)                 # [B, 2d]
        d = h.size(-1)
        gamma, beta = gamma_beta[..., :d], gamma_beta[..., d:]
        gamma = gamma.unsqueeze(1)                 # [B,1,d]
        beta  = beta.unsqueeze(1)
        return gamma * h + beta


class ContextProjector(nn.Module):
    """
    직전 문장 인코더 은닉을 평균풀링→LN→선형 r차원으로 축약.
    H_prev: [B, Lp, d] -> ct: [B, r]
    """
    def __init__(self, d_model: int, r: int = 96):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, r, bias=True)

    def forward(self, H_prev: torch.Tensor) -> torch.Tensor:
        c = H_prev.mean(dim=1)          # [B, d]
        c = self.ln(c)
        ct = self.proj(c)               # [B, r]
        return ct


class MiniPrefixKV(nn.Module):
    """
    미니 프리픽스 K/V 생성기. ct -> Kctx, Vctx
    """
    def __init__(self, r: int, d_k: int, d_v: int, m: int = 8):
        super().__init__()
        self.m = m
        self.Kp = nn.Linear(r, m * d_k, bias=True)
        self.Vp = nn.Linear(r, m * d_v, bias=True)

    def forward(self, ct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ct: [B, r] -> Kctx: [B, m, d_k], Vctx: [B, m, d_v]
        B = ct.size(0)
        Kctx = self.Kp(ct).view(B, self.m, -1)
        Vctx = self.Vp(ct).view(B, self.m, -1)
        return Kctx, Vctx


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
# 2) HF 모델 래퍼
# ------------------------

@dataclass
class ContextConfig:
    r: int = 96                 # context bottleneck
    bottleneck: int = 96        # adapter bottleneck
    prefix_m: int = 8           # mini prefix length
    apply_on_encoder_final: bool = True  # True: 최종은닉에 적용, False: 레이어별 hook
    topk_decoder_layers_for_prefix: int = 3  # 상위 N 레이어에만 prefix/gate
    gate_with_similarity: bool = False       # 유사도 감쇠 옵션
    sim_clip_min: float = 0.0
    sim_clip_max: float = 1.0


class ContextAwareMTWrapper(nn.Module):
    """
    HuggingFace encoder-decoder 모델을 감싼 컨텍스트-어댑터 래퍼.
    - encoder: FiLM + Adapter
    - decoder: cross-attn Q 게이팅 + mini prefix K/V
    """
    def __init__(self, base_model: nn.Module, d_model: int, d_k: int, d_v: int, cfg: ContextConfig):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False      # 백본 freeze

        self.cfg = cfg
        self.ctx_proj = ContextProjector(d_model, cfg.r)

        # 인코더 측
        self.film = FiLM(d_model, cfg.r)
        self.adapter = Adapter(d_model, cfg.bottleneck)

        # 디코더 측: 상위 N개 레이어용 모듈 리스트
        self.scalar_gates = nn.ModuleList()
        self.prefix_kv    = nn.ModuleList()

        n_dec_layers = self._get_num_decoder_layers()
        pick = list(range(max(0, n_dec_layers - cfg.topk_decoder_layers_for_prefix), n_dec_layers))
        self.target_dec_layers = set(pick)
        for _ in range(n_dec_layers):
            self.scalar_gates.append(ScalarGate(cfg.r))
            self.prefix_kv.append(MiniPrefixKV(cfg.r, d_k, d_v, cfg.prefix_m))

        # 선택: 레이어별 hook으로 인코더 중간에 삽입하고 싶다면 아래 메서드 참고
        # if not cfg.apply_on_encoder_final:
        #     self._register_encoder_hooks(d_model)

    # --------- 필수 헬퍼 ---------

    def _get_num_decoder_layers(self) -> int:
        # HF 표준 속성 추정
        if hasattr(self.base, "model") and hasattr(self.base.model, "decoder"):
            return len(self.base.model.decoder.layers)
        if hasattr(self.base, "decoder"):
            return len(self.base.decoder.layers)
        raise ValueError("Decoder layers not found")

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

    # --------- 컨텍스트 추출/캐시 ---------

    @torch.no_grad()
    def encode_prev_and_cache_ct(self, prev_input_ids, prev_attention_mask, **gen_kwargs) -> torch.Tensor:
        """
        직전 문장 인코딩 → ct 캐시 반환. 추론 루프에서 1스텝 선계산.
        """
        enc = self._get_encoder()
        enc_out = enc(input_ids=prev_input_ids,
                      attention_mask=prev_attention_mask,
                      return_dict=True)
        H_prev = enc_out.last_hidden_state  # [B, Lp, d]
        ct = self.ctx_proj(H_prev)          # [B, r]
        return ct

    # --------- 포워드 ---------

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
                raise ValueError("Provide either prev_ct or prev_input_ids+prev_attention_mask")
            with torch.no_grad():
                prev_ct = self.encode_prev_and_cache_ct(prev_input_ids, prev_attention_mask)

        # 1) 인코더 실행
        encoder = self._get_encoder()
        enc_out = encoder(input_ids=input_ids,
                          attention_mask=attention_mask,
                          output_hidden_states=True,
                          return_dict=True)
        # 최종 은닉에 FiLM+Adapter (간단 모드)
        H = enc_out.last_hidden_state                       # [B, L, d]
        H = self.film(H, prev_ct)
        H = self.adapter(H)                                 # [B, L, d]

        # 2) 디코더 실행 + cross-attn 개입
        decoder = self._get_decoder()

        # HF decoder는 encoder_hidden_states를 입력으로 받음
        # 여기에서 K/V 프리픽스를 concat해야 한다. 표준 모듈에는 직접 concat 지점이 없다.
        # 실용적 대안: prefix를 "가짜 토큰 은닉"으로 만들어 encoder_hidden_states 앞에 붙인다.
        # 주의: 이는 self-attn의 positional 의미를 바꾸지 않게 별도 마스크가 필요.
        # 아래는 스켈레톤 구현: 프리픽스를 생성해 concat하고, 마스크도 확장한다.

        B, L, d = H.size()
        Kctx_list, Vctx_list, gate_list = [], [], []

        # 상위 N 레이어에서만 prefix/gate를 쓰기 위해 메타정보를 전달
        # HF 디코더에 직접 레이어별 제어를 넣기 어렵기 때문에,
        # 여기서는 "공통 프리픽스 은닉"을 만들어 전달하고,
        # 게이트는 쿼리 축소 계수로 흉내낸다(간단 버전).
        # 고급: 각 레이어의 encoder_attn 모듈을 교체하거나 forward hook으로 Q scaling.

        # 공통 프리픽스 은닉을 하나 만들어 encoder_hidden_states 앞에 붙인다.
        # d_k, d_v 차원과 d_model이 달라도, attention 투영 전이므로 d_model로 생성해 충분히 근사 가능.
        # 간단히 d_model 프리픽스 생성:
        # 평균적으로 K/V 공간과 동일 선형 변환을 공유하므로 효과는 유지됨.
        # 필요시, 각 레이어별로 별도 prefix를 생성해 hook에서 주입하라.
        m = self.cfg.prefix_m
        prefix_proj = nn.Linear(self.cfg.r, m * d, bias=True).to(H.device)
        with torch.no_grad():
            prefix_hidden = prefix_proj(prev_ct).view(B, m, d)  # [B, m, d]

        enc_hidden_with_prefix = torch.cat([prefix_hidden, H], dim=1)   # [B, m+L, d]
        if attention_mask is not None:
            prefix_mask = torch.ones(B, m, dtype=attention_mask.dtype, device=attention_mask.device)
            enc_mask_with_prefix = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, m+L]
        else:
            enc_mask_with_prefix = None

        # 쿼리 게이팅: 간단 버전(전체 디코더에 스칼라 적용)
        # 고급: 레이어별 hook으로 Q를 스케일링
        gate_global = self.scalar_gates[-1](prev_ct)  # [B,1,1]
        # 디코더 입력을 게이트로 선형 축소해 효과를 부여
        # 실제로는 Q만 스케일해야 하지만 후킹 없이 근사: dec hidden을 스케일
        def _scale_decoder_inputs(x):
            return x * gate_global

        if decoder_input_ids is not None:
            # teacher forcing
            dec_in = decoder_input_ids
            # 간단 근사: 임베딩 행렬을 얻고 스케일. HF 내부에서 하므로 여기선 생략.
            out = decoder(
                input_ids=dec_in,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=enc_hidden_with_prefix,
                encoder_attention_mask=enc_mask_with_prefix,
                return_dict=True,
            )
        else:
            # generate step용: decoder_start_token_id 등은 베이스 모델의 generate 사용 권장
            out = decoder(
                input_ids=None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=enc_hidden_with_prefix,
                encoder_attention_mask=enc_mask_with_prefix,
                return_dict=True,
            )

        # LM head 호출은 베이스 모델에 위임
        if hasattr(self.base, "lm_head"):
            logits = self.base.lm_head(out.last_hidden_state)
        elif hasattr(self.base, "model") and hasattr(self.base.model, "lm_head"):
            logits = self.base.model.lm_head(out.last_hidden_state)
        else:
            raise ValueError("LM head not found on base model")

        loss = None
        if labels is not None:
            # 표준 CE
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits, "encoder_hidden_states": H, "gate": gate_global}

    # ------------------------
    # (선택) 고급: 인코더 레이어별 훅으로 FiLM+Adapter 삽입
    # ------------------------
    def _register_encoder_hooks(self, d_model: int):
        """
        HF 인코더의 각 block 출력에 FiLM+Adapter 적용하는 예시.
        모델마다 블록 명이 다를 수 있으니 실제 모듈 경로 확인 필요.
        """
        encoder = self._get_encoder()
        for i, layer in enumerate(encoder.layers):
            def _make_hook():
                def hook(module, inp, out):
                    # out: (hidden_states, present_key_value, ...)일 수도 있어 모델마다 다름
                    h = out[0] if isinstance(out, tuple) else out
                    # 여기서 prev_ct 접근이 필요. forward ctx에 담거나, thread-local 저장.
                    # 스켈레톤에서는 생략. 실전은 forward 중 self._current_ct를 set 후 사용.
                    ct = getattr(self, "_current_ct", None)
                    if ct is None:
                        return out
                    h2 = self.film(h, ct)
                    h2 = self.adapter(h2)
                    if isinstance(out, tuple):
                        out = (h2,) + out[1:]
                        return out
                    return h2
                return hook
            layer.register_forward_hook(_make_hook())


# ------------------------
# 3) 사용 예시
# ------------------------
"""
from transformers import MarianMTModel

base = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
d_model = base.config.d_model           # 예: 512, 768 등
d_k = d_v = d_model // base.config.encoder_attention_heads

cfg = ContextConfig(
    r=96, bottleneck=96, prefix_m=8,
    apply_on_encoder_final=True,        # 간단 모드
    topk_decoder_layers_for_prefix=3,
    gate_with_similarity=False
)

model = ContextAwareMTWrapper(base, d_model=d_model, d_k=d_k, d_v=d_v, cfg=cfg)

# 학습 루프 스케치
batch = next(iter(dataloader))
prev_ct = model.encode_prev_and_cache_ct(batch["prev_input_ids"], batch["prev_attention_mask"])
out = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    decoder_input_ids=batch["decoder_input_ids"],
    decoder_attention_mask=batch["decoder_attention_mask"],
    labels=batch["labels"],
    prev_ct=prev_ct,   # 실시간/학습 모두 가능
)
loss = out["loss"]
loss.backward()
optimizer.step()
"""

# ------------------------
# 4) 유틸: 실시간 루프용 캐시
# ------------------------

class ContextCache:
    """직전 ct 캐시. 대화 스트림마다 ID를 키로 관리."""
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
