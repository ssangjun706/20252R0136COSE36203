'''
CFG 객체 딕셔너리

파이프라인 구조:
  eval.py / train.py
  -> __init__.py: CONFIGS
  -> cfg = CONFIGS["데이터셋"]: CFG 객체 로드
    -> 여기서 CFG 클래스는 _config.py에 정의됨
'''
from .aihub import AIHUB_EN2KO
from .lemonmint import LEMONMINT_EN2KO
from .wiki import WIKI_EN2KO

CONFIGS = {
    "aihub_en2ko": AIHUB_EN2KO,
    "lemonmint_en2ko": LEMONMINT_EN2KO,
    "wiki_en2ko": WIKI_EN2KO,
}