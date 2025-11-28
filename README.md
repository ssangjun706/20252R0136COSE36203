# COSE362 Project - Korean-English Translation

252R COSE362 기계학습 프로젝트
2022320140 김준영
2022320000 김지찬
2022320000 송상준

모델 목표:
이전 문장의 정보를 현재 문장 번역에 반영하는 "경량 컨텍스트-aware MT"

## Project Structure

```
cose362/
├── data/
│   └── aihub_news/       # AIHub News Dataset
│   └── loaders.py        # compatible loader with many source and format
│   └── _xlsx.py          # using load xlsx format
├── prototype/
│   └── model.py          # 
├── src/
│   ├── __init__.py
│   ├── baseline.py       # 
│   ├── config.py         # Configuration settings
│   ├── eval.py           # Model evaluation scripts 
│   ├── train.py          # 
│   ├── utils.py          # utils
│   └── configs/
│       ├── __init__.py   # configs/ 내부 개별 데이터 세트 CFG 매핑 힌트 제공.
│       ├── _config.py    # CFG Parent Class 정의
│       ├── aihub.py      # 개별 데이터셋에 맞춘 CFG Child Class 정의
│       └── lemonmint.py  # 개별 데이터셋에 맞춘 CFG Child Class 정의
└── README.md
```

### 핵심 학습 및 평가 파이프라인

* `src/eval.py`
  학습된 모델(또는 베이스라인 모델)을 로드하고, 디바이스 및 데이터 타입 설정을 적용한다.
  필요한 경우 타깃 언어 토큰 강제 입력을 적용하며, 테스트 세트에 대한 번역 생성, SacreBLEU 기반 BLEU 계산, 생성 결과·레퍼런스 저장까지 수행한다.

* `data/loaders.py`
  데이터 셋의 소스가 로컬이든 허깅페이스든 상관없이 사용할 수 있는 형태로 가공해서 로드한다. 현재 지원 포맷은 xlsx, json, arrow 이다.

* `src/configs/_config.py`
  데이터 경로, 번역 방향, 모델 이름, 최대 길이, 언어 코드, 학습 하이퍼파라미터 등 전체 설정을 중앙에서 관리한다. 이 값들은 모든 데이터셋에 대한 학습/평가 스크립트에서 공통적으로 사용된다.

* `src/baseline.py`
  MarianMT 기반 베이스라인 모델을 위한 헬퍼들을 제공한다. 데이터셋 구성, 토크나이저·모델 로딩, 전처리 함수 준비, 데이터 collator 생성 등을 포함한다.

* `src/train.py`
  end-to-end 학습 루틴을 정의한다.
  설정 및 데이터셋 로드 → 언어 코드 기준 토크나이징 → `Seq2SeqTrainer` 설정(생성-friendly 옵션 포함) → 학습 수행 → 파인튜닝된 모델과 토크나이저 저장.



---

### 실험용 프로토타입 (`prototype/`)

* `prototype/model.py`
  맥락 정보를 반영하는 연구 지향 MT 프로토타입 모델을 포함한다.
  Adapter/FiLM 모듈, 컨텍스트 projection, prefix key/value generation, 디코더용 gating 유틸리티 등이 구현되어 있으며 Hugging Face encoder-decoder 백본 위에서 동작한다.

---


## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/lpaiu-cs/COSE362-Project.git
cd COSE362-Project
```

2. Install required packages:
```bash
pip install -r requirements.txt  # Note: requirements.txt will be added when dependencies are finalized
```

## Usage

### Training

To train the model:

```bash
python -m cose362.src.train
```

### Evaluation

To evaluate the model:

```bash
python -m cose362.src.eval
```

## Data

- `data/aihub_news/`: Contains various Excel files with different categories of news translations
  - 구어체(1).xlsx, 구어체(2).xlsx: Conversational style texts
  - 대화체.xlsx: Dialogue style texts
  - 문어체_뉴스(1-4).xlsx: Written style news texts
  - 문어체_한국문화.xlsx: Korean culture related texts
  - 문어체_종료.xlsx: Miscellaneous texts
  - 문어체_지자체웹사이트.xlsx: Local government website texts