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
│   └── aihub_news/       # AIHub News 데이터셋
│   └── loaders.py        # 소스와 포맷이 호환되게 하는 데이터 전처리
│   └── _xlsx.py          # xlsx 포맷 로드 도움 함수
│
├── prototype/
│   └── model.py          # 추후 src로 이동.
│
├── src/
│   ├── configs/  # 데이터셋 및 모델에 대한 configuration 정의
│   │   └── __init__.py   # configs/ 내부 개별 <dataset>.py 매핑 힌트.
│   │   └── _config.py    # CFG Parent Class 정의
│   │   └── aihub.py      # 개별 CFG Child Class 정의
│   │   └── lemonmint.py  # 개별 CFG Child Class 정의
│   │
│   └── baseline.py       # 모델 백본 빌더
│   └── eval.py           # 모델 평가 엔트리 포인트 
│   └── train.py          # 학습 루틴 엔트리 포인트
│   └── utils.py          # utils
└── README.md
```

---


### 학습 파이프라인

```
train.py    (학습 엔트리 포인트)
  └── loaders.py  (데이터셋 로딩)
        ↓
      baseline.py (백본 모델 빌드)
        ↓
      model.py    (커스텀 모델 정의)
        ↓
      학습 루프 수행
```

* `src/train.py`
  end-to-end 학습 루틴의 엔트리 포인트.
  "CFG 설정 → 데이터셋 로드 → (baseline/context) 모델 백본 빌드 → 커스텀 모델 학습"으로 이어지는 상위 레벨 학습 파이프라인 담당.

* `data/loaders.py`
  데이터 셋의 소스가 로컬이든 허깅페이스든 상관없이 사용할 수 있는 형태로 가공해서 로드한다. 현재 지원 포맷은 xlsx, json, arrow 이다.

* `src/baseline.py`
  "HF 백본 + 전처리/콜레이터 팩토리"에만 집중.

* `src/model.py`

### 평가 파이프라인

```
eval.py     (평가 엔트리 포인트)
  └── loaders.py  (데이터셋 로딩)
        ↓
      모델 로드
        ↓
      평가 수행
```

* `src/eval.py`
  학습된 모델(또는 베이스라인 모델)을 로드하고, 디바이스 및 데이터 타입 설정을 적용한다.
  필요한 경우 타깃 언어 토큰 강제 입력을 적용하며, 테스트 세트에 대한 번역 생성, SacreBLEU 기반 BLEU 계산, 생성 결과·레퍼런스 저장까지 수행한다.

* `src/configs/_config.py`
  데이터 경로, 번역 방향, 모델 이름, 최대 길이, 언어 코드, 학습 하이퍼파라미터 등 전체 설정을 중앙에서 관리한다. 이 값들은 모든 데이터셋에 대한 학습/평가 스크립트에서 공통적으로 사용된다.

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

- `lemonmit/korean_english_parallel_wiki_augmented_v1`: This dataset contains a large number of Korean-English parallel sentences extracted from Wikipedia. It was created by augmenting the original English Wikipedia dataset with machine-translated Korean sentences. The dataset is designed for training and evaluating machine translation models, especially those focusing on English-to-Korean and Korean-to-English translation.