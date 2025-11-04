# COSE362 Project - Korean-English Translation

This project implements a Korean-English translation model using the AIHub News Translation Dataset.

## Project Structure

```
cose362/
├── data/
│   └── aihub_news/        # AIHub News Translation Dataset
├── prototype/
│   └── model.py           # Model prototyping and experiments
├── src/
│   ├── baseline.py        # Baseline model implementation
│   ├── config.py          # Configuration settings
│   ├── eval.py            # Model evaluation scripts
│   ├── train.py          # Training scripts
│   └── utils.py          # Utility functions
└── test.py               # Test scripts
```

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
python src/train.py
```

### Evaluation

To evaluate the model:

```bash
python src/eval.py
```

### Testing

To run tests:

```bash
python test.py
```

## Data

The project uses the AIHub Korean-English News Translation Dataset. The data is organized in the following structure:
- `data/aihub_news/`: Contains various Excel files with different categories of news translations
  - 구어체(1).xlsx, 구어체(2).xlsx: Conversational style texts
  - 대화체.xlsx: Dialogue style texts
  - 문어체_뉴스(1-4).xlsx: Written style news texts
  - 문어체_한국문화.xlsx: Korean culture related texts
  - 문어체_종료.xlsx: Miscellaneous texts
  - 문어체_지자체웹사이트.xlsx: Local government website texts