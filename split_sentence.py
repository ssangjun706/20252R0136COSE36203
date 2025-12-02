import argparse
import json
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DELIMITERS = [". ", ".\n\n"]
SPLIT_PATTERN = re.compile("|".join(map(re.escape, DELIMITERS)))


def load_data(dataset_name: str):
    dataset = load_dataset(dataset_name)
    return dataset["train"]


def get_processed_indices(output_file: Path) -> set[int]:
    if not output_file.exists():
        return set()

    processed = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    processed.add(data.get("idx", -1))
                except json.JSONDecodeError:
                    continue

    processed.discard(-1)
    return processed


def split_sentences(text: str) -> list[str]:
    parts = SPLIT_PATTERN.split(text)
    return [p.strip() + "." for p in parts if p.strip()]


def process_row(args: tuple) -> list[dict] | None:
    idx, english, korean, score = args

    eng_sentences = split_sentences(english)
    kor_sentences = split_sentences(korean)

    if len(eng_sentences) != len(kor_sentences):
        return None

    return [
        {
            "idx": idx,
            "sent_idx": sent_idx,
            "english": eng,
            "korean": kor,
            "score": score,
        }
        for sent_idx, (eng, kor) in enumerate(zip(eng_sentences, kor_sentences))
    ]


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "wiki_augmented_v1_sentences.jsonl"

    processed = get_processed_indices(output_file)
    dataset = load_data(args.dataset_name)

    tasks = [
        (idx, row["english"], row["korean"], row["score"])
        for idx, row in enumerate(dataset)
        if idx not in processed
    ]

    if not tasks:
        print("All rows already processed!")
        return

    num_workers = min(cpu_count(), args.workers)

    skipped = 0
    success = 0

    with open(output_file, "a", encoding="utf-8") as f:
        with Pool(num_workers) as pool:
            results = pool.imap(process_row, tasks, chunksize=100)

            for result in tqdm(results, total=len(tasks), desc="Processing"):
                if result is None:
                    skipped += 1
                else:
                    success += 1
                    for record in result:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nCompleted! Saved to: {output_file}")
    print(f"Total: {len(tasks)}, Success: {success}, Skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lemon-mint/korean_english_parallel_wiki_augmented_v1",
    )
    parser.add_argument("--output_dir", type=str, default="data/wiki_augmented_v1")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    args = parser.parse_args()

    main(args)
