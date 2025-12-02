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


def get_processed_indices(output_dir: Path, base_name: str) -> set[int]:
    processed = set()
    for file in output_dir.glob(f"{base_name}_*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
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


def get_current_file_info(output_dir: Path, base_name: str) -> tuple[int, int]:
    """현재 파일 번호와 크기 반환"""
    existing_files = sorted(output_dir.glob(f"{base_name}_*.jsonl"))
    if not existing_files:
        return 0, 0

    last_file = existing_files[-1]
    file_num = int(last_file.stem.split("_")[-1])
    file_size = last_file.stat().st_size
    return file_num, file_size


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
    base_name = "wiki_augmented_v1"
    max_file_size = args.max_size_mb * 1000 * 1000  # MB to bytes

    processed = get_processed_indices(output_dir, base_name)
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

    # 현재 파일 정보
    file_num, current_size = get_current_file_info(output_dir, base_name)
    output_file = output_dir / f"{base_name}_{file_num:04d}.jsonl"

    skipped = 0
    success = 0
    f = open(output_file, "a", encoding="utf-8")

    try:
        with Pool(num_workers) as pool:
            results = pool.imap(process_row, tasks, chunksize=100)

            for result in tqdm(results, total=len(tasks), desc="Processing"):
                if result is None:
                    skipped += 1
                else:
                    success += 1
                    for record in result:
                        line = json.dumps(record, ensure_ascii=False) + "\n"
                        current_size += len(line.encode("utf-8"))
                        f.write(line)

                        # 파일 크기 초과 시 새 파일로 전환
                        if current_size >= max_file_size:
                            f.close()
                            file_num += 1
                            output_file = (
                                output_dir / f"{base_name}_{file_num:04d}.jsonl"
                            )
                            f = open(output_file, "w", encoding="utf-8")
                            current_size = 0
    finally:
        f.close()

    print(f"\nCompleted! Saved to: {output_dir}/{base_name}_*.jsonl")
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
    parser.add_argument(
        "--max_size_mb", type=int, default=99, help="Max file size in MB"
    )
    args = parser.parse_args()

    main(args)
