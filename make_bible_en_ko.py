# scripts/make_bible_en_ko.py
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import subprocess

def get_project_root() -> Path:
    """
    __file__ 이 없는 노트북 환경에서도 동작하도록:
    - 스크립트로 실행되면: COSE362-Project 루트 기준
    - 노트북에서 복붙해서 실행하면: 현재 working dir 기준
    """
    try:
        here = Path(__file__).resolve()
        # .../COSE362-Project/scripts/make_bible_en_ko.py 라고 가정
        return here.parents[1]
    except NameError:
        # __file__ 없는 경우 (Colab에서 셀로 돌린 경우)
        return Path.cwd()

def ensure_repo_cloned(base_dir: Path) -> Path:
    """
    base_dir 아래에 korean-parallel-corpora 가 없으면 git clone.
    """
    repo_dir = base_dir / "external" / "korean-parallel-corpora"
    if repo_dir.exists():
        print(f"[+] Found repo: {repo_dir}")
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/jungyeul/korean-parallel-corpora.git"
    print(f"[+] Cloning {url} -> {repo_dir}")
    subprocess.run(["git", "clone", url, str(repo_dir)], check=True)
    return repo_dir

def detect_bible_files(bible_dir: Path):
    """
    bible/ 디렉토리 안에서 영어/한국어 파일 자동으로 추측.
    - 가장 먼저 다음 순서대로 존재하는지 확인:
        1) bible.en / bible.ko
        2) bible.en.ko.en / bible.en.ko.ko (예시)
    - 필요하면 여기만 수정해서 파일 이름 맞춰주면 됨.
    """
    candidates = [
        (bible_dir / "bible.en", bible_dir / "bible.ko"),
        (bible_dir / "bible.en.ko.en", bible_dir / "bible.en.ko.ko"),
    ]

    for en_path, ko_path in candidates:
        if en_path.exists() and ko_path.exists():
            print(f"[+] Using files:\n    EN={en_path}\n    KO={ko_path}")
            return en_path, ko_path

    # 못 찾으면 그냥 listing 보여주고 에러
    print("[!] Could not auto-detect bible files.")
    print(f"[!] Please check files under: {bible_dir}")
    for p in sorted(bible_dir.glob("*")):
        print("   -", p.name)
    raise FileNotFoundError("Please update detect_bible_files() with correct filenames.")

def read_parallel(en_path: Path, ko_path: Path):
    """
    영어/한국어 라인 수가 동일한지 확인하고 리스트로 반환.
    """
    with open(en_path, encoding="utf-8") as f_en:
        en_lines = [l.strip() for l in f_en]
    with open(ko_path, encoding="utf-8") as f_ko:
        ko_lines = [l.strip() for l in f_ko]

    if len(en_lines) != len(ko_lines):
        print(f"[!] WARNING: length mismatch: en={len(en_lines)} ko={len(ko_lines)}")
        min_len = min(len(en_lines), len(ko_lines))
        print(f"[!] Truncating to {min_len} lines.")
        en_lines = en_lines[:min_len]
        ko_lines = ko_lines[:min_len]

    # 완전히 빈 줄은 그냥 같이 버리기 (노이즈 제거용)
    src, tgt = [], []
    for e, k in zip(en_lines, ko_lines):
        e = e.strip()
        k = k.strip()
        if not e or not k:
            continue
        src.append(e)
        tgt.append(k)

    print(f"[+] Final pairs: {len(src)}")
    return src, tgt

def save_as_test_src_tgt(project_root: Path, src, tgt):
    """
    COSE362-Project 루트 기준 data/bible_en_ko/test.src, test.tgt 로 저장.
    loaders._load_local_txt()에서 그대로 읽을 수 있는 형태.
    """
    out_dir = project_root / "data" / "bible_en_ko"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = out_dir / "test.src"
    tgt_path = out_dir / "test.tgt"

    with open(src_path, "w", encoding="utf-8") as f:
        for line in src:
            f.write(line.replace("\n", " ") + "\n")

    with open(tgt_path, "w", encoding="utf-8") as f:
        for line in tgt:
            f.write(line.replace("\n", " ") + "\n")

    print(f"[+] Saved test.src: {src_path}")
    print(f"[+] Saved test.tgt: {tgt_path}")
    return out_dir

def main():
    project_root = get_project_root()
    print(f"[+] PROJECT_ROOT = {project_root}")

    # 1) korean-parallel-corpora clone
    repo_dir = ensure_repo_cloned(project_root)

    # 2) bible 디렉토리
    bible_dir = repo_dir / "bible"
    if not bible_dir.is_dir():
        raise FileNotFoundError(f"bible directory not found: {bible_dir}")

    # 3) 영어/한국어 파일 경로 추측
    en_path, ko_path = detect_bible_files(bible_dir)

    # 4) 평행 말뭉치 읽기
    src, tgt = read_parallel(en_path, ko_path)

    # 5) data/bible_en_ko/test.src, test.tgt 로 저장
    out_dir = save_as_test_src_tgt(project_root, src, tgt)
    print(f"[+] Done. Dataset dir = {out_dir}")

if __name__ == "__main__":
    main()
