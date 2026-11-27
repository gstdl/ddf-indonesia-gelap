"""Run or rescore the paired plain Florence-2 OCR and caption benchmark."""

import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path
from time import perf_counter

GROUPS = {
    "news cards": (1, 6, 8, 15, 17, 34, 75, 81),
    "posters and memes": (2, 4, 12, 41, 42, 56, 67, 80),
    "screenshots and documents": (23, 24, 26, 37, 49, 54, 69, 83),
    "photos and low-text frames": (9, 10, 13, 25, 27, 35),
}
CANDIDATES = {candidate: group for group, ids in GROUPS.items() for candidate in ids}
REFERENCE_COUNTS = dict.fromkeys((1, 6, 8, 75, 81, 12, 23, 24, 37, 49, 54, 83), 3)
REFERENCE_COUNTS.update(dict.fromkeys((15, 17, 34, 2, 4, 41, 42, 56, 80, 26), 2))
REFERENCE_COUNTS[67] = 1
BASE_REVISION = "00921df66db728a9ceb750f5eca43e5c203a2051"


def synchronize(device):
    import torch

    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def run_benchmark(model, candidates, data_root, output, *, ocr_task="<OCR>", describe_region=None):
    from PIL import Image

    from pipeline.layer1_prepare import image

    original_model, original_revision = image.FLORENCE_MODEL, image.FLORENCE_REVISION
    with candidates.open(newline="", encoding="utf-8") as source:
        paths = {int(row["id"]): row["media_path"] for row in csv.DictReader(source)}
    if missing := (CANDIDATES.keys() | {88}) - paths.keys():
        raise ValueError(f"Candidate CSV is missing benchmark ordinals: {sorted(missing)}")
    if model == "base":
        image.FLORENCE_MODEL = "florence-community/Florence-2-base"
        image.FLORENCE_REVISION = BASE_REVISION
    image.florence.cache_clear()
    try:
        start = perf_counter()
        image.florence()
        synchronize(image.DEVICE)
        print(f"{image.FLORENCE_MODEL}@{image.FLORENCE_REVISION} on {image.DEVICE}; load: {perf_counter() - start:.2f}s")
        ocr = describe_region or image.describe
        tasks = (("ocr", ocr_task, ocr), ("caption", "<CAPTION>", image.describe))
        with Image.open(data_root / paths[88]) as warmup:
            for _, task, describe in tasks:
                describe(warmup, task)
            synchronize(image.DEVICE)
        with output.open("w", buffering=1, encoding="utf-8") as destination:
            for candidate, group in CANDIDATES.items():
                row = {"id": candidate, "group": group, "path": paths[candidate]}
                with Image.open(data_root / row["path"]) as source:
                    source.load()
                    for key, task, describe in tasks:
                        start = perf_counter()
                        row[key] = describe(source, task)
                        synchronize(image.DEVICE)
                        row[f"{key}_s"] = perf_counter() - start
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        image.florence.cache_clear()
        image.FLORENCE_MODEL, image.FLORENCE_REVISION = original_model, original_revision


def read_results(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(CANDIDATES) or {row["id"]: row["group"] for row in rows} != CANDIDATES:
        raise ValueError(f"{path}: expected the fixed 30 benchmark candidates and cohorts")
    return {row["id"]: row for row in rows}


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def score_results(result_paths, phrases_path):
    references = {int(key): value for key, value in json.loads(phrases_path.read_text(encoding="utf-8")).items()}
    if {key: len(value) for key, value in references.items()} != REFERENCE_COUNTS:
        raise ValueError("Expected the original 57 reference phrases and candidate ordinals")
    reports = {}
    sample_paths = None
    for path in result_paths:
        rows = read_results(path)
        selected_paths = {candidate: row["path"] for candidate, row in rows.items()}
        if sample_paths is not None and sample_paths != selected_paths:
            raise ValueError("Result files must describe the same candidate media")
        sample_paths = selected_paths
        per_group = {group: {"compact": 0, "whole_words": 0, "references": 0} for group in GROUPS}
        for candidate, phrases in references.items():
            text = normalize(rows[candidate]["ocr"])
            compact = text.replace(" ", "")
            group = per_group[CANDIDATES[candidate]]
            for phrase in phrases:
                tokens = normalize(phrase)
                if not tokens:
                    raise ValueError("Reference phrases must contain at least one word token")
                group["compact"] += tokens.replace(" ", "") in compact
                group["whole_words"] += f" {tokens} " in f" {text} "
                group["references"] += 1
        times = {}
        for key in ("ocr", "caption", "combined"):
            seconds = [row["ocr_s"] + row["caption_s"] if key == "combined" else row[f"{key}_s"] for row in rows.values()]
            times[key] = {"total_s": sum(seconds), "mean_s": statistics.mean(seconds), "median_s": statistics.median(seconds)}
        reports[str(path)] = {
            "images": len(rows),
            "phrases": {key: sum(group[key] for group in per_group.values()) for key in ("compact", "whole_words", "references")},
            "cohorts": per_group,
            "multiline": sum("\n" in row["ocr"] for row in rows.values()),
            "empty": sum(not row["ocr"] for row in rows.values()),
            "timing": times,
        }
    print(json.dumps(reports, indent=2))


def add_benchmark_args(parser):
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def add_score_args(parser):
    parser.add_argument("--phrases", type=Path, required=True)
    parser.add_argument("results", type=Path, nargs="+")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one Florence model on the fixed sample")
    run.add_argument("model", choices=("base", "large"))
    add_benchmark_args(run)
    add_score_args(commands.add_parser("score", help="score cached benchmark JSONL results"))
    args = parser.parse_args()
    if args.command == "run":
        run_benchmark(args.model, args.candidates, args.data_root, args.output)
    else:
        score_results(args.results, args.phrases)


if __name__ == "__main__":
    main()
