"""Benchmark PaddleOCR and compare it with cached Florence Large results."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from scripts.hillclimb.florence import read_results, score_results


def warm_paddle(data_root, warmup_path):
    from paddleocr import PaddleOCR

    start = perf_counter()
    ocr = PaddleOCR(
        device="cpu", cpu_threads=4,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
    )
    print(f"Paddle load: {perf_counter() - start:.2f}s", flush=True)

    def predict(path):
        result, = ocr.predict(str(data_root / path))
        return "\n".join(text for raw in result["rec_texts"] if (text := raw.strip()))

    predict(warmup_path)
    return predict


def run_benchmark(large, data_root, output):
    baseline = list(read_results(large).values())
    predict = warm_paddle(data_root, baseline[0]["path"])
    with output.open("w", buffering=1, encoding="utf-8") as destination:
        for row in baseline:
            start = perf_counter()
            text = predict(row["path"])
            elapsed = perf_counter() - start
            destination.write(json.dumps({**row, "ocr": text, "ocr_s": elapsed}, ensure_ascii=False) + "\n")


def run_held_out_worker(sample, warmup, data_root, output):
    rows = json.loads(sample.read_text(encoding="utf-8"))
    predict = warm_paddle(data_root, json.loads(warmup.read_text(encoding="utf-8")))
    start = perf_counter()
    texts = [predict(row["media_path"]) for row in rows]
    result = {"texts": texts, "ocr_s": perf_counter() - start}
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def worker_process(python, *args):
    start = perf_counter()
    subprocess.run([python or sys.executable, "-m", "scripts.hillclimb.paddle", *args], check=True)
    return perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="replace OCR only in the fixed Large cohort JSONL")
    run.add_argument("--large", type=Path, required=True)
    run.add_argument("--worker-python", type=Path)
    worker = commands.add_parser("worker", help="run OCR only on the held-out sample")
    worker.add_argument("--sample", type=Path, required=True)
    worker.add_argument("--warmup", type=Path, required=True)
    for command in (run, worker):
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="compare cached Large and Paddle JSONL")
    score.add_argument("--large", type=Path, required=True)
    score.add_argument("--paddle", type=Path, required=True)
    score.add_argument("--phrases", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.worker_python is not None:
            worker_process(args.worker_python, "run", "--large", args.large, "--data-root", args.data_root, "--output", args.output)
        else:
            run_benchmark(args.large, args.data_root, args.output)
    elif args.command == "worker":
        run_held_out_worker(args.sample, args.warmup, args.data_root, args.output)
    else:
        score_results((args.large, args.paddle), args.phrases)


if __name__ == "__main__":
    main()
