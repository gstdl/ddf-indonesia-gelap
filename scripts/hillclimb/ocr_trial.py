"""Time held-out OCR and captioning arms, or compare cached outputs with the frozen matcher corpus."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from scripts.hillclimb.florence import synchronize
from scripts.hillclimb.paddle import worker_process

SAMPLE_SIZE = 60


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_keys(rows):
    keys = [(str(row["post_id"]), row["media_path"]) for row in rows]
    if len(keys) != SAMPLE_SIZE or len({key[0] for key in keys}) != SAMPLE_SIZE:
        raise ValueError(f"Expected the ordered held-out sample of {SAMPLE_SIZE} unique posts")
    return keys


def run_trial(args):
    from PIL import Image

    from pipeline.layer1_prepare import image

    sample = read_json(args.sample)
    keys = sample_keys(sample)
    warmup_path = read_json(args.warmup)
    paddle_process_s = None
    if args.arm == "hybrid":
        paddle_process_s = worker_process(
            args.worker_python, "worker", "--sample", args.sample, "--warmup", args.warmup,
            "--data-root", args.data_root, "--output", args.paddle_output,
        )
        paddle = read_json(args.paddle_output)
        ocr, ocr_s = paddle["texts"], paddle["ocr_s"]

    image.florence()
    with Image.open(args.data_root / warmup_path) as warmup:
        if args.arm == "large":
            image.describe(warmup, "<OCR>")
        image.describe(warmup, "<CAPTION>")
    synchronize(image.DEVICE)

    with ExitStack() as stack:
        opened = [stack.enter_context(Image.open(args.data_root / path)) for _, path in keys]
        for source in opened:
            source.load()

        def timed_describe(task):
            start = perf_counter()
            texts = [image.describe(source, task) for source in opened]
            synchronize(image.DEVICE)
            return texts, perf_counter() - start

        if args.arm == "large":
            ocr, ocr_s = timed_describe("<OCR>")
        captions, caption_s = timed_describe("<CAPTION>")

    result = {
        "ocr_s": ocr_s,
        "caption_s": caption_s,
        "paddle_process_s": paddle_process_s,
        "rows": [
            {"post_id": post_id, "media_path": media_path, "ocr": text, "caption": caption}
            for (post_id, media_path), text, caption in zip(keys, ocr, captions, strict=True)
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(timing(result), indent=2))


def timing(result):
    warm_ocr = result["ocr_s"]
    caption = result["caption_s"]
    worker = result["paddle_process_s"]
    return {
        "ocr_s": warm_ocr,
        "caption_s": caption,
        "paddle_process_s": worker,
        "warm_combined_s": warm_ocr + caption,
        "combined_including_paddle_process_s": (warm_ocr if worker is None else worker) + caption,
    }


def score_trial(args):
    import numpy as np
    import pandas as pd

    from pipeline.layer1_prepare.document import build_document
    from pipeline.layer1_prepare.image import join_texts
    from pipeline.layer2_label.track1_hoax.claimed_false import CLAIMED_FALSE_SIMILARITY, best_match, false_claims
    from pipeline.layer2_label.track1_hoax.similar_to_known_hoax import KNOWN_HOAX_SIMILARITY

    large, hybrid = read_json(args.large), read_json(args.hybrid)
    keys = sample_keys(large["rows"])
    if keys != sample_keys(hybrid["rows"]):
        raise ValueError("Trial results must contain the same held-out posts and media in the same order")
    ids = [post_id for post_id, _ in keys]
    posts = pd.read_parquet(args.data_root / "layer1_prepare/image.parquet", columns=["post_id", "text"])
    posts["post_id"] = posts["post_id"].astype(str)
    posts = posts[posts["post_id"].isin(ids)].set_index("post_id")
    if not posts.index.is_unique:
        raise ValueError("Held-out post IDs must be unique in the image parquet")
    posts = posts.loc[ids].reset_index()
    source = args.data_root / "layer2_label/track1_hoax"
    claims = false_claims(tuple(pd.read_parquet(source / f"{name}.parquet") for name in ("komdigi", "google_fact_check")))

    documents = [
        build_document(posts.assign(media_text=[join_texts(row["ocr"], row["caption"]) for row in trial["rows"]]))
        for trial in (large, hybrid)
    ]
    scores, urls = best_match(pd.concat(documents, ignore_index=True), claims)
    large_scores, hybrid_scores = np.split(scores, 2)
    large_urls, hybrid_urls = np.split(urls, 2)
    cutoffs = {}
    for cutoff in (KNOWN_HOAX_SIMILARITY, CLAIMED_FALSE_SIMILARITY):
        left, right = large_scores >= cutoff, hybrid_scores >= cutoff
        cutoffs[f"{cutoff:.2f}"] = {
            "large": int(left.sum()), "hybrid": int(right.sum()),
            "crossed_up": int((~left & right).sum()), "crossed_down": int((left & ~right).sum()),
        }
    crossed = np.flatnonzero((large_scores >= KNOWN_HOAX_SIMILARITY) != (hybrid_scores >= KNOWN_HOAX_SIMILARITY))
    print(json.dumps({
        "posts": len(ids),
        "claims": len(claims),
        "identical_captions": sum(a["caption"] == b["caption"] for a, b in zip(large["rows"], hybrid["rows"], strict=True)),
        "changed_top_urls": int(np.count_nonzero(large_urls != hybrid_urls)),
        "median_absolute_top_cosine_change": float(np.median(np.abs(hybrid_scores - large_scores))),
        "cutoffs": cutoffs,
        "changed_loose_cutoff_matches": [
            {"post_id": ids[i], "large_url": large_urls[i], "hybrid_url": hybrid_urls[i]}
            for i in crossed
        ],
        "timing": {"large": timing(large), "hybrid": timing(hybrid)},
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="time one arm in a fresh process")
    run.add_argument("arm", choices=("large", "hybrid"))
    for field in ("sample", "warmup", "data-root", "output"):
        run.add_argument(f"--{field}", type=Path, required=True)
    run.add_argument("--worker-python", type=Path)
    run.add_argument("--paddle-output", type=Path, help="local OCR-only result, required for hybrid")
    score = commands.add_parser("score", help="compare cached results without running OCR or captioning")
    for field in ("large", "hybrid", "data-root"):
        score.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.arm == "hybrid" and args.paddle_output is None:
            run.error("hybrid requires --paddle-output")
        run_trial(args)
    else:
        score_trial(args)


if __name__ == "__main__":
    main()
