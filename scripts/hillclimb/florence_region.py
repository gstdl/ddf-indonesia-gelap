"""Run Base region-aware OCR, or compare cached region and plain results."""

import argparse

from scripts.hillclimb.florence import add_benchmark_args, add_score_args, run_benchmark, score_results


def describe_region(source, task):
    import torch

    from pipeline.layer1_prepare import image

    processor, model = image.florence()
    inputs = processor(text=task, images=source.convert("RGB"), return_tensors="pt").to(image.DEVICE)
    with torch.inference_mode():
        ids = model.generate(
            **inputs, max_new_tokens=image.FLORENCE_MAX_NEW_TOKENS,
            num_beams=image.FLORENCE_NUM_BEAMS, do_sample=False,
        )
    answer = processor.batch_decode(ids, skip_special_tokens=False)[0]
    regions = processor.post_process_generation(answer, task=task, image_size=source.size)[task]
    if len(regions["labels"]) != len(regions["quad_boxes"]):
        raise ValueError("Florence returned different numbers of labels and regions")
    return "\n".join(text for label in regions["labels"] if (text := label.strip()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_benchmark_args(commands.add_parser("run"))
    add_score_args(commands.add_parser("score"))
    args = parser.parse_args()
    if args.command == "run":
        run_benchmark(
            "base", args.candidates, args.data_root, args.output,
            ocr_task="<OCR_WITH_REGION>", describe_region=describe_region,
        )
    else:
        score_results(args.results, args.phrases)


if __name__ == "__main__":
    main()
