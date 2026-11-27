# Hillclimbing log

Baseline: `0f7a92e`, used for the 2,000-post sample run. The [README](README.md) records the shift from February–March #IndonesiaGelap to the August DPR protest wave, and the proposed spam, hoax, AI-text, and Grok evidence rules. Those research decisions did not change the baseline pipeline. See the [progress deck](notebooks/progress_2026-11-26.py) for the earlier scope and validation plan.

Run the linked experiments from the repository root using `python -m scripts.hillclimb.<module> --help` for arguments. The original Brandwatch sample, review labels, media, and model outputs are local inputs, not part of the repository. Exact reproduction requires the original data snapshot and selections.

## Florence-2 Base vs. Large: faster, but not a drop-in replacement

### Hypothesis

Florence-2 Base will cut image OCR and caption time without losing text needed for claim matching.

### Decision

Keep Large in the pipeline. Base nearly halved inference time but joined words across lines and missed more dense-image phrases.

### Evidence

[Benchmark and scorer](scripts/hillclimb/florence.py): 30 August images on Apple M4 Pro (MPS), three beams, 1,024-token cap, excluding load and warmup. Use the local candidate CSV (`id,media_path`) and reference phrases (JSON keyed by candidate ID).

| Measure | Base | Large |
| --- | ---: | ---: |
| OCR + caption total | 106.3 s | 209.8 s |
| OCR + caption per image | 3.54 s | 6.99 s |
| Phrase checks, ignoring punctuation and whitespace | 50/57 | 49/57 |
| Dense-image phrase checks | 15/20 | 18/20 |
| OCR outputs with line breaks | 0/30 | 25/30 |

These are selected phrase checks, not transcription accuracy. Neither model consistently produced better captions in the qualitative review.

### Suggested next steps

Test Base with region-aware OCR and score whole-word recovery so joined words count as misses.

## Region-aware Base restores line breaks but misses more dense text than Large

### Hypothesis

Joining Base’s region labels with newlines will restore word boundaries without losing its speed advantage over Large.

### Decision

Keep plain Large. Region-aware Base kept most line breaks and was faster, but recovered six fewer dense-image phrases.

### Evidence

[Region OCR runner](scripts/hillclimb/florence_region.py): same 30 images and inference settings, joining nonempty `<OCR_WITH_REGION>` labels in model order. Captions were unchanged. The whole-word scorer ignores case and punctuation.

| Measure | Base plain | Base regions | Large plain |
| --- | ---: | ---: | ---: |
| OCR + caption per image | 3.54 s | 4.91 s | 6.99 s |
| OCR outputs with line breaks | 0/30 | 25/30 | 25/30 |
| Phrases recovered, ignoring word boundaries | 50/57 | 47/57 | 49/57 |
| Whole-word phrases recovered | 10/57 | 45/57 | 49/57 |
| Dense whole-word phrases recovered | 3/20 | 12/20 | 18/20 |

### Suggested next steps

Benchmark a dedicated OCR engine on the same images, with particular attention to dense claim text and numbers.

## PaddleOCR recovered more claim text and was faster than plain Large

### Hypothesis

PaddleOCR will recover more whole-word phrases and process images faster than Florence-2 Large OCR.

### Decision

Advance PaddleOCR with Large captioning to a held-out trial. Do not add it as a production dependency.

### Evidence

[PaddleOCR runner](scripts/hillclimb/paddle.py): PaddleOCR 3.7.0 / PaddlePaddle 3.3.1, PP-OCRv5 mobile detection and Latin-script recognition, four CPU threads on M4 Pro, on the same 30 images. Timings exclude load and warmup.

| Measure | PaddleOCR | Large |
| --- | ---: | ---: |
| OCR per image | 4.99 s | 5.85 s |
| OCR + Large caption per image | 6.14 s (estimated) | 6.99 s (measured) |
| Whole-word phrases recovered | 56/57 | 49/57 |
| Dense whole-word phrases recovered | 20/20 | 18/20 |

PaddleOCR recovered both rupiah amounts in a dense pay chart. The combined estimate adds the measured Large caption time; the two engines did not run together in this benchmark.

### Suggested next steps

On held-out posts, measure end-to-end wall time including startup and inspect claim matches that change.

## Held-out trial: PaddleOCR was faster after warmup, but did not improve claim matches

### Hypothesis

Replacing Large OCR with PaddleOCR will improve claim matching and reduce total processing time on unseen August posts.

### Decision

Keep Large. PaddleOCR saved 11% after warmup but almost none with worker startup included. None of the score-cutoff changes linked the same claim.

### Evidence

[Held-out OCR trial](scripts/hillclimb/ocr_trial.py): 60 August posts, one per perceptual hash, excluding the benchmark and its exact hash matches. The local sample JSON is a list of `post_id`/`media_path` objects; the warmup JSON is a media path string. Run the `large` and `hybrid` arms separately, using `--worker-python` for a separate PaddleOCR environment, then score their outputs against the original data root.

Both arms used Large captions and the current cosine matcher over 17,571 cached Komdigi and Google fact-check claims.

| Measure | Large OCR + caption | PaddleOCR + Large caption |
| --- | ---: | ---: |
| OCR after warmup | 261.06 s | 226.45 s |
| OCR + caption after warmup | 336.46 s | 299.21 s |
| OCR + caption with Paddle worker startup and warmup | 336.46 s | 335.58 s |
| Top cosine ≥ 0.70 | 14/60 | 15/60 |
| Top cosine ≥ 0.85 | 0/60 | 0/60 |

Florence load and warmup are excluded from both columns. The top URL changed for 14 posts; two crossed up and one down at 0.70. Manual review found all three links were mismatches, not a precision or recall estimate.

### Suggested next steps

Improve fact-check retrieval and verify whether the post asserts the candidate claim before retesting OCR downstream.
