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
