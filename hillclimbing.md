# Hillclimbing log

The 2,000-post sample run used commit `0f7a92e`. The [README](README.md) records the shift to the August DPR protests and proposes how to label spam and assess hoax claims, AI-generated text, and Grok replies. The [progress deck](notebooks/progress_2026-11-26.py) covers the earlier February and March #IndonesiaGelap scope and validation plan.

None of the experiments below changed the production pipeline.

## Florence-2 Base ran faster but missed more phrases in dense images

### Hypothesis

Florence-2 Base will read image text and write captions faster than Large without losing text needed to match claims to fact-checks.

### Decision

Keep Large in production. Base took about half as long, but ran words together across lines and missed more reference phrases in dense images.

### Evidence

We ran both models on 30 images from August posts using an Apple M4 Pro with MPS. Both used three beams and a 1,024-token limit. Timings exclude model loading and warmup.

| Measure | Base | Large |
| --- | ---: | ---: |
| OCR + caption total | 106.3 s | 209.8 s |
| OCR + caption per image | 3.54 s | 6.99 s |
| Phrase checks, ignoring punctuation and whitespace | 50/57 | 49/57 |
| Dense-image phrase checks | 15/20 | 18/20 |
| OCR outputs with line breaks | 0/30 | 25/30 |

The counts cover selected phrases, not the full transcripts. In a manual review, neither model consistently wrote better captions.

### Suggested next steps

Test region-aware OCR with Base. Require whole words when scoring so words that run together count as misses.
