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

## Region-aware Base restored line breaks but missed more dense text than Large

### Hypothesis

Putting each region label on a new line will restore word boundaries while keeping Base faster than Large.

### Decision

Keep plain Large. Region-aware Base restored line breaks and ran faster, but recovered six fewer phrases from dense images than Large.

### Evidence

We used the same 30 images and model settings. We put each nonempty `<OCR_WITH_REGION>` label on its own line in the order the model returned it. We kept the captions unchanged. Whole-word scoring ignores case and punctuation.

| Measure | Base plain | Base regions | Large plain |
| --- | ---: | ---: | ---: |
| OCR + caption per image | 3.54 s | 4.91 s | 6.99 s |
| OCR outputs with line breaks | 0/30 | 25/30 | 25/30 |
| Phrases recovered, ignoring word boundaries | 50/57 | 47/57 | 49/57 |
| Whole-word phrases recovered | 10/57 | 45/57 | 49/57 |
| Dense whole-word phrases recovered | 3/20 | 12/20 | 18/20 |

### Suggested next steps

Try a dedicated OCR model on the same images. Check whether it reads dense text and numbers more accurately.

## PaddleOCR recovered more phrases and ran faster than Large OCR

### Hypothesis

PaddleOCR will recover more whole-word phrases and read the images faster than Florence-2 Large OCR.

### Decision

Test PaddleOCR with Large captions on a separate sample. Do not add PaddleOCR to the production pipeline.

### Evidence

We tested the same 30 images with PaddleOCR 3.7.0 and PaddlePaddle 3.3.1. We used PP-OCRv5 mobile text detection and Latin-script recognition with four CPU threads on an Apple M4 Pro. Timings exclude loading and warmup.

| Measure | PaddleOCR | Large |
| --- | ---: | ---: |
| OCR per image | 4.99 s | 5.85 s |
| OCR + Large caption per image | 6.14 s estimated | 6.99 s measured |
| Whole-word phrases recovered | 56/57 | 49/57 |
| Dense whole-word phrases recovered | 20/20 | 18/20 |

PaddleOCR read both rupiah amounts in a dense pay chart. We added the measured Large caption time to estimate the combined time. We did not run the two engines together in this test.

### Suggested next steps

Measure the total time including startup on a separate sample. Review posts whose best fact-check changes.
