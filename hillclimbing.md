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

## PaddleOCR saved time after warmup but did not improve claim matching

### Hypothesis

Using PaddleOCR instead of Large OCR will improve claim matching and cut the total processing time on August posts outside the first benchmark.

### Decision

Keep Large. PaddleOCR cut total time by 11% after warmup. Starting and warming up the Paddle worker erased nearly all of that saving. The fact-checks for all three posts that crossed the 0.70 cutoff were about different claims.

### Evidence

We tested 60 August posts, one per perceptual hash. We left out the earlier benchmark images and images with the same hashes. Both runs used Large captions and the same cosine matcher against 17,571 cached claims from Komdigi and Google Fact Check.

| Measure | Large OCR + caption | PaddleOCR + Large caption |
| --- | ---: | ---: |
| OCR after warmup | 261.06 s | 226.45 s |
| OCR + caption after warmup | 336.46 s | 299.21 s |
| OCR + caption with Paddle worker startup and warmup | 336.46 s | 335.58 s |
| Top cosine ≥ 0.70 | 14/60 | 15/60 |
| Top cosine ≥ 0.85 | 0/60 | 0/60 |

Neither column includes loading or warming up Florence. The top fact-check URL changed for 14 posts. Two posts moved above 0.70 and one fell below it. We checked those three posts by hand. That review does not estimate precision or recall.

### Suggested next steps

Improve fact-check retrieval and verify that posts make the same claim as their candidate fact-checks. Then retest how OCR affects claim matching.

## The assertion check rejected all 24 non-assertions but missed a valid paraphrase

### Hypothesis

An entailment check after cosine retrieval will reject corrections, questions, and related statements without rejecting valid paraphrases.

### Decision

Do not add the verifier to production. It rejected all 24 test examples that did not assert the claim, but also rejected a valid paraphrase.

### Evidence

We tested 32 written examples for eight August DPR claims and 60 real posts from the OCR trial. We used the pipeline's multilingual mDeBERTa natural language inference model and required an entailment score of at least 0.80. The model checks whether the post makes the claim. The published fact-check supplies the verdict.

| Measure | Cosine ≥ 0.70 | Cosine ≥ 0.85 | Cosine ≥ 0.70 + entailment ≥ 0.80 |
| --- | ---: | ---: | ---: |
| Assertions linked to the intended claim | 5/8 | 1/8 | 4/8 |
| Non-assertions accepted | 16/24 | 3/24 | 0/24 |
| Real posts flagged | 14/60 | 0/60 | 0/60 |

Three valid assertions did not reach the verifier with the right candidate. The verifier scored a fourth, about the Soekarno-Hatta airport demonstration, at 0.65. We did not label all the real posts.

### Suggested next steps

Try checking more than one candidate per post. Then tune the cutoff on pairs of real posts and claims that reviewers label independently.

## Checking three candidates found one more valid match and one false match

### Hypothesis

Checking the first three candidates instead of only the first will find valid matches without adding false ones.

### Decision

Use three candidates for evaluation only. The extra valid match came with an unrelated fact-check on a real post.

### Evidence

We reused the written examples and 60 real posts and kept the entailment cutoff at 0.80. We picked the first candidate that met the cutoff. The table shows results at a cosine cutoff of 0.70.

| Candidates | Intended claims eligible | Correct assertions accepted | Non-assertions accepted | Real posts flagged | Pairs scored |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5/8 | 4/8 | 0/24 | 0/60 | 36 |
| 3 | 6/8 | 5/8 | 0/24 | 1/60 | 55 |
| 5 | 6/8 | 5/8 | 0/24 | 1/60 | 60 |
| 10 | 6/8 | 5/8 | 0/24 | 1/60 | 66 |

Checking three candidates found the asset-forfeiture claim in second place. It also linked a post about a condolence visit to a fact-check about public celebrations. Lowering the cosine cutoff to 0.60 found no more valid matches.

### Suggested next steps

Compare one and three candidates on labeled real posts before changing the number of candidates or the cutoff in production.
