# Hillclimbing log

Baseline: `0f7a92e`, used for the 2,000-post sample run. The [README](README.md) records the shift from February–March #IndonesiaGelap to the August DPR protest wave, and the proposed spam, hoax, AI-text, and Grok evidence rules. Those research decisions did not change the baseline pipeline. See the [progress deck](notebooks/progress_2026-11-26.py) for the earlier scope and validation plan.

This log records experimental results. None of these trials changed the production pipeline.

## Florence-2 Base vs. Large: faster, but not a drop-in replacement

### Hypothesis

Florence-2 Base will cut image OCR and caption time without losing text needed for claim matching.

### Decision

Keep Large in the pipeline. Base nearly halved inference time but joined words across lines and missed more dense-image phrases.

### Evidence

Compared both models on 30 August images on Apple M4 Pro (MPS), with three beams and a 1,024-token cap. Timings exclude loading and warmup.

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

Used the same 30 images and inference settings, joining nonempty `<OCR_WITH_REGION>` labels in model order. Captions were unchanged. Whole-word scoring ignores case and punctuation.

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

Used PaddleOCR 3.7.0 / PaddlePaddle 3.3.1, PP-OCRv5 mobile detection and Latin-script recognition, and four CPU threads on M4 Pro. Compared the same 30 images, excluding load and warmup.

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

Tested 60 August posts, one per perceptual hash, excluding the earlier benchmark images and their exact hash matches.

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

## An assertion check rejects topic matches but misses valid paraphrases

### Hypothesis

An entailment check after cosine retrieval will reject corrections, questions, and related statements while preserving valid paraphrases.

### Decision

Keep the verifier experimental. It rejected all controlled non-assertions but also rejected a valid paraphrase.

### Evidence

Tested 32 written examples across eight August DPR claims and the 60 real posts from the OCR trial. The verifier uses the pipeline’s multilingual mDeBERTa natural language inference model at a fixed 0.80 entailment cutoff. The published fact-check supplies the verdict; the model only checks what the post asserts.

| Measure | Cosine ≥ 0.70 | Cosine ≥ 0.85 | Cosine ≥ 0.70 + entailment ≥ 0.80 |
| --- | ---: | ---: | ---: |
| Assertions linked to the intended claim | 5/8 | 1/8 | 4/8 |
| Non-assertions accepted | 16/24 | 3/24 | 0/24 |
| Real posts flagged | 14/60 | 0/60 | 0/60 |

Three valid assertions did not reach the verifier with the right candidate; another, about the Soekarno-Hatta airport demonstration, scored 0.65 entailment. The real slice was not fully labeled.

### Suggested next steps

Try several retrieval candidates per post, then calibrate on independently labeled real post–claim pairs.

## Top-3 recovered one valid match and introduced one real mismatch

### Hypothesis

Verifying the first three candidates instead of the first one will recover valid claims missed by top-1 without adding false matches.

### Decision

Keep top-3 as an evaluation setting only. It recovered one valid assertion but also introduced an unrelated link on a real post.

### Evidence

Reused the controlled examples, 60 real posts, and 0.80 entailment cutoff. Selected the first passing candidate. At the 0.70 cosine floor:

| Candidates | Intended claims eligible | Correct assertions accepted | Non-assertions accepted | Real posts flagged | Pairs scored |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5/8 | 4/8 | 0/24 | 0/60 | 36 |
| 3 | 6/8 | 5/8 | 0/24 | 1/60 | 55 |
| 5 | 6/8 | 5/8 | 0/24 | 1/60 | 60 |
| 10 | 6/8 | 5/8 | 0/24 | 1/60 | 66 |

Top-3 recovered the asset-forfeiture claim at rank 2. Its mismatch linked a condolence visit to a fact-check about public celebrations. Lowering the floor to 0.60 recovered no more valid assertions.

### Suggested next steps

Compare top-1 and top-3 on labeled real posts before widening the shortlist or lowering the floor in production.

## On real posts, the verifier rejected mismatches but retrieval missed every known positive

### Hypothesis

The fixed entailment gate will reject unrelated checks on real posts while retaining known claim matches.

### Decision

Prioritize retrieval for the next experiment; leave the production matcher unchanged. The verifier rejected reviewed mismatches, but none of the three archived positives reached it.

### Evidence

Reviewed 24 X posts drawn from the high-similarity slice, after excluding the earlier OCR trial; none of their 50 eligible candidate pairs matched. Three separate social-media quotes from Komdigi reviews supplied known positives: [Sahroni’s household employee](https://www.komdigi.go.id/berita/berita-hoaks/detail/hoaks-art-ahmad-sahroni-luka-parah-akibat-massa-menyerang-kediaman-ahmad-sahroni), [the UN dissolving the DPR](https://www.komdigi.go.id/berita/berita-hoaks/detail/hoaks-pbb-gelar-sidang-darurat-resmi-bubarkan-dpr-indonesia), and [a legislator’s car](https://www.komdigi.go.id/berita/berita-hoaks/detail/hoaks-mobil-anggota-dpr-dirusak-pendemo).

| Variant | Incorrect links on reviewed X posts | Correct links on archived positives |
| --- | ---: | ---: |
| Top-1 cosine ≥ 0.70 | 24/24 | 0/3 |
| Top-1 cosine ≥ 0.85 | 1/24 | 0/3 |
| Top-1 cosine ≥ 0.70 + entailment ≥ 0.80 | 0/24 | 0/3 |
| Top-3 cosine ≥ 0.70 + entailment ≥ 0.80 | 0/24 | 0/3 |

Positive targets ranked 3rd, 75th, and 3rd, below the 0.70 floor in every case. The handpicked excerpts and high-similarity negative slice give diagnostic counts, not population precision or recall.

### Suggested next steps

Compare event-focused and lexical-plus-semantic retrieval. Measure whether the right check enters the shortlist before tuning entailment.

## August-focused search improved ranking, but the hard filter dropped a relevant check

### Hypothesis

An August DPR corpus filter and keyword retrieval alongside semantic search will surface archived positives earlier.

### Decision

Keep full-corpus hybrid as the experimental baseline. The hard filter improved rank but removed a known check.

### Evidence

Compared the full 17,571-claim cache with 73 claims published in August–September 2025 Jakarta time and matching a protest-related whole-word term. Keyword retrieval used word and bigram TF-IDF; hybrid combined it with semantic cosine via reciprocal rank fusion. No acceptance cutoff or assertion verifier was applied.

| Corpus and ranking | Archived at rank 1 | Archived in top 3 | Archived in top 10 | Controlled in top 10 |
| --- | ---: | ---: | ---: | ---: |
| Full · semantic | 0/3 | 2/3 | 2/3 | 8/8 |
| Full · keyword | 2/3 | 2/3 | 2/3 | 8/8 |
| Full · hybrid | 1/3 | 2/3 | 3/3 | 8/8 |
| Focused · semantic | 2/3 | 3/3 | 3/3 | 7/8 |
| Focused · keyword | 3/3 | 3/3 | 3/3 | 7/8 |
| Focused · hybrid | 3/3 | 3/3 | 3/3 | 7/8 |

The filter missed `berdemonstrasi` and excluded undated reviews; 410 of 486 cached Google rows had no publication date. None of the focused hybrid’s first results matched the 24 reviewed X posts. All three archived positives remained below the production cosine floor.

### Suggested next steps

Boost related checks instead of excluding other results. Calibrate acceptance separately on more labeled positives.
