# Hillclimbing log

This log separates code changes from research decisions and records why an earlier direction was dropped. The baseline is `0f7a92e`, the implementation used for the 2,000-post sample run.

## Changed or retained

| Area | Decision | State |
| --- | --- | --- |
| Study scope | Study the August 2025 DPR protest wave on X. The July–October Brandwatch collection window is not itself the study corpus. | Recorded in the [README](README.md); corpus selection and fact-check queries have not changed in code. |
| Spam | Give spam its own post label. Report it separately from hoax and AI prevalence, but retain it for coordination analysis. | Documented in the README Mermaid diagram; not implemented. |
| Hoax evidence | Use similarity to retrieve candidate fact-checks, then verify that the post asserts the same debunked claim. Audit matches and misses. | Documented method; existing matcher unchanged. |
| AI-text evidence | Test on known-origin Indonesian social text with held-out topics and generators. Do not treat detector scores as authorship ground truth. | Documented method; existing detector unchanged. |
| Grok | Code the interaction in thread context using Mei et al.’s roles. Keep explicit Truth Arbiter verdicts as attributed signals, not independent verification. | Documented method; existing reply classifier unchanged. |
| Image OCR and captioning | Keep Florence-2-large as the baseline while comparing it with Base on the same August images. | No model switch in the pipeline. |

## Superseded or reverted

| Earlier direction | Disposition |
| --- | --- |
| Request a February–March #IndonesiaGelap export and keep that research question. | Superseded by the August DPR scope decision. The original rationale remains in the [progress deck](notebooks/progress_2026-11-26.py). |
| Hand-label about 100 posts as the default validation plan. | Superseded by targeted claim-match auditing and a programmatic AI-text benchmark with known provenance. Neither removes the need for in-domain validation. |
| Treat a Grok reply stance as stronger evidence merely because it came from Grok. | Superseded as a research rule by role and target checks; the baseline still gives Grok replies precedence in code. |

No code reversion is recorded in this pass. A superseded decision is not a reverted commit.

## Florence-2 Base vs. Large: faster, but not a drop-in replacement

One paired run on Apple M4 Pro (MPS), using the pipeline’s `<OCR>` and `<CAPTION>` prompts, three beams, and a 1,024-token cap. The 30 unique images were selected from August posts before inspecting either model’s output: 8 news cards, 8 posters or memes, 8 screenshots or documents, and 6 photos or low-text frames. Timings exclude loading and warmup.

| Measure | Base | Large |
| --- | ---: | ---: |
| OCR + caption total | 106.3 s | 209.8 s |
| Mean per image | 3.54 s | 6.99 s |
| OCR phrase checks, all | 50/57 | 49/57 |
| OCR phrase checks, screenshots and documents | 15/20 | 18/20 |
| OCR output with line breaks | 0/30 | 25/30 |

The phrase checks used selected text read from the images, ignoring punctuation and whitespace; they measure phrase recovery, not full transcription accuracy. They hide Base’s tendency to join words across lines. Large preserved more phrases in dense images, but both garbled some fine print and financial figures. The caption review was qualitative: neither model consistently described the images better.

**Initial decision: retain Large.** Base nearly halved inference time, but its merged word boundaries enter the [document text](pipeline/layer1_prepare/document.py) used downstream.

## Region-aware Base restores line breaks but misses more dense text than Large

Reran the same 30 images on the same hardware and generation settings with Base’s `<OCR_WITH_REGION>` prompt. Joined nonempty region labels with newlines in model-returned order; did not reorder by coordinates. Captioning was unchanged. Rescored the original outputs and the new run against the same 57 reference phrases. The boundary-aware score requires consecutive whole-word tokens and ignores case and punctuation; it does not measure downstream claim-matching accuracy.

| Measure | Base plain | Base region-aware | Large plain |
| --- | ---: | ---: | ---: |
| Mean OCR + caption, seconds/image | 3.54 | 4.91 | 6.99 |
| OCR outputs with line breaks | 0/30 | 25/30 | 25/30 |
| Phrases recovered, ignoring word boundaries | 50/57 | 47/57 | 49/57 |
| Phrases recovered, requiring word boundaries | 10/57 | 45/57 | 49/57 |
| Dense screenshots and documents, requiring word boundaries | 3/20 | 12/20 | 18/20 |

Region-aware Base took 147.2 s of inference, about 30% less than plain Large. It fixed word joins on posters such as “Bentrokan Mahasiswa,” but still missed claim text that Large recovered, such as the headline phrase “Moratorium Kunker ke Luar Negeri.”

**Decision: retain plain Large.** Region-aware Base was faster and largely addressed word joins, but recovered six fewer of the 20 dense-image phrases. Neither the model nor the prompt changed in the pipeline; no code was reverted.

## PaddleOCR recovered more claim text and was faster than plain Large

Ran PaddleOCR 3.7.0 with PaddlePaddle 3.3.1 on the same 30 images and the same 57 reference phrases. Used PP-OCRv5 mobile detection and Latin-script recognition (which supports Indonesian) on the Apple M4 Pro CPU with four threads. Disabled document orientation, unwarping, and text-line orientation. Joined text lines in model-returned order. Times exclude setup and warmup.

| Measure | PaddleOCR | Florence-2-large plain |
| --- | ---: | ---: |
| Mean OCR, seconds/image | 4.99 | 5.85 |
| Mean OCR + Large caption, seconds/image | 6.14 (estimated) | 6.99 (measured) |
| Phrases recovered, requiring word boundaries | 56/57 | 49/57 |
| Dense screenshots and documents, requiring word boundaries | 20/20 | 18/20 |

PaddleOCR correctly recovered both the Rp50,000,000 housing allowance and the Rp104,051,903 total from a dense pay chart. Its one scored miss was a headline broken up by unrelated text inserted between two parts of a name. The score measures selected phrase recovery, not full transcription accuracy. PaddleOCR does not generate captions; the combined estimate adds the measured Large caption times for the same images to the PaddleOCR times. The two engines were not run together in the pipeline.

**Decision: advance PaddleOCR with Large captioning to a pipeline trial.** It recovered more target text with an estimated 12% reduction in combined inference time. Check downstream claim matching and wall-clock speed on a larger slice before replacing Large OCR. No production dependency, model, or prompt changed.

## Held-out trial: PaddleOCR was faster after warmup, but did not improve claim matches

Sampled 60 August posts (Jakarta time) from the local 2,000-post run, one per perceptual hash, excluding the 30-image benchmark and its exact perceptual-hash matches. Ran the same PP-OCRv5 mobile detector and Latin recognizer on the M4 Pro CPU with four threads, and Florence-2-large on MPS. The two runs produced identical captions for all 60 images. Rebuilt documents with the pipeline’s text assembly and scored them with the current cosine matcher against the same 17,571 cached Komdigi and Google Fact Check claims.

| Measure | Large OCR + Large captions | PaddleOCR + Large captions |
| --- | ---: | ---: |
| OCR only, wall clock after warmup | 261.06 s | 226.45 s |
| OCR + captions, wall clock after warmup | 336.46 s | 299.21 s |
| OCR + captions, including the Paddle worker startup and warmup | 336.46 s | 335.58 s |
| Posts at or above the 0.70 similarity cutoff | 14/60 | 15/60 |
| Posts at or above the 0.85 claimed-false cutoff | 0/60 | 0/60 |

Florence loading and warmup are excluded from both columns. PaddleOCR cut warm processing time by 11%, but its separate worker erased nearly all of the saving in this run. The top fact-check URL changed for 14 posts; the median absolute change in top cosine score was 0.0089. At 0.70, two posts crossed up and one crossed down. A manual check of those three changes found that none of the top fact-checks addressed the same claim as its post. This is a check of the current matcher on the cached corpus, not a precision or recall estimate.

**Decision: keep Large in the pipeline for now.** PaddleOCR remains a candidate for better transcription, but this trial showed neither better claim matching nor a meaningful speedup including worker startup. Reevaluate after the fact-check corpus is scoped to the August DPR wave and the matcher checks whether the post asserts the same claim. No production code or dependencies changed.

## An assertion check rejects topic matches but misses valid paraphrases

Kept the current cosine retriever and the same 17,571 cached Komdigi and Google Fact Check claims. For each post with a top cosine score of at least 0.70, tested whether the post document entailed that candidate’s claim using the same multilingual mDeBERTa NLI checkpoint already used for reply classification. Used an entailment cutoff of 0.80, fixed before the run. The experiment checks whether the post asserts a claim; the existing fact-check supplies the verdict.

The controlled set used eight published August DPR claims, with four written variants per claim: a paraphrased assertion, a correction, a related but different statement, and a question or unconfirmed report. These 32 examples are a diagnostic, not a representative sample of live posts.

| Measure | Cosine ≥ 0.70 | Cosine ≥ 0.85 | Cosine ≥ 0.70 + entailment ≥ 0.80 |
| --- | ---: | ---: | ---: |
| Assertions linked to the intended claim | 5/8 | 1/8 | 4/8 |
| Non-assertions accepted | 16/24 | 3/24 | 0/24 |
| Real posts flagged on the earlier 60-image slice | 14/60 | 0/60 | 0/60 |

Two assertions did not rank the intended claim first; one ranked it first below 0.70. The verifier then rejected one of the five assertions that reached it with the right candidate: a paraphrase of the Soekarno-Hatta airport demonstration claim (entailment 0.65). The verifier took 8.94 seconds for 36 pairs on MPS, excluding model setup and tokenization. The real slice was not fully labeled, so its lower flag count does not establish higher precision or preserved recall.

**Decision: keep the assertion check experimental.** Test a top-k candidate list to address retrieval misses, then calibrate the verifier on independently labeled real post–claim pairs before changing the pipeline. No production code, cutoffs, or dependencies changed.

## Top-3 recovered one valid match and introduced one real mismatch

Reused the 32 controlled examples, 60 real posts, and 17,571 cached claims. Ranked candidate claim rows by cosine similarity and selected the highest-ranked candidate passing the same 0.80 entailment cutoff. Tested top 1, 3, 5, and 10 at the existing 0.70 cosine floor and a diagnostic 0.60 floor.

| Candidates per post at the 0.70 floor | Intended claims eligible for verification | Correct assertions accepted | Controlled non-assertions accepted | Real posts flagged | Pairs scored across both slices |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5/8 | 4/8 | 0/24 | 0/60 | 36 |
| 3 | 6/8 | 5/8 | 0/24 | 1/60 | 55 |
| 5 | 6/8 | 5/8 | 0/24 | 1/60 | 60 |
| 10 | 6/8 | 5/8 | 0/24 | 1/60 | 66 |

Top-3 recovered the asset-forfeiture claim at rank 2. It also linked a report about a condolence visit for Affan Kurniawan to an unrelated fact-check about the Sukoharjo regent allowing public celebrations. At the 0.60 floor, none of the tested list sizes recovered another valid assertion, and top-10 flagged two additional real posts whose selected checks also did not address their claims. All eight intended claims appeared somewhere in the unfiltered top 10, but two scored below 0.60. The widest test scored 502 pairs in 19.99 seconds on MPS, including tokenization and excluding model loading; pair counts reflect scoring the entire eligible shortlist.

**Decision: do not widen the production matcher.** Top-3 is the smallest tested list that recovered the valid match, but it introduced a real false match. Use it as a candidate configuration for evaluation on labeled real posts rather than increasing k or lowering the floor based on these fixtures. No production code or thresholds changed.

## On real posts, the verifier rejected mismatches but retrieval missed every known positive

Kept the 17,571-claim corpus and all cutoffs fixed. From 1,314 distinct August 2025 X documents in the local sample, drew 24 posts at random from the 301 with top cosine ≥ 0.70 after excluding the earlier 60-post trial. Reviewed the original post text and available OCR/captions against each of its eligible top-three claims before seeing verifier scores. None of the 50 candidate pairs asserted the same claim. This labels candidate pairs, not the posts’ overall veracity; a relevant check might exist outside the shortlist.

For a separate positive check, used three verbatim social-media captions quoted in Komdigi reviews about the protest wave: [Sahroni’s household employee](https://www.komdigi.go.id/berita/berita-hoaks/detail/hoaks-art-ahmad-sahroni-luka-parah-akibat-massa-menyerang-kediaman-ahmad-sahroni), [the UN dissolving the DPR](https://www.komdigi.go.id/berita/berita-hoaks/detail/hoaks-pbb-gelar-sidang-darurat-resmi-bubarkan-dpr-indonesia), and [a legislator’s car being damaged](https://www.komdigi.go.id/berita/berita-hoaks/detail/hoaks-mobil-anggota-dpr-dirusak-pendemo). These are archived excerpts, not full posts drawn from the X export.

| Frozen variant | Incorrect links on reviewed X posts | Correct links on archived positives |
| --- | ---: | ---: |
| Top-1 cosine ≥ 0.70 | 24/24 | 0/3 |
| Top-1 cosine ≥ 0.85 | 1/24 | 0/3 |
| Top-1 cosine ≥ 0.70 + entailment ≥ 0.80 | 0/24 | 0/3 |
| Top-3 cosine ≥ 0.70 + entailment ≥ 0.80 | 0/24 | 0/3 |

The intended positive checks ranked 3rd, 75th, and 3rd with cosine scores of 0.646, 0.477, and 0.556. None reached the verifier. The negative slice was selected from posts already above the loose cutoff, and the three positives were handpicked archived excerpts, so these are diagnostic counts, not population precision or recall.

**Decision: improve retrieval before tuning the verifier.** Compare an August DPR–focused fact-check corpus and lexical-plus-semantic retrieval on labeled real positives and hard negatives. Measure whether the right check enters the shortlist before adjusting entailment thresholds. No production settings changed.

## August-focused search improved ranking, but the hard filter dropped a relevant check

Compared the full 17,571-claim cache with an August-focused subset: false-rated reviews published August 1 through September 30, 2025 (Jakarta time), with a DPR, protest, or named-event term in the claim text. The subset had 73 claim rows: 70 Komdigi and 3 Google. This was a local filter, not a fresh API search. Compared the existing embedding cosine, keyword TF-IDF over words and adjacent word pairs, and equal-weight reciprocal rank fusion of both rankings. The test measured candidate rank without an acceptance cutoff or assertion verifier.

| Corpus and ranking | Archived positives at rank 1 | Archived positives in top 3 | Archived positives in top 10 | Controlled assertions in top 10 |
| --- | ---: | ---: | ---: | ---: |
| Full · semantic | 0/3 | 2/3 | 2/3 | 8/8 |
| Full · keyword | 2/3 | 2/3 | 2/3 | 8/8 |
| Full · hybrid | 1/3 | 2/3 | 3/3 | 8/8 |
| Focused · semantic | 2/3 | 3/3 | 3/3 | 7/8 |
| Focused · keyword | 3/3 | 3/3 | 3/3 | 7/8 |
| Focused · hybrid | 3/3 | 3/3 | 3/3 | 7/8 |

Focused keyword and hybrid both moved all three archived positives to rank 1. The focus rule also dropped the check about Prabowo meeting student protesters: its date qualified, but `berdemonstrasi` did not match the whole-word term list. Undated reviews could not pass the filter; 410 of 486 cached Google review rows lack a publication date. The archived targets’ cosine scores remained below 0.70 in both corpora, so the current cosine gate would still block them.

None of the focused hybrid’s top-ranked candidates for the same 24 reviewed X posts asserted the same claim. For example, a screenshot about the police chief’s apology for the Brimob vehicle collision retrieved a fact-check about a Brimob sniper. A higher candidate rank is not a verdict. The archived positives and controlled examples are small, reused diagnostic sets, not a new held-out estimate.

**Decision: keep full-corpus hybrid as the next experimental baseline.** It recovered all three archived positives in the top 10 without excluding a known check. Test August focus as a ranking boost rather than a hard filter, and calibrate acceptance separately on more labeled real positives. No production settings changed.

## A 10% August boost improved shortlists without dropping any claims

Kept all 17,571 claims and reused the full-corpus keyword + semantic ranking. Multiplied the fused rank score for the prior August-focused subset’s 73 rows by 1.10, 1.25, or 1.50, leaving every other score unchanged. Reused the same three archived positives, eight controlled assertions, and 24 reviewed X posts. No cosine cutoff or assertion verifier was applied in this ranking test.

| August score boost | Archived positives at rank 1 | Archived positives in top 3 | Controlled assertions at rank 1 | Controlled assertions in top 3 | Top result changed on reviewed X posts |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 1/3 | 2/3 | 7/8 | 8/8 | 0/24 |
| +10% | 2/3 | 3/3 | 8/8 | 8/8 | 2/24 |
| +25% | 2/3 | 3/3 | 7/8 | 8/8 | 5/24 |
| +50% | 3/3 | 3/3 | 7/8 | 7/8 | 13/24 |

All variants kept all known positives in the top 10. The +10% boost moved the archived targets from ranks 1, 4, and 2 to 1, 3, and 1. The +50% boost put all three archived targets first but pushed the Prabowo–student-protesters check, which was not eligible for the boost, from rank 1 to rank 4. None of the changed top results matched the claims in the reviewed X posts; these are candidate rankings, not accepted labels.

**Decision: carry +10% forward as an experimental setting only.** It was the smallest tested boost that put all three archived positives in the top three while retaining top-three coverage on the controlled cases. Check it against unboosted hybrid on new labeled posts before adopting it. No production settings changed.
