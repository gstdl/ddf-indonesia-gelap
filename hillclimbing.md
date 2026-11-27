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
