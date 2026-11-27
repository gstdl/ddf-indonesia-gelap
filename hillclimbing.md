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
