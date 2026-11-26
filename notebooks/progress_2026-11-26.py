# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", layout_file="layouts/progress_2026-11-26.slides.json")


@app.cell(hide_code=True)
def _():
    import re
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data"
    EXPORT = DATA / "raw" / "brandwatch_export_indonesia_gelap.csv"
    return DATA, EXPORT, alt, mo, pd, re


@app.cell(hide_code=True)
def _(EXPORT, pd):
    export_rows = pd.read_csv(EXPORT, header=5, usecols=["Content Source Name", "Date", "Engagement Type"], dtype=str)
    export_rows["Date"] = pd.to_datetime(export_rows["Date"])
    return (export_rows,)


@app.cell(hide_code=True)
def _(DATA, pd):
    media = pd.read_parquet(DATA / "layer1_prepare" / "download_media.parquet",
                            columns=["post_id", "brandwatch_media_type", "media_url", "media_path", "media_type"])
    merged = pd.read_parquet(DATA / "layer3_analyze" / "merge.parquet")
    text_votes = pd.read_parquet(DATA / "layer2_label" / "track2_ai_generated" / "text_detector.parquet")
    return media, merged, text_votes


@app.cell(hide_code=True)
def _(DATA, re):
    done = re.compile(r"done  (\S+): (\d+) rows x \d+ columns in ([\d.]+)s")
    runs = {}
    for line in (DATA / "pipeline.log").read_text().splitlines():
        match = done.search(line)
        if match:
            runs[match.group(1)] = (int(match.group(2)), float(match.group(3)))
    image_rows, image_seconds = runs["layer1_prepare/image"]
    # Full data: ~17,000 analysable images; everything but OCR scales from the sample to ~3.5 h.
    ocr_hours = image_seconds / image_rows * 17000 / 3600
    other_hours = 3.5
    return image_rows, image_seconds, ocr_hours, other_hours, runs


@app.cell(hide_code=True)
def _(mo):
    def tracker(active: str):
        """Section tracker shown under each slide title: where this slide sits in the storyline."""
        sections = ["A · Built", "B · Data", "C · Labels", "D · Full run"]
        return mo.md(" &nbsp;›&nbsp; ".join(f"**{s}**" if s.startswith(active) else s for s in sections))

    return (tracker,)


@app.cell(hide_code=True)
def _(export_rows, merged, mo, ocr_hours, other_hours):
    x_posts = int((export_rows["Content Source Name"] == "X").sum())
    mo.md(f"""
    # We can run the full {x_posts:,} posts within days — but only after fixing the data scope and the labels

    **#IndonesiaGelap AI-generated hoax pipeline · progress update · 26 Nov 2026**

    | | Finding | Ask |
    |---|---|---|
    | **A · Built** | Every Miro box is code; the pipeline ran end to end on a 2,000-post sample | — |
    | **B · Data** | The export is a different protest wave and lacks reposts, video audio and 1 of 3 fact-check sources | Re-scope or re-export |
    | **C · Labels** | {int((merged["hoax_label"] == "hoax").sum())} "hoax" and {int((merged["ai_label"] == "ai_generated").sum())} "AI" labels in the sample are mostly false positives; spam and Grok replies were never planned | Calibrate; decide on spam and Grok |
    | **D · Full run** | ~{ocr_hours + other_hours:.0f} h with today's OCR model; ~{ocr_hours / 3 + other_hours:.0f} h with a lighter one | Approve model |
    """)
    return


@app.cell(hide_code=True)
def _(mo, tracker):
    mo.vstack([
        mo.md("## A · The plan is built: one Miro box = one code file, and it ran end to end on real data"),
        tracker("A"),
        mo.md(r"""
    | Layer (Miro overview) | Boxes in code | Status |
    |---|---|---|
    | 1 · Collect and prepare posts | export, X filter*, download media, normalize, text / image / video, document, cluster, link | ran |
    | 2 · Label each post six ways | 6 tracks, 17 boxes; every detector a 3-model ensemble with a 2-of-3 quorum | ran |
    | 3 · Merge and analyze networks | merge, network analysis, RQ1a / RQ1b / RQ1c tables | ran |
    | 4 · Ethics review, then publish | anonymize (salted hashes; URLs and text dropped) | ran; ethics review is human |

    \* not on the board: Brandwatch also returns YouTube, Facebook and Instagram.
    Runs are atomic and resumable; 74 tests; 37 method papers cited.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, tracker):
    mo.vstack([
        mo.md("## B · The data answers a different question than the RQ asks: it is the Aug 2025 DPR protests, not #IndonesiaGelap"),
        tracker("B"),
        mo.md(r"""
    - The export is brand **"Fellowship Kanzia Gusti – Hoaxes Transformation"**, **31 Jul – 31 Oct 2025**
    - Its posts are about DPR pay and allowances, `#BubarkanDPR`, the 25 August demonstration and Affan Kurniawan
    - The #IndonesiaGelap wave and Komdigi's clarifications of it (USAID funding, paid protesters) are **Feb–Mar 2025**

    **So what:** results on this export cannot be reported as #IndonesiaGelap findings.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(alt, export_rows, media, mo, pd, tracker):
    x_rows = export_rows[export_rows["Content Source Name"] == "X"]
    listed = media[media["brandwatch_media_type"].isin(["photo", "video"])]
    videos = listed[listed["brandwatch_media_type"] == "video"]
    gaps = pd.DataFrame({
        "signal": ["Reposts (track 4: coordination)", "Video audio (track 2: deepfake speech)",
                   "Fact-check sources (track 1)"],
        "available": [0, 0, 2 / 3],
    })
    gaps_chart = (
        alt.Chart(gaps)
        .mark_bar()
        .encode(
            x=alt.X("available:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="share available"),
            y=alt.Y("signal:N", sort=None, title=None),
        )
        .properties(width=520, height=170, title="Three signals the Miro plan relies on are missing or partial in the export")
    )
    mo.vstack([
        mo.md("## B · The export also lacks three signals the plan relies on"),
        tracker("B"),
        gaps_chart,
        mo.md(f"""
    - **Reposts: 0** of {len(x_rows):,} X posts — the buzzer track's interaction network has no retweet edges
    - **Video audio: none** — {len(videos)} sampled video posts carry only a thumbnail, so speech-to-text and the 3 audio deepfake models never run
    - **Fact-checks: 2 of 3 sources** — Komdigi and Google Fact Check work; CekFakta is unreachable. Google's queries target Feb 2025 #IndonesiaGelap terms, so they need re-scoping too
    """),
    ])
    return


@app.cell(hide_code=True)
def _(alt, merged, mo, pd, tracker):
    label_rows = []
    for column, title in [("hoax_label", "Hoax"), ("ai_label", "AI-generated"), ("buzzer_label", "Buzzer")]:
        label_rows += [{"label": title, "value": value, "posts": int(n)} for value, n in merged[column].value_counts().items()]
    labels_chart = (
        alt.Chart(pd.DataFrame(label_rows))
        .mark_bar()
        .encode(
            x=alt.X("posts:Q", stack="normalize", axis=alt.Axis(format="%"), title="share of the 2,000 sampled posts"),
            y=alt.Y("label:N", title=None),
            color=alt.Color("value:N", title=None),
            tooltip=["label", "value", "posts"],
        )
        .properties(width=520, height=140, title="Labels on the 2,000-post sample")
    )
    mo.vstack([
        mo.md("## C · The labels are not valid yet: hoax and AI labels are mostly false positives, and spam is undefined"),
        tracker("C"),
        labels_chart,
        mo.md("""
    Four causes, one per following slide: **hoax matching** is too loose · **AI-text detection** is unvalidated on Indonesian · **spam** and **Grok replies** have no place in the plan.
    Buzzer and reply labels need whole threads, so the random sample cannot judge them.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(merged, mo, tracker):
    claimed = merged[merged["claimed_false"]][["full_text", "matched_fact_check_url"]].reset_index(drop=True)
    claimed = claimed.assign(
        full_text=claimed["full_text"].str.slice(0, 110),
        matched_fact_check_url=claimed["matched_fact_check_url"].str.replace(
            "https://www.komdigi.go.id/berita/berita-hoaks/detail/", "", regex=False),
    ).rename(columns={"full_text": "post", "matched_fact_check_url": "matched Komdigi item"})
    mo.vstack([
        mo.md("## C1 · Hoax matching finds posts that look like a hoax title, not posts that repeat the claim"),
        tracker("C"),
        mo.md(f"""
    - "Claimed false" (similarity ≥ 0.85): **{len(claimed)} posts, about 1 plausible** — e.g. a mourning post matched a Qur'an hoax
    - "Similar to a known hoax" (≥ 0.70): **{int(merged["similar_to_known_hoax"].sum())} posts ({merged["similar_to_known_hoax"].mean():.0%})** — on-topic, but not hoaxes
    """),
        mo.ui.table(claimed, selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(merged, mo, text_votes, tracker):
    flagged = text_votes[text_votes["media_signal"] == "synthetic"].merge(merged[["post_id", "full_text"]], on="post_id")
    examples = flagged.assign(post=flagged["full_text"].str.slice(0, 120))[["post"]].head(6).reset_index(drop=True)
    mo.vstack([
        mo.md("## C2 · The AI-text detectors flag news headlines and trend-bait, because none is validated on Indonesian tweets"),
        tracker("C"),
        mo.md(f"""
    - **{len(flagged)} of {len(text_votes)}** text posts voted AI by 2 of 3 models
    - One of the three models is a German academic-writing detector; no Indonesian benchmark exists for the others
    """),
        mo.ui.table(examples, selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, tracker):
    mo.vstack([
        mo.md("## C3 · Spam was never part of the plan, yet it is both noise and evidence"),
        tracker("C"),
        mo.md(r"""
    The Miro board has no spam step. The sample has trend-hijacking spam riding protest hashtags:

    > *"Guling tisu bubarkan DPR sontoloyo … trade \$AIO … WTS jasa buzzer exha #zonajajan #zonauang #jasabuzzer"*

    - **Noise:** inflates volume, trips the AI-text detectors, mimics coordination in the buzzer score
    - **Evidence:** a post *selling buzzer services* is direct evidence for RQ1b — filtering it out would remove the signal we study
    """),
    ])
    return


@app.cell(hide_code=True)
def _(DATA, mo, pd, re, tracker):
    replies = pd.read_parquet(DATA / "layer2_label" / "track2_ai_generated" / "collect_replies.parquet",
                              columns=["post_id", "parent_post_id", "author", "full_text"])
    grok = replies[replies["author"].str.lower() == "grok"]
    stances = pd.read_parquet(DATA / "layer2_label" / "track2_ai_generated" / "classify_replies.parquet")
    grok_stances = stances[stances["reply_source"] == "grok"]
    # Grok replies that talk about media authenticity: the Truth Arbiter role on media (Mei et al. 2026).
    authenticity_terms = r"\b(?:AI|deepfake|palsu|manipulasi|rekayasa|editan)\b"
    verdicts = grok[grok["full_text"].str.contains(authenticity_terms, regex=True)]
    excerpts = []
    for text in verdicts["full_text"].tail(3):
        start = re.search(authenticity_terms, text).start()
        excerpts.append("- *…" + " ".join(text[max(0, start - 70): start + 90].split()) + "…*")
    mo.vstack([
        mo.md(f"## C4 · Grok writes {len(grok) / len(replies):.0%} of replies; a few are real AI-detection evidence, most are not verdicts"),
        tracker("C"),
        mo.md(f"""
    - **{len(grok)} of {len(replies)}** sampled replies are Grok. Per **Mei et al. (2026, *Grok in the Wild*)** only its **Truth Arbiter** role judges veracity; most of ours are **Information Provider** answers, yet our classifier read {int((grok_stances["reply_signal"] == "real").sum())} as "authentic" verdicts
    - Only {len(verdicts)} Grok replies judge whether media is real or AI-made — and they flag real AI hoaxes (a deepfake Sahroni video, an AI photo of Sahroni):
    """),
        mo.md("\n".join(excerpts)),
        mo.md("**Proposal:** code @grok exchanges into Mei et al.'s roles (LLM coder validated on human codes, κ = .60); use only Truth Arbiter verdicts. Open: is Grok's own output in scope for RQ1?"),
    ])
    return


@app.cell(hide_code=True)
def _(alt, image_rows, image_seconds, mo, ocr_hours, other_hours, pd, runs, tracker):
    timing = pd.DataFrame([(box, seconds / 60) for box, (rows, seconds) in runs.items()],
                          columns=["box", "minutes"]).sort_values("minutes", ascending=False).head(5)
    timing_chart = (
        alt.Chart(timing)
        .mark_bar()
        .encode(x=alt.X("minutes:Q", title="minutes on the 2,000-post sample"), y=alt.Y("box:N", sort="-x", title=None))
        .properties(width=520, height=170, title="Image OCR and captioning take most of the run time")
    )
    mo.vstack([
        mo.md(f"## D · The full run takes ~{ocr_hours + other_hours:.0f} h, mostly image OCR; a lighter model cuts it to ~{ocr_hours / 3 + other_hours:.0f} h"),
        tracker("D"),
        timing_chart,
        mo.md(f"""
    - OCR + caption: {image_seconds / image_rows:.1f} s per image × ~17,000 analysable images in the full data
    - Everything else: roughly 3–4 h
    - Runs resume where they stopped, so a multi-day run is safe to interrupt
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decisions for today: one per pillar, in the order they block the full run

    | Pillar | Decision | Recommendation |
    |---|---|---|
    | **B · Data** | Re-scope the RQ to the Aug–Oct 2025 DPR protests, or request a Feb–Mar 2025 #IndonesiaGelap export | Request the export; it also tests whether Brandwatch can include reposts |
    | **C · Labels** | How to treat spam | Label it as its own category: keeps the buzzer evidence, stops it polluting other labels |
    | **C · Labels** | How to validate hoax and AI labels | Hand-label ~100 posts; calibrate the hoax cutoffs; report text AI detection as weak evidence |
    | **C · Labels** | How to use Grok replies | Code @grok exchanges into Mei et al.'s roles; keep only Truth Arbiter verdicts; decide whether Grok's own output is in RQ1 |
    | **D · Full run** | OCR model | Florence-2-base for the full run |

    **Next:** re-scope the Google Fact Check queries · hand-labelling sheet for C · rerun the sample, then the full data
    """)
    return


if __name__ == "__main__":
    app.run()
