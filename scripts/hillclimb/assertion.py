"""Compare the production cosine cutoffs against a top-1 assertion verifier."""

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.hf import DEVICE, embed
from pipeline.layer1_prepare.document import build_document
from pipeline.layer1_prepare.image import join_texts
from pipeline.layer2_label.track1_hoax.claimed_false import CLAIMED_FALSE_SIMILARITY, false_claims
from pipeline.layer2_label.track1_hoax.similar_to_known_hoax import KNOWN_HOAX_SIMILARITY
from pipeline.layer2_label.track2_ai_generated.classify_replies import NLI_MODEL, NLI_MODEL_REVISION

ENTAILMENT_CUTOFF = 0.80
CASE_KINDS = ("assertion", "correction", "same_topic", "uncertain")


def load_claim_sources(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(data_dir) / "layer2_label" / "track1_hoax"
    return pd.read_parquet(source / "komdigi.parquet"), pd.read_parquet(source / "google_fact_check.parquet")


def read_trial_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def load_trial(data_dir: Path, trial: Path) -> pd.DataFrame:
    families = json.loads(Path(__file__).with_name("claim_cases.json").read_text(encoding="utf-8"))
    controlled = pd.DataFrame(
        {"kind": kind, "target": claim, "document": text}
        for claim, *texts in families
        for kind, text in zip(CASE_KINDS, texts, strict=True)
    )
    media = pd.DataFrame(read_trial_rows(trial)).astype({"post_id": str})
    posts = pd.read_parquet(data_dir / "layer1_prepare" / "image.parquet", columns=["post_id", "text"])
    posts = posts.astype({"post_id": str})
    real = media.merge(posts, on="post_id", how="left", sort=False, validate="one_to_one", indicator=True)
    if not real["_merge"].eq("both").all():
        raise ValueError("Trial post IDs must be present in the image parquet from the same snapshot")
    real["media_text"] = [join_texts(ocr or None, caption or None) for ocr, caption in zip(real.ocr, real.caption, strict=True)]
    real_docs = build_document(real)
    return pd.concat([controlled, real_docs[["document"]].assign(kind="real")], ignore_index=True)


def top_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def rank_trial(local: pd.DataFrame, claims: pd.DataFrame, k: int) -> pd.DataFrame:
    if not set(local.target.dropna()) <= set(claims.claim_text):
        raise ValueError("A controlled claim is missing from the cached corpus")
    similarity = embed(local.document.tolist()) @ embed(claims.claim_text.tolist()).T
    top = top_indices(similarity, k)
    width = top.shape[1]
    ranked = pd.DataFrame({
        "post": np.repeat(local.index, width),
        "rank": np.tile(np.arange(1, width + 1), len(local)),
        "candidate": claims.claim_text.to_numpy()[top.ravel()],
        "cosine": np.take_along_axis(similarity, top, axis=1).ravel(),
    }).join(local, on="post")
    return ranked.assign(correct=ranked.candidate.eq(ranked.target))


def clean_claim(text: str) -> str:
    return re.sub(r"^\s*\[(?:salah|hoaks|hoax|misleading|false)\]\s*", "", text, flags=re.I).removeprefix("Video ").strip()


def entailment_scores(pairs: pd.DataFrame) -> tuple[list[float], float, float]:
    """Score document/candidate pairs; return probabilities, forward seconds and total scored seconds."""
    if pairs.empty:
        return [], 0.0, 0.0
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, revision=NLI_MODEL_REVISION,
    ).to(DEVICE).eval()
    if model.config.id2label[0].lower() != "entailment":
        raise ValueError("Expected entailment at index 0 in the pinned NLI checkpoint")
    probabilities = []
    forward_seconds = 0.0
    start = perf_counter()
    with torch.inference_mode():
        for i in range(0, len(pairs), 8):
            batch = pairs.iloc[i:i + 8]
            inputs = tokenizer(
                batch.document.tolist(), batch.candidate.map(clean_claim).tolist(), truncation="only_first",
                max_length=512, padding=True, return_tensors="pt",
            ).to(DEVICE)
            forward_start = perf_counter()
            probabilities.extend(model(**inputs).logits.float().softmax(-1)[:, 0].cpu().tolist())
            forward_seconds += perf_counter() - forward_start
    return probabilities, forward_seconds, perf_counter() - start


def verify_ranked(ranked: pd.DataFrame, floor: float) -> tuple[int, float, float]:
    eligible = ranked.loc[ranked.cosine.ge(floor)]
    probabilities, forward_seconds, total_seconds = entailment_scores(eligible)
    ranked["entailment"] = np.nan
    ranked.loc[eligible.index, "entailment"] = probabilities
    return len(eligible), forward_seconds, total_seconds


def selected_counts(selected: pd.DataFrame) -> dict:
    assertions = selected.loc[selected.kind.eq("assertion")]
    return {
        "correct_assertions": int(assertions.correct.sum()),
        "wrong_assertion_links": int((~assertions.correct).sum()),
        "nonassertions_accepted": int((~selected.kind.isin(["assertion", "real"])).sum()),
        "real_flagged": int(selected.kind.eq("real").sum()),
    }


def trial_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", required=True, type=Path, help="Pipeline data root")
    parser.add_argument("--trial", required=True, type=Path, help="Local trial-large.json")
    parser.add_argument("--output", required=True, type=Path, help="Local results JSONL (contains private text)")
    return parser


def main() -> None:
    args = trial_parser(__doc__).parse_args()
    claims = false_claims(load_claim_sources(args.data_root))
    local = load_trial(args.data_root, args.trial)
    ranked = rank_trial(local, claims, k=1)
    pairs, forward_seconds, _ = verify_ranked(ranked, KNOWN_HOAX_SIMILARITY)
    loose = ranked.cosine.ge(KNOWN_HOAX_SIMILARITY)
    strict = ranked.cosine.ge(CLAIMED_FALSE_SIMILARITY)
    ranked["gate"] = loose & ranked.entailment.ge(ENTAILMENT_CUTOFF)
    print(f"Claims: {len(claims)}; documents by kind: {local.kind.value_counts(sort=False).to_dict()}")
    print(f"NLI seconds (excluding model setup and tokenization): {forward_seconds:.2f}; pairs: {pairs}")
    for label, mask in [("cosine .70", loose), ("cosine .85", strict), ("cosine .70 + NLI .80", ranked.gate)]:
        print(label, selected_counts(ranked.loc[mask]))
    ranked.to_json(args.output, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
