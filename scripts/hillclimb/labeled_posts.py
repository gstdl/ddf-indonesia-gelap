"""Build the local August review pool and evaluate frozen claim-matching variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.hf import embed
from pipeline.layer2_label.track1_hoax.claimed_false import CLAIMED_FALSE_SIMILARITY, false_claims
from pipeline.layer2_label.track1_hoax.similar_to_known_hoax import KNOWN_HOAX_SIMILARITY

from .assertion import ENTAILMENT_CUTOFF, entailment_scores, load_claim_sources, read_trial_rows, top_indices

POST_COLUMNS = ["post_id", "text", "media_text", "document", "created_at"]
EVENT_TERMS = r"\bDPR\b|Affan|Sahroni|demonstrasi|\bdemo\b|Pejompongan|\bmassa\b|penjarahan|\bojol\b"
POSTS_FILE = "real-claims-posts.parquet"
CORPUS_FILE = "real-claims-corpus.parquet"
RANKED_FILE = "real-claims-ranked.npz"


def load_august_posts(data_dir: Path) -> pd.DataFrame:
    posts = pd.read_parquet(Path(data_dir) / "layer1_prepare" / "link.parquet", columns=POST_COLUMNS)
    posts["created_at"] = posts.created_at.dt.tz_convert("Asia/Jakarta")
    days = posts.created_at.dt.strftime("%Y-%m-%d")
    return posts.loc[days.between("2025-08-01", "2025-08-31")].drop_duplicates("document").reset_index(drop=True)


def read_labels(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_rows(labels: dict, idx: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = (
        (post, rank, int(claim), float(score), None if positives is None else int(claim) in positives)
        for post, positives in labels.items()
        for rank, (claim, score) in enumerate(zip(idx[post], scores[post], strict=True), 1)
        if score >= KNOWN_HOAX_SIMILARITY
    )
    return pd.DataFrame(rows, columns=["post", "rank", "claim", "cosine", "correct"])


def write_proposals(output: Path, posts: pd.DataFrame, kom: pd.DataFrame, claims: pd.DataFrame,
                    similarity: np.ndarray, top: np.ndarray) -> None:
    known = kom.loc[kom.published_at.dt.strftime("%Y-%m-%d").between("2025-08-23", "2025-09-20")]
    known = known.loc[known.claim_text.str.contains(EVENT_TERMS, case=False, na=False), ["claim_text", "source_url"]]
    seed_ids = claims.reset_index().merge(known.drop_duplicates(), on=["claim_text", "source_url"])["index"]
    print(f"August event seeds: {len(seed_ids)}")
    rows = []
    for claim in seed_ids:
        scores = similarity[:, claim]
        indices = np.argsort(-scores)[:2]
        if scores[indices[0]] >= 0.65:
            rows.extend(("reverse_seed", post, claim, scores[post]) for post in indices)
    for post in np.argsort(-similarity.max(axis=1))[:32]:
        claim = top[post, 0]
        rows.append(("forward_top", post, claim, similarity[post, claim]))
    proposals = pd.DataFrame(rows, columns=["method", "post", "claim", "cosine"])
    attach_pair_text(proposals, posts, claims).to_json(output, orient="records", lines=True, force_ascii=False)


def attach_pair_text(rows: pd.DataFrame, posts: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    post_detail = posts.loc[:, ["post_id", "text", "media_text", "document"]]
    claim_detail = claims.rename(columns={"claim_text": "candidate"})
    return rows.join(post_detail, on="post").join(claim_detail, on="claim")


def build_pool(args: argparse.Namespace) -> None:
    kom, google = load_claim_sources(args.data_root)
    claims = false_claims((kom, google))
    posts = load_august_posts(args.data_root)
    similarity = embed(posts.document.tolist()) @ embed(claims.claim_text.tolist()).T
    top = top_indices(similarity, 3)
    top_scores = np.take_along_axis(similarity, top, axis=1)
    trial_ids = {str(row["post_id"]) for row in read_trial_rows(args.trial)}
    eligible = np.flatnonzero((top_scores[:, 0] >= KNOWN_HOAX_SIMILARITY) & ~posts.post_id.astype(str).isin(trial_ids))
    labels = dict.fromkeys(read_labels(args.labels)["x_review"], set())
    if not set(labels).issubset(eligible):
        raise ValueError("Reviewed X rows must belong to the eligible pool from the original snapshots")
    candidates = candidate_rows(labels, top, top_scores)
    print(f"August documents: {len(posts)}; claims: {len(claims)}; eligible review pool: {len(eligible)}")
    print(f"Review documents: {len(labels)}; eligible audit pairs: {len(candidates)}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    posts.to_parquet(output / POSTS_FILE)
    claims.to_parquet(output / CORPUS_FILE)
    np.savez(output / RANKED_FILE, top_idx=top, top_scores=top_scores)
    write_proposals(output / "real-claims-proposals.jsonl", posts, kom, claims, similarity, top)
    attach_pair_text(candidates, posts, claims).to_json(
        output / "real-claims-review.jsonl", orient="records", lines=True, force_ascii=False,
    )


def evaluate(args: argparse.Namespace) -> None:
    posts = pd.read_parquet(args.pool_dir / POSTS_FILE)
    claims = pd.read_parquet(args.pool_dir / CORPUS_FILE)
    review = read_labels(args.labels)
    labels = dict.fromkeys(review["x_review"], set())
    if not all(0 <= post < len(posts) for post in labels):
        raise ValueError("Reviewed X index is out of bounds for the cached August posts")
    targets, quotes = zip(*review["archived_positives"], strict=True)
    if not all(0 <= target < len(claims) for target in targets):
        raise ValueError("Archived target index is out of bounds for the sorted claim corpus")
    quote_scores = embed(list(quotes)) @ embed(claims.claim_text.tolist()).T
    quote_idx = top_indices(quote_scores, 3)
    for offset, target in enumerate(targets):
        labels[len(posts) + offset] = {target}
        row = quote_scores[offset]
        rank = int((row > row[target]).sum()) + 1
        print(f"Archived intended claim {target}: {claims.claim_text.iloc[target]}; rank={rank}; cosine={row[target]:.4f}")
    with np.load(args.pool_dir / RANKED_FILE) as ranked:
        idx = np.concatenate([ranked["top_idx"], quote_idx])
        scores = np.concatenate([ranked["top_scores"], np.take_along_axis(quote_scores, quote_idx, axis=1)])
    candidates = candidate_rows(labels, idx, scores)
    documents = np.asarray([*posts.document, *quotes], dtype=object)
    pairs = candidates.assign(document=documents[candidates.post.to_numpy(dtype=int)],
                              candidate=claims.claim_text.to_numpy()[candidates.claim.to_numpy(dtype=int)])
    candidates["entailment"], _, _ = entailment_scores(pairs)
    first = candidates["rank"].eq(1)
    verified = candidates.entailment.ge(ENTAILMENT_CUTOFF)
    archived = candidates.post.ge(len(posts))
    print(f"Labeled documents: {len(labels)}; archived positives: {len(targets)}; "
          f"eligible top-3 pairs: {len(candidates)}; eligible correct pairs: {int(candidates.correct.sum())}")
    for name, keep in (
        ("cosine .70", first), ("cosine .85", first & candidates.cosine.ge(CLAIMED_FALSE_SIMILARITY)),
        ("top-1 NLI", first & verified), ("top-3 NLI", verified),
    ):
        picked = candidates.loc[keep].drop_duplicates("post")
        correct = int(picked.correct.sum())
        print(f"{name}: flags={len(picked)}; correct={correct}; incorrect={len(picked) - correct}; "
              f"reviewed X mismatches={int((~archived.loc[picked.index]).sum())}; "
              f"archived positives missed={len(targets) - correct}")
    candidates.to_json(args.output, orient="records", lines=True, force_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pool = commands.add_parser("pool", help="Build ranked pool and write the local, pre-NLI audit")
    pool.add_argument("--data-root", required=True, type=Path, help="Pipeline data root")
    pool.add_argument("--trial", required=True, type=Path, help="Exclude post IDs in this local trial-large.json")
    pool.add_argument("--output-dir", required=True, type=Path, help="Local directory for private experiment files")
    pool.add_argument("--labels", required=True, type=Path, help="Use the original positional audit labels")
    evaluation = commands.add_parser("evaluate", help="Score annotated candidates and archived positive quotes")
    evaluation.add_argument("--pool-dir", required=True, type=Path, help="Output directory from the pool command")
    evaluation.add_argument("--labels", required=True, type=Path, help="Local real-claims-labels.json from the same snapshots")
    evaluation.add_argument("--output", required=True, type=Path, help="Local candidate metrics JSONL")
    args = parser.parse_args()
    if args.command == "pool":
        build_pool(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
